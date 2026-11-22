"""Fast guitar articulation from the mixed signal, every analysis frame (20 ms).

It does not find guitar: the recognition model does that, slowly. This follows the
articulation of the part of a mix where guitars usually are, so that there is something
truthful to animate while the recognition model says guitar is playing.

**Why the stereo difference.** In most rock, metal and pop mixes the rhythm guitars are
panned to the sides, often as two separate takes, while kick, snare, bass and lead vocal
sit in the centre. Left minus right removes what is identical in both channels. Measured on
30 MoisesDB songs (200 Hz to 5 kHz), the guitar stem holds a median 62% of the energy of
that difference signal against 27% of the mono mix, and attacks found in it coincide with
drum-stem attacks less often. No spectral measure of the mono mix that was tried (plain
and peak-based spectral flux, band-limited, with drum vetoes) follows the guitar better
than it follows the drums; see ``docs/INSTRUMENT_LAYERS.md`` for the numbers.

**What is computed.** The rise of the log spectrum of the difference signal in the guitar
band. From that: attacks, how much is ringing, where in the
band the articulation sat and how wide it was, and whether it repeats and how fast.

**What it cannot do.** Tell a guitar from anything else that is panned and articulated:
keyboards, backing vocals, stereo drum overheads, a wide synthesizer. Taking the guitar
out of a mix leaves this signal about as busy as before. Whether any of it is guitar is
the recognition model's judgement (``gmv.instruments.combine``). A mono recording, or one
with everything in the centre, has no difference signal; then the mono band is used, counting
only rises concentrated in a few partials (a drum hit raises every bin alike). That is
weaker, and ``basis`` says so. Nothing here is an isolated
guitar signal.
"""

import numpy as np
from scipy.ndimage import maximum_filter1d

from . import Activity

METHOD = "guitar_articulation_v1"
LOW_HZ, HIGH_HZ = 200.0, 5000.0  # above bass fundamentals and kick, below cymbal wash
HISTORY_S = 1.28  # how much articulation the repetition estimate looks at
SLOWEST_HZ, FASTEST_HZ = 2.0, 17.0  # repetition rates that are reported
STEREO_DB = -26.0  # difference signal this far below the sum: treated as a mono recording
FLAT = 0.45  # mono fallback: a strong rise spread this evenly is a broadband hit
SPARSE = (0.55, 0.85)  # share of the centre's rise in the top fifth of bins: flat .. plucked
FLUX_SCALE = 0.11  # mean log rise per bin that maps to a strength of 0.63


def threshold_scale(sensitivity):
    """Sensitivity 0..1 (0.5 is the tuned default) as a factor on every threshold:
    1 at the default, 2 at the least and 0.5 at the most sensitive setting."""
    return float(2.0 ** (1.0 - 2.0 * min(1.0, max(0.0, sensitivity))))


class Rise:
    """Positive change of a log magnitude spectrum, with a level floor that follows the
    recent loudness so that passages far below it are not stretched up."""

    def __init__(self, hop):
        self.decay = float(np.exp(-hop / 4.0))
        self.reference, self.previous = 0.0, None

    def step(self, magnitude):
        mean = float(magnitude.mean())
        self.reference = max(mean, self.reference * self.decay)
        log = np.log(magnitude + 0.1 * self.reference + 1e-12)
        if self.previous is None:
            rise = np.zeros(len(magnitude))
        else:
            # A partial that moved to the next bin (vibrato, a bend) is not a new attack.
            rise = np.maximum(log - maximum_filter1d(self.previous, 3, mode="nearest"), 0.0)
        self.previous = log
        loud = 1.0 + np.log10(max(mean, 1e-12) / max(self.reference, 1e-12)) / 1.5
        return rise, mean, min(1.0, max(0.0, loud))


class GuitarArticulation:
    key = "guitar"
    method = METHOD
    delay_frames = 1  # an attack is a local maximum: known one frame later

    def __init__(self, rate, size, hop_s, sensitivity=0.5):
        self.hop = hop_s
        freq = np.fft.rfftfreq(size, 1 / rate)
        self.band = slice(
            int(np.searchsorted(freq, LOW_HZ)), int(np.searchsorted(freq, min(HIGH_HZ, rate / 2)))
        )
        self.fifth = max(1, (self.band.stop - self.band.start) // 5)
        if self.band.stop - self.band.start < 16:
            raise ValueError("The analysis window is too short for guitar articulation")
        self.position = np.log(freq[self.band] / LOW_HZ) / np.log(HIGH_HZ / LOW_HZ)
        self.sensitivity = sensitivity
        self.delay_s = self.delay_frames * hop_s
        self.frames = max(16, round(HISTORY_S / hop_s))
        self.lags = np.arange(
            max(2, round(1 / (FASTEST_HZ * hop_s))), round(1 / (SLOWEST_HZ * hop_s)) + 1
        )
        self.reset()

    def reset(self):
        self.side, self.mid = Rise(self.hop), Rise(self.hop)
        self.hit_frames = 0  # mono fallback: frames left of the veto after a broadband hit
        self.width = None  # smoothed level of the difference relative to the sum, in dB
        self.flux = np.zeros(self.frames)  # articulation history, newest last
        self.struck = np.zeros(self.frames, dtype=bool)  # which of those frames were attacks
        self.pending = [0.0, 0.0]  # articulation of the two newest frames
        self.shape = [(0.5, 0.0), (0.5, 0.0)]
        self.levels = [0.0, 0.0]
        self.since_attack = 1e6
        self.last_attack = 0.0
        self.count = 0
        self.rate_hz, self.regularity = None, 0.0
        self.basis = "stereo"

    def step(self, spectrum, quiet=False, samples=None):
        """Take the complex spectrum of one frame, shape (bins, channels); return the
        ``Activity`` of the frame before it, or None for the very first frame."""
        left = spectrum[:, 0]
        right = spectrum[:, 1] if spectrum.shape[1] > 1 else left
        mid = np.abs(left + right) / 2
        side = np.abs(left - right) / 2
        mid_rise, mid_mean, mid_loud = self.mid.step(mid[self.band])
        side_rise, side_mean, side_loud = self.side.step(side[self.band])
        width = 20 * np.log10(max(side_mean, 1e-12) / max(mid_mean, 1e-12))
        if not quiet:
            self.width = width if self.width is None else self.width + 0.02 * (width - self.width)
        stereo = self.width is not None and self.width > STEREO_DB
        self.basis = "stereo" if stereo else "mono"
        centre_rise = float(mid_rise.mean())
        # How the centre's rise is spread over the band. A hit that raises every bin alike is
        # noise (a drum); a pluck raises its partials and little else. The share of the rise
        # carried by the strongest fifth of the bins tells them apart: 0.2 for a flat rise,
        # near 1 for a few partials. ``plucked`` is 0 for a flat rise, 1 for a sparse one.
        top = np.partition(mid_rise, -self.fifth)[-self.fifth :]
        share = float(top.sum() / max(float(mid_rise.sum()), 1e-9))
        plucked = min(1.0, max(0.0, (share - SPARSE[0]) / (SPARSE[1] - SPARSE[0])))
        if quiet:
            raw, rise, loud = 0.0, None, 0.0
        elif stereo:
            # The rise of the difference signal as it is. Taking out a share of the centre's
            # rise (to remove what a centre hit leaks into the sides) was tried and dropped:
            # a guitar that is partly in the middle rises in both signals at once, and the
            # correction removed it (measured on stems and on a captured recording).
            raw = float(side_rise.mean())
            rise, loud = side_rise, side_loud
        else:
            # One signal only: count what is plucked, and veto around a broadband hit.
            raw = centre_rise * plucked
            # A strong flat rise is a hit; the noise it leaves behind flickers for a few
            # frames and is not articulation either.
            if centre_rise > 0.15 and share < FLAT:
                self.hit_frames = round(0.08 / self.hop)
                # The frame before saw the edge of the same hit through the window's tail.
                self.pending[1] = 0.0
                self.flux[-1] = 0.0
            if self.hit_frames > 0:
                self.hit_frames -= 1
                raw = 0.0
            rise, loud = mid_rise, mid_loud
        scale = threshold_scale(self.sensitivity)
        flux = float(1.0 - np.exp(-raw / (FLUX_SCALE * scale)))
        centre, spread = 0.5, 0.0
        if rise is not None and raw > 0:
            total = float(rise.sum())
            if total > 1e-9:
                centre = float(np.dot(rise, self.position) / total)
                spread = float(np.sqrt(np.dot(rise, (self.position - centre) ** 2) / total))
        self.flux[:-1] = self.flux[1:]
        self.flux[-1] = flux
        self.struck[:-1] = self.struck[1:]
        self.struck[-1] = False
        self.count += 1
        # An attack is a local maximum of the articulation that clears the recent floor. It
        # is decided one frame later, for the frame that turned out to be the maximum.
        before, peak = self.pending
        peak_centre, peak_spread = self.shape[1]
        level = self.levels[1]
        self.pending = [peak, flux]
        self.shape = [self.shape[1], (centre, spread)]
        self.levels = [self.levels[1], float(loud)]
        self.since_attack += self.hop
        # ... that clears what the articulation has mostly been doing lately (its lower
        # quartile, so that a dense figure does not raise its own bar) and that rose
        # sharply: a slow swell, such as two takes drifting in and out of phase, is not one.
        floor = float(np.percentile(self.flux[-max(8, round(0.5 / self.hop)) :], 25))
        valley = float(self.flux[-5:-2].min())
        attack = 0.0
        if (
            peak > before
            and peak >= flux
            and peak >= 0.3 * scale + 1.3 * floor
            and peak - valley >= 0.22 * scale
            and self.since_attack >= 0.055
            # Soon after a strong stroke, something much weaker is that stroke ringing
            # (two takes drifting against each other), not another one.
            and (self.since_attack >= 0.4 or peak >= 0.4 * self.last_attack)
        ):
            attack = peak
            self.since_attack = 0.0
            self.last_attack = peak
            self.struck[-2] = True
        if self.count % max(1, round(0.1 / self.hop)) == 0:
            self._repetition()
        if self.count < 2:
            return None
        return Activity(
            delay_frames=self.delay_frames,
            level=level,
            flux=peak,
            attack=attack,
            rate_hz=self.rate_hz,
            regularity=self.regularity,
            centre=peak_centre,
            spread=min(1.0, peak_spread / 0.3),
            basis=self.basis,
        )

    def _repetition(self):
        """Rate and regularity of the articulation: its strongest self-similarity lag."""
        x = self.flux - self.flux.mean()
        power = float(np.dot(x, x))
        # A repeating figure needs strokes: a held chord can waver periodically without any.
        if power < 2e-3 * len(x) or self.count < len(x) // 2 or self.struck.sum() < 3:
            self.rate_hz, self.regularity = None, 0.0
            return
        scores = np.array(
            [
                np.dot(x[:-lag], x[lag:])
                / (np.linalg.norm(x[:-lag]) * np.linalg.norm(x[lag:]) + 1e-12)
                for lag in self.lags
            ]
        )
        best = int(np.argmax(scores))
        # A pattern repeating every lag also repeats every 2 x lag: report the shortest
        # lag that is nearly as good as the best one.
        for index in range(best):
            if scores[index] >= 0.85 * scores[best] and scores[index] > 0.3:
                best = index
                break
        regularity = float(max(0.0, min(1.0, scores[best])))
        self.regularity = regularity
        self.rate_hz = float(1.0 / (self.lags[best] * self.hop)) if regularity >= 0.4 else None
