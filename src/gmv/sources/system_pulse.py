"""System-output capture on Linux: the monitor of the default output, through the PulseAudio
protocol that PipeWire serves (``pipewire-pulse``) and that PulseAudio itself speaks.

What is captured is what the system mixes for its default output: Spotify, games, browsers,
every application that plays through it. Never a microphone: the stream is opened by name on
the default output's monitor, after the server has confirmed that this source is that
output's monitor, and the source it records from is checked again while it runs.

How:

* ``pactl`` (its JSON output) answers which output is the default and which source is its
  monitor. Nothing is assumed about device names; they are passed as single arguments,
  never through a shell.
* ``parec`` records that monitor as raw little-endian float32, stereo (the server mixes other
  layouts down), at the monitor's own rate, into a pipe that one thread reads in blocks of
  ``block_ms``. There is no Python audio callback: while the reader is busy the server
  buffers, so a late reader shows up as late blocks, not as silently missing ones.
* The stream asks PipeWire not to move it and not to fall back to another source
  (``node.dont-reconnect``, ``node.dont-fallback``, ``node.dont-move``): when its output
  goes away the stream ends instead of being moved, for instance to a microphone. Every
  ``VERIFY_S`` the source the server really records from is checked as well (PulseAudio
  itself ignores those properties); a stream found anywhere else is stopped.
* Every ``POLL_S`` the default output is asked for again. When it has changed, the stream is
  closed and one is opened on the new default's monitor, as a new segment. A stream that
  ends or fails is reopened after a pause that grows to ``BACKOFF_S[-1]``.

Timestamps: ``parec`` reports no capture times, so the clock is the stream's sample count,
anchored on delivery (``DeliveryClock``). Published times are not later than delivery (by
more than ``AHEAD_S``); how much earlier the sound really was (the server's and the device's
latency) is not known and not claimed. Lost audio, a drifting output clock, a pause in delivery and a suspend of the
computer each start a new run of the clock, marked as a discontinuity.

Nothing here plays sound, changes a volume, a default device or the routing, or loads a
module into the server.
"""

import json
import os
import re
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from threading import Event, Thread
from time import perf_counter

from . import Unavailable
from .base import AudioBlock, AudioSource

# How the two client programs are started; tests put stand-ins here.
COMMANDS = {"pactl": ("pactl",), "parec": ("parec",)}
# The package that has both programs on CachyOS and Arch (``pacman -Qo /usr/bin/parec``).
PACKAGE = "libpulse"
TIMEOUT_S = 3.0  # the longest a pactl question may take
POLL_S = 0.5  # how often the default output is asked for
VERIFY_S = 2.0  # how often the stream's source is checked
QUIET_S = 1.0  # no audio for this long: waiting for playback
BACKOFF_S = (0.5, 1.0, 2.0, 4.0, 5.0)  # pauses before reopening, one per failure in a row
CHANNELS = 2
FORMAT = "float32le"
FALLBACK_RATE = 48000
QUALITY = "pulse_delivery_anchored_sample_clock"
# The clock's tolerances (``DeliveryClock``): how far published times may run ahead of
# delivery, and how far behind it they may fall, before a new run of the clock starts.
# Ahead stays well inside half an analysis window (23 ms), so evidence is never from the
# future (``contract.eligible_for_live_impact``).
AHEAD_S, BEHIND_S = 0.010, 0.02
STREAM_PROPERTIES = (
    "node.dont-reconnect=true",
    "node.dont-fallback=true",
    "node.dont-move=true",
)


class PulseProblem(Unavailable):
    """Capture cannot run, in a sentence for the user that says what to do."""


class OutputChanged(Exception):
    """The default output is another one now."""


# -- the client programs ---------------------------------------------------------------------


def missing_programs():
    """Which of the client programs cannot be found."""
    return [name for name in ("pactl", "parec") if _resolve(name) is None]


def _resolve(name):
    argv = list(COMMANDS[name])
    if len(argv) == 1 and os.sep not in argv[0]:
        found = shutil.which(argv[0])
        if found is None:
            return None
        argv[0] = found
    return argv


def programs_problem(missing=None):
    """The sentence for missing client programs; '' when both are installed."""
    missing = missing_programs() if missing is None else missing
    if not missing:
        return ""
    which = " and ".join(missing)
    return (
        f"System audio capture needs the programs pactl and parec, which talk to the sound "
        f"server; {which} {'is' if len(missing) == 1 else 'are'} not installed. On CachyOS "
        f"and Arch they are in the {PACKAGE} package: sudo pacman -S {PACKAGE}"
    )


def command(name):
    """The argument list that starts ``name``; a sentence when it is not installed."""
    argv = _resolve(name)
    if argv is None:
        raise PulseProblem(programs_problem(missing_programs() or [name]))
    return argv


def pactl(*arguments, timeout=TIMEOUT_S):
    """Ask the sound server through ``pactl``; its standard output as text."""
    argv = command("pactl") + list(arguments)
    try:
        done = subprocess.run(
            argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        raise PulseProblem(
            f"The sound server did not answer within {timeout:.0f} s. Check that PipeWire is "
            "running: pactl info"
        ) from None
    except OSError as exc:
        raise PulseProblem(f"pactl could not be started: {exc}") from None
    if done.returncode != 0:
        said = (done.stderr or done.stdout or "").strip().splitlines()
        raise PulseProblem(
            "No PulseAudio-compatible sound server answered"
            + (f" ({said[0]})" if said else f" (pactl exit code {done.returncode})")
            + ". On CachyOS that server is PipeWire's pipewire-pulse: check it with "
            "systemctl --user status pipewire-pulse.socket, or with pactl info."
        )
    return done.stdout


def _json(text, what):
    try:
        return json.loads(text)
    except ValueError:
        raise PulseProblem(
            f"pactl did not give the {what} as JSON. Capture needs pactl from PulseAudio 16 "
            f"or newer (package {PACKAGE})."
        ) from None


# -- the default output and its monitor ------------------------------------------------------


@dataclass(frozen=True)
class Output:
    sink: str  # the server's name for the output
    description: str  # its name for people
    monitor: str  # the source that carries what the output plays
    rate: int  # the monitor's own sample rate
    server: str  # "PulseAudio (on PipeWire 1.6.11)", say


def sample_rate(specification):
    """The rate in a sample specification such as ``s24le 2ch 48000Hz``; None if unclear."""
    found = re.search(r"(\d+)\s*Hz", specification or "")
    rate = int(found.group(1)) if found else 0
    return rate if 8000 <= rate <= 384000 else None


def server_info():
    return _json(pactl("--format=json", "info"), "server information")


def default_sink():
    """The name of the default output, or '' when the server has none."""
    name = (server_info().get("default_sink_name") or "").strip()
    return "" if name.startswith("@") else name  # "@DEFAULT_SINK@": nothing to point to


def default_output():
    """The default output and its monitor, confirmed by the server; a sentence otherwise."""
    info = server_info()
    server = info.get("server_name") or "The sound server"
    sink = (info.get("default_sink_name") or "").strip()
    if not sink or sink.startswith("@"):
        raise PulseProblem(
            f"{server} has no default output. Choose an output in the system's sound "
            "settings; pactl list short sinks lists them."
        )
    sinks = _json(pactl("--format=json", "list", "sinks"), "outputs")
    entry = next((s for s in sinks if s.get("name") == sink), None)
    if entry is None:
        raise PulseProblem(
            f"The default output {sink} is not among the outputs {server} lists; it may "
            "have just been removed."
        )
    description = entry.get("description") or sink
    monitor = (entry.get("monitor_source") or "").strip()
    if not monitor:
        raise PulseProblem(
            f"The default output {description} has no monitor source, so what it plays "
            "cannot be recorded. pactl list short sources lists the monitors there are."
        )
    found = _json(pactl("--format=json", "list", "sources"), "sources")
    source = next((s for s in found if s.get("name") == monitor), None)
    if source is None:
        raise PulseProblem(
            f"The monitor {monitor} of the default output {description} is not among the "
            f"sources {server} lists. pactl list short sources lists the sources there are."
        )
    # For a source, pactl's JSON names the output it monitors (``monitor_source``; empty for
    # a microphone). Anything else is not recorded.
    monitors = source.get("monitor_of_sink") or source.get("monitor_source") or ""
    if monitors != sink:
        raise PulseProblem(
            f"The source {monitor} is not the monitor of the default output {description}, so "
            "it is not recorded."
        )
    rate = (
        sample_rate(source.get("sample_specification"))
        or sample_rate(entry.get("sample_specification"))
        or FALLBACK_RATE
    )
    return Output(sink, description, monitor, rate, server)


def recording_source(pid):
    """The source the stream of process ``pid`` records from; None if it has no stream."""
    outputs = _json(pactl("--format=json", "list", "source-outputs"), "recording streams")
    mine = [o for o in outputs if o.get("properties", {}).get("application.process.id") == str(pid)]
    if not mine:
        return None
    index = mine[0].get("source")
    for source in _json(pactl("--format=json", "list", "short", "sources"), "sources"):
        if source.get("index") == index:
            return source.get("name")
    return f"source #{index}"


def check():
    """Whether system audio can be captured now: the default output, or a sentence."""
    command("parec")
    return default_output()


# -- timestamps --------------------------------------------------------------------------------


def _boot_offset():
    """How far CLOCK_BOOTTIME is ahead of CLOCK_MONOTONIC; it grows by every suspend."""
    boot = getattr(time, "CLOCK_BOOTTIME", None)
    if boot is None:
        return None
    return time.clock_gettime(boot) - time.clock_gettime(time.CLOCK_MONOTONIC)


class DeliveryClock:
    """Sample clock per stream, anchored on when its blocks are delivered.

    A block delivered at ``now`` whose last frame is stream frame ``n`` means stream frame 0
    was captured no later than ``now - n / rate``: an upper bound. While a run settles
    (``settle_s`` of audio, measured and not published) the clock takes the lowest bound
    seen as its anchor, so published times start out no later than delivery. Delivery
    jitters by up to a server period (about 21 ms on PipeWire's default quantum), so later
    blocks arrive somewhat after what the anchor allows; that lateness is kept, its floor
    over ``window_s`` of delivery watched. Published times are never later than delivery by
    more than ``ahead_s``.

    A new run, marked as a discontinuity, starts when
    * the floor rises past ``behind_s``: audio was lost, or the output's clock runs slower
      than this computer's. The anchor moves to the floor and the timeline jumps forward;
    * the floor falls below ``-ahead_s``: the output's clock runs faster (the Sound BlasterX
      G6 on CachyOS ran about 11 ppm fast against ``CLOCK_MONOTONIC``). The anchor moves to
      the floor, and blocks are left out until it has passed the published audio, a block or
      two; the timeline never goes back;
    * nothing was delivered for ``gap_s`` (a paused or suspended output), or the computer was
      suspended (``CLOCK_BOOTTIME`` moved away from ``CLOCK_MONOTONIC``): what the stream's
      sample count means is not known any more, so the run settles again.
    """

    def __init__(
        self, origin, *, settle_s=0.1, window_s=2.0, behind_s=BEHIND_S, ahead_s=AHEAD_S, gap_s=0.25
    ):
        self.origin = origin
        self.settle_s, self.window_s = settle_s, window_s
        self.behind_s, self.ahead_s, self.gap_s = behind_s, ahead_s, gap_s
        self.end = None  # where the audio published so far ends, on the session timeline
        self.runs = 0
        self.behind = self.ahead = self.delivery_gaps = self.suspends = 0
        self.settle_discarded_frames = self.resync_discarded_frames = 0
        self.floor_ms = None
        self.max_lateness_ms = 0.0
        self._boot = _boot_offset()
        self._gap = True
        self.restart()

    def restart(self):
        """A new stream: count its frames from zero, then settle and anchor again."""
        self.position = 0
        self._last = None
        self._settle()

    def _settle(self):
        self.anchor = None
        self._settle_from = self.position
        self._lowest = float("inf")
        self._floor = deque()  # (delivery time, lateness): increasing lateness, oldest first
        self._anchored_at = None

    def _anchor(self, anchor, now):
        self.anchor, self._anchored_at = anchor, now
        self._floor = deque()
        self.runs += 1
        self._gap = True

    def stamp(self, frames, rate, now):
        """``(timestamp, capture_monotonic, discontinuity)`` for a block delivered at
        ``now``, or None for a block that is not published (settling, or before the end of
        what is already published)."""
        start = self.position
        self.position += frames
        bound = now - self.origin - self.position / rate
        boot = _boot_offset()
        if boot is not None and self._boot is not None and boot - self._boot > 0.25:
            self.suspends += 1
            self._settle()
        self._boot = boot
        if self.anchor is not None and self._last is not None and now - self._last > self.gap_s:
            self.delivery_gaps += 1
            self._settle()
        self._last = now
        if self.anchor is None:
            self._lowest = min(self._lowest, bound)
            if (self.position - self._settle_from) / rate < self.settle_s:
                self.settle_discarded_frames += frames
                return None
            self._anchor(self._lowest, now)
        else:
            lateness = bound - self.anchor
            self.max_lateness_ms = max(self.max_lateness_ms, lateness * 1000)
            while self._floor and self._floor[-1][1] >= lateness:
                self._floor.pop()
            self._floor.append((now, lateness))
            while self._floor[0][0] < now - self.window_s:
                self._floor.popleft()
            floor = self._floor[0][1]
            self.floor_ms = floor * 1000
            if now - self._anchored_at >= self.window_s and floor > self.behind_s:
                self.behind += 1
                self._anchor(self.anchor + floor, now)
            elif floor < -self.ahead_s:
                self.ahead += 1
                self._anchor(self.anchor + floor, now)
        timestamp = self.anchor + start / rate
        if self.end is not None and timestamp < self.end - 1e-9:  # rounding only
            self.resync_discarded_frames += frames  # the timeline never goes back
            return None
        gap, self._gap = self._gap, False
        self.end = timestamp + frames / rate
        return timestamp, self.origin + timestamp, gap

    def report(self):
        return {
            "runs": self.runs,
            "behind_resyncs": self.behind,
            "ahead_resyncs": self.ahead,
            "delivery_gaps": self.delivery_gaps,
            "suspends": self.suspends,
            "settle_discarded_frames": self.settle_discarded_frames,
            "resync_discarded_frames": self.resync_discarded_frames,
            "lateness_floor_ms": self.floor_ms,
            "max_lateness_ms": self.max_lateness_ms,
        }


# -- one recording stream ----------------------------------------------------------------------


class Stream:
    """One ``parec`` recording ``output``'s monitor, read by a thread of its own.

    ``deliver(data, frames, now)`` is called on that thread for each block. Standard error is
    read by a second thread into a short buffer, so neither pipe can fill and stall parec."""

    def __init__(self, output, rate, block_frames, deliver):
        self.output, self.rate = output, rate
        self.block_frames, self.deliver = block_frames, deliver
        self.process = None
        self.threads = []
        self.errors, self.said = [], deque(maxlen=20)
        self.ended = Event()

    def arguments(self):
        latency = max(5, round(1000 * self.block_frames / self.rate))
        return command("parec") + [
            f"--device={self.output.monitor}",
            "--raw",
            f"--format={FORMAT}",
            f"--rate={self.rate}",
            f"--channels={CHANNELS}",
            f"--latency-msec={latency}",
            "--client-name=Granular Music Visualizer",
            "--stream-name=System audio capture",
            *(f"--property={entry}" for entry in STREAM_PROPERTIES),
        ]

    def start(self):
        self.process = subprocess.Popen(
            self.arguments(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.threads = [
            Thread(target=self._read, name="pulse-capture-reader", daemon=True),
            Thread(target=self._listen, name="pulse-capture-stderr", daemon=True),
        ]
        for thread in self.threads:
            thread.start()
        return self

    def _read(self):
        size = self.block_frames * CHANNELS * 4
        try:
            while True:
                data = self.process.stdout.read(size)
                frames = len(data) // (CHANNELS * 4)
                if not frames:
                    break
                self.deliver(data[: frames * CHANNELS * 4], frames, perf_counter())
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            self.ended.set()

    def _listen(self):
        try:
            for line in iter(self.process.stderr.readline, b""):
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    self.said.append(text)
        except (OSError, ValueError):
            pass

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None and not self.ended.is_set()

    def why(self):
        if self.threads:
            self.threads[1].join(timeout=0.5)  # its last words, if parec has gone
        code = self.process.poll() if self.process else None
        said = "; ".join(self.said)
        return said or (f"parec exit code {code}" if code is not None else "the stream ended")

    def close(self):
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()  # only our own child
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        for thread in self.threads:
            thread.join(timeout=2)
        for pipe in (process.stdout, process.stderr):
            try:
                pipe.close()
            except OSError:
                pass


class Feed:
    """Turns one stream's delivered blocks into ``AudioBlock``s on the engine's queue. The
    queue is bounded and drops its oldest block when full (``BoundedQueue``): a slow analysis
    loses blocks, counted there, instead of making capture wait."""

    def __init__(self, output, clock, segment, rate):
        self.output, self.clock, self.segment, self.rate = output, clock, segment, rate
        self.last = perf_counter()  # when audio last arrived
        self.blocks = 0  # blocks published

    def __call__(self, data, frames, now):
        self.last = now
        stamped = self.clock.stamp(frames, self.rate, now)
        if stamped is None:
            return
        timestamp, mono, gap = stamped
        self.output.put(
            AudioBlock(
                data,
                frames,
                self.rate,
                CHANNELS,
                timestamp,
                mono,
                now,
                segment=self.segment,
                discontinuity=gap,
                timestamp_quality=QUALITY,
            )
        )
        self.blocks += 1


# -- the audio source ----------------------------------------------------------------------------


class PulseSystemAudioSource(AudioSource):
    """What the system plays on its default output, following it when it changes."""

    kind = "system_audio"

    def __init__(self, *, block_ms=10):
        self.block_ms = block_ms
        self.output = None
        self.clock = None
        self.recoveries = 0
        self.output_switches = 0
        self.streams_opened = 0
        self.native_drops = 0  # parec reports no losses; the clock finds them (``behind``)
        self.last_problem = ""

    def describe(self):
        output = self.output
        return {
            "kind": self.kind,
            "backend": "pulse_monitor_parec",
            "selection": "default output",
            "server": output.server if output else None,
            "output": output.description if output else None,
            "sink": output.sink if output else None,
            "monitor": output.monitor if output else None,
            "format": [output.rate, CHANNELS, FORMAT] if output else None,
            "streams_opened": self.streams_opened,
            "output_switches": self.output_switches,
            "recoveries": self.recoveries,
            "last_problem": self.last_problem,
            "clock": self.clock.report() if self.clock else None,
            "timestamp_quality": QUALITY,
        }

    def run(self, output, stop, status):
        clock = self.clock = DeliveryClock(getattr(self, "timeline_origin", perf_counter()))
        segment, failures = 0, 0
        try:
            while not stop.is_set():
                stream, feed = None, None
                try:
                    target = default_output()
                    self.output = target
                    segment += 1
                    clock.restart()
                    feed = Feed(output, clock, segment, target.rate)
                    block_frames = max(1, round(target.rate * self.block_ms / 1000))
                    stream = Stream(target, target.rate, block_frames, feed).start()
                    self.streams_opened += 1
                    status(
                        "capturing",
                        f"{target.description} (default output), recorded from its monitor at "
                        f"{target.rate} Hz through {target.server}",
                    )
                    self._watch(stream, target, feed, stop, status)
                except OutputChanged as change:
                    self.output_switches += 1
                    status("recovering", str(change))
                except Exception as exc:
                    if stop.is_set():
                        break
                    failures = 0 if feed is not None and feed.blocks else failures + 1
                    pause = BACKOFF_S[min(max(failures - 1, 0), len(BACKOFF_S) - 1)]
                    self.recoveries += 1
                    self.last_problem = str(exc)
                    status("recovering", f"{exc} Trying again in {pause:g} s.")
                    stop.wait(pause)
                finally:
                    if stream is not None:
                        stream.close()
        finally:
            output.close()

    def _watch(self, stream, target, feed, stop, status):
        """Until stopped: raise when the stream ends, is found on another source, or the
        default output changes."""
        verify_at, quiet = perf_counter() + 1.0, False
        while not stop.wait(POLL_S):
            if stream.errors:
                raise PulseProblem(f"Reading the capture stream failed: {stream.errors[0]}.")
            if not stream.running:
                raise PulseProblem(
                    f"The capture stream from {target.description} ended ({stream.why()}). "
                    "The output may have been removed, or the sound server restarted."
                )
            current = default_sink()
            if current != target.sink:
                # The next status names the new output by its description: sink names can
                # carry serial numbers, and the status goes into the session's log.
                raise OutputChanged(
                    "The default output changed; following it."
                    if current
                    else "There is no default output now; waiting for one."
                )
            now = perf_counter()
            if now >= verify_at:
                source = recording_source(stream.process.pid)
                if source is not None and source != target.monitor:
                    raise PulseProblem(
                        f"The capture stream was moved to {source}, which is not the monitor "
                        f"of the default output; it was stopped so that nothing else is "
                        "recorded."
                    )
                verify_at = now + VERIFY_S
            silent = now - feed.last > QUIET_S
            if silent != quiet:
                status(
                    "waiting" if silent else "capturing",
                    "No audio from the default output's monitor; waiting for playback"
                    if silent
                    else "Audio resumed",
                )
                quiet = silent


def system_audio_source(device_name=None, backend="auto"):
    """The default output's monitor, once the server has shown it can be recorded."""
    if device_name:
        raise Unavailable(
            "Linux captures the default output only; choosing an output device is not "
            "available there. Choose System audio, an audio file, or replay a recorded session."
        )
    if backend not in ("auto", "pulse"):
        raise Unavailable(
            f"The {backend} capture backend is a Windows one; on Linux the system output is "
            "captured through the PulseAudio service (backend auto)."
        )
    check()
    return PulseSystemAudioSource()
