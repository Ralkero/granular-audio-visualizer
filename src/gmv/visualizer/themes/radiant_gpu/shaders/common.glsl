// Shared by every scene pass: where a pixel is, the music, and the noise everything is
// made of.
//
// Positions are in units of the composition's reach (the radius of the radial view), with
// the centre at the origin and y downwards. Angles are clockwise from the top. All noise
// is a fixed function of its argument: the only things that change a picture are the
// uniforms, and the only clock is audio time.

in vec2 v_uv;
out vec4 fragment;

uniform vec2 u_size;      // the view, in logical pixels
uniform vec2 u_centre;    // centre of the composition, logical pixels, y down
uniform float u_reach;    // its radius, logical pixels
uniform float u_time;     // audio time, seconds (wrapped by the caller)
uniform float u_seed;

// Radii of the composition (radiant_gpu/scene.py), in units of the reach.
uniform float u_base, u_tips;        // the spectrum formation: its floor, its highest surface
uniform float u_contour, u_swing;    // the waveform
uniform float u_seg_low, u_seg_high; // the segments
uniform float u_frame, u_rim;        // the middle and the outer formation
uniform float u_zone;

// The music now, each 0..1.
uniform float u_level, u_intensity, u_flash, u_pulse, u_low, u_mid, u_high;
uniform float u_spectrum[48];  // the display spectrum, lows first
uniform float u_wave[48];      // envelope of the last seconds, newest first

// The response: how strongly things are drawn (never what happened).
uniform float u_plasma, u_motion, u_impact, u_shine;

const float TAU = 6.28318530718;
const float PI = 3.14159265359;

vec2 position() {
    vec2 pixel = vec2(v_uv.x, 1.0 - v_uv.y) * u_size;
    return (pixel - u_centre) / u_reach;
}

float angle_of(vec2 p) {  // clockwise from the top, 0..TAU
    float a = atan(p.x, -p.y);
    return a < 0.0 ? a + TAU : a;
}

mat2 turn(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

float vnoise(vec3 x) {
    vec3 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash13(i), hash13(i + vec3(1, 0, 0)), f.x),
            mix(hash13(i + vec3(0, 1, 0)), hash13(i + vec3(1, 1, 0)), f.x), f.y),
        mix(mix(hash13(i + vec3(0, 0, 1)), hash13(i + vec3(1, 0, 1)), f.x),
            mix(hash13(i + vec3(0, 1, 1)), hash13(i + vec3(1, 1, 1)), f.x), f.y),
        f.z);
}

// Five octaves of noise, about 0..1 with its middle near 0.5.
float fbm(vec3 p) {
    float sum = 0.0, weight = 0.5;
    for (int octave = 0; octave < 5; octave++) {
        sum += weight * vnoise(p);
        p = p * 2.03 + vec3(17.1, 3.7, 9.2);
        weight *= 0.5;
    }
    return sum / 0.96875;
}

// Ridges: bright thin lines where the noise crosses its middle, 0..1.
float ridged(vec3 p) {
    float sum = 0.0, weight = 0.5;
    for (int octave = 0; octave < 4; octave++) {
        float n = 1.0 - abs(2.0 * vnoise(p) - 1.0);
        sum += weight * n * n;
        p = p * 2.11 + vec3(5.3, 11.9, 2.6);
        weight *= 0.5;
    }
    return sum / 0.9375;
}

// How much is going on, for brightness and density: never quite nothing.
float energy() {
    return clamp(0.22 + 0.78 * u_intensity, 0.0, 1.0) * u_plasma;
}

// How far down its side of the circle an angle is: 0 at the top, 1 at the bottom, the
// same on the left as on the right.
float down(float a) {
    return (a < PI ? a : TAU - a) / PI;
}

// The measured spectrum at an angle: lows at the top, highs at the bottom, mirrored. Between
// two bands it is the straight line between them: nothing is added to what was measured.
float spectrum_at(float a) {
    float place = clamp(down(a) * 48.0 - 0.5, 0.0, 47.0);
    int lower = int(floor(place));
    int upper = min(lower + 1, 47);
    return mix(u_spectrum[lower], u_spectrum[upper], place - float(lower));
}

// The envelope of the last seconds at an angle: now at the top, older further down.
float wave_at(float a) {
    float place = clamp(down(a) * 47.0, 0.0, 47.0);
    int lower = int(floor(place));
    int upper = min(lower + 1, 47);
    return mix(u_wave[lower], u_wave[upper], place - float(lower));
}

// A thin bright line where a field crosses its middle: what a filament is made of.
float arc(float value, float width) {
    return width / (abs(value - 0.5) + width);
}
