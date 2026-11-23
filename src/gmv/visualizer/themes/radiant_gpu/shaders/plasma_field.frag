#version 330
#include "common.glsl"
// The plasma round the composition: a turbulent corona hugging the outer formation, thin
// plasma between the formations, the two hazes, the points of light and the
// instrument-coloured glow. The smoke beyond it is another pass (smoke.frag).
//
// Clouds are noise looked up through a warp that is itself noise (domain warping), which
// is what gives them folds and branches instead of blobs. The whole field turns slowly,
// faster near the centre, so the motion is a flow and not a flicker.
//   level and intensity : how bright and how far out the corona reaches
//   lows                : how strongly the field is folded, and how far it swells
//   highs               : how bright the fine veins are
//   an onset            : a wave of light going outwards through it (nothing is stretched)

uniform vec3 c_plasma_primary, c_plasma_secondary, c_plasma_highlights;
uniform vec3 c_atmosphere_primary, c_atmosphere_secondary, c_glow, c_particle_accents;
uniform float u_glow;
uniform int u_onset_count;
uniform vec3 u_onsets[8];  // age (s), strength, seed

void main() {
    vec2 p = position();
    float r = length(p);
    float lively = energy();

    // An onset's wave: a front leaving the inner ring and crossing the rim in about 0.4 s.
    float shock = 0.0;
    for (int i = 0; i < u_onset_count; i++) {
        float front = u_frame + (u_rim + 0.5 - u_frame) * u_onsets[i].x / 0.6;
        float width = 0.07 + 0.25 * u_onsets[i].x;
        shock += u_onsets[i].y * exp(-pow((r - front) / width, 2.0)) * exp(-u_onsets[i].x * 4.0);
    }
    shock = min(shock, 1.5) * u_impact;

    vec2 q = turn(u_time * 0.03 / (0.4 + r)) * p;
    vec3 where = vec3(q * 1.5, u_time * 0.05 + u_seed);
    vec2 warp = vec2(fbm(where), fbm(where + vec3(5.2, 1.3, 2.8))) - 0.5;
    warp *= 1.0 + 1.1 * u_low * u_motion;
    float clouds = fbm(vec3((q + 0.95 * warp) * 2.1, u_time * 0.08 + u_seed * 1.7));
    float veins = ridged(vec3((q + 1.35 * warp) * 4.3, u_time * 0.14 + u_seed));
    float threads = ridged(vec3((q + 0.6 * warp) * 9.0, u_time * 0.22 + 3.0));

    // Where the plasma is: a corona whose outer edge is as ragged as the clouds.
    float edge = u_rim + 0.10 + 0.34 * clouds * (0.55 + 0.55 * lively + 0.45 * u_low);
    float corona = smoothstep(u_rim - 0.05, u_rim + 0.05, r) * (1.0 - smoothstep(edge - 0.20, edge + 0.04, r));
    float between = smoothstep(u_frame, u_frame + 0.06, r) * (1.0 - smoothstep(u_rim - 0.08, u_rim, r));
    float inside = 1.0 - smoothstep(u_frame * 0.55, u_frame, r);
    float present = corona + 0.16 * between + 0.03 * inside;

    float density = smoothstep(0.42, 0.78, clouds);  // voids where the clouds thin out
    float body = density * sqrt(density) * (0.42 + 0.95 * lively);
    float light = pow(veins, 3.0) * density * (0.45 + 0.6 * lively + 1.3 * u_high)
                + pow(threads, 5.0) * density * (0.15 + 1.2 * u_high) * 0.7;

    vec3 colour = mix(c_plasma_secondary, c_plasma_primary, smoothstep(0.42, 0.82, clouds)) * body
                + mix(c_plasma_primary, c_plasma_highlights, 0.3) * light * 1.2
                + c_plasma_highlights * shock * density * 0.9;
    colour *= present * u_shine;

    // The two hazes, and the glow in the leading instrument's colour.
    colour += c_atmosphere_primary * 0.11 * exp(-r * r / 0.85);
    colour += c_atmosphere_secondary * 0.045 * smoothstep(0.5, 1.9, r) * (0.4 + clouds);
    colour += c_glow * u_glow * 0.45 * exp(-r * r / 2.2);
    // Points of light carried round with the field, each twinkling at its own rate.
    vec2 grid = q * 26.0;
    vec3 cell = vec3(floor(grid), 3.0);
    float chosen = hash13(cell);
    vec2 spot = fract(grid) - 0.5 - (vec2(hash13(cell + 1.7), hash13(cell + 4.1)) - 0.5) * 0.6;
    float twinkle = 0.55 + 0.45 * sin(u_time * (0.8 + 2.4 * chosen) + chosen * 40.0);
    colour += c_particle_accents * step(0.90, chosen) * smoothstep(0.09, 0.0, length(spot)) * twinkle * (0.35 + 0.5 * lively);
    fragment = vec4(colour, 1.0);
}
