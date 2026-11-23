#version 330
#include "common.glsl"
// The concentric formations: what the composition is built of. Each has its own noise,
// scale, direction and driver; none is an outline.
//
//   spectrum formation  a plasma annulus round the core whose outer surface is the
//       spectrum: lows at the top, highs at the bottom, mirrored. Where a band is strong
//       the plasma stands high, is dense and bright and full of fine filaments; where it
//       is quiet there is only a low dim floor. Between two bands the surface is the
//       straight line between them. This is the frequency display; there are no spikes.
//   waveform            a thin luminous ring displaced by the envelope of the last
//       seconds, newest at the top.
//   segments            forty segments, each lit by its own stretch of the spectrum, with
//       a texture inside, ragged edges and light running from one to the next.
//   middle formation    a thin fast shell, fine-grained, going anticlockwise: mids, highs.
//   outer formation     a broad slow heavy shell going clockwise that swells and buckles
//       with the lows, with tendrils leaving it outwards.
//
// Noise round a ring is looked up at (direction, radius): it has no seam.

uniform vec3 c_spectrum_body, c_spectrum_highlights, c_secondary_ring, c_segmented_ring;
uniform vec3 c_primary_ring, c_plasma_highlights;
uniform float u_segments[40];  // how lit each segment is, once round from the top
uniform vec4 u_bass;           // weight, energy, place, swell: the outer formation feels the bass

void main() {
    vec2 p = position();
    float r = length(p);
    float a = angle_of(p);
    vec2 dir = vec2(sin(a), -cos(a));
    float lively = energy();
    vec3 colour = vec3(0.0);

    // -- spectrum formation ---------------------------------------------------------------
    float band = spectrum_at(a);
    float floor_r = u_base;
    float low_top = floor_r + 0.035;
    float ragged = (fbm(vec3(dir * 9.0, u_time * 0.5 + u_seed)) - 0.5) * 0.035 * (0.25 + band) * u_motion;
    float surface = low_top + (u_tips - low_top) * pow(band, 0.8) + ragged;
    float inside = smoothstep(floor_r - 0.008, floor_r + 0.008, r) * (1.0 - smoothstep(surface - 0.014, surface + 0.006, r));
    float flow = ridged(vec3(dir * 7.0, r * 13.0 - u_time * 0.55 + u_seed));
    float strands = pow(ridged(vec3(dir * 24.0, r * 4.5 + u_time * 0.28)), 3.0);
    float height = clamp((r - floor_r) / max(surface - floor_r, 0.001), 0.0, 1.0);
    float body = inside * (0.22 + 0.9 * flow * flow) * (0.3 + 1.0 * band) * (0.55 + 0.45 * height);
    float skin = exp(-pow((r - surface) / 0.009, 2.0)) * (0.3 + 1.7 * band);
    colour += c_spectrum_body * (body + exp(-pow((r - floor_r) / 0.006, 2.0)) * 0.5) * (0.5 + 0.7 * lively);
    colour += c_spectrum_highlights * (skin + strands * inside * band * 1.3 + u_flash * skin * 0.6);

    // -- waveform -------------------------------------------------------------------------
    float heard = wave_at(a);
    float wavy = u_contour + u_swing * u_motion * heard * (1.2 + 0.9 * sin(a * 96.0));
    colour += c_secondary_ring * exp(-pow((r - wavy) / 0.0045, 2.0)) * (0.35 + 1.3 * heard);

    // -- segments -------------------------------------------------------------------------
    float place = a / TAU * 40.0;
    int index = int(min(floor(place), 39.0));
    float along = abs(fract(place) - 0.5);
    float lit = u_segments[index];
    float segment = 1.0 - smoothstep(0.31, 0.37, along + 0.03 * (vnoise(vec3(dir * 40.0, u_time * 1.3)) - 0.5));
    float middle = 0.5 * (u_seg_low + u_seg_high);
    float half_depth = 0.5 * (u_seg_high - u_seg_low) * (1.0 + 0.5 * (vnoise(vec3(dir * 34.0, u_time * 1.7 + 5.0)) - 0.5));
    float across = exp(-pow((r - middle) / half_depth, 4.0));
    float weave = 0.55 + 0.9 * fbm(vec3(dir * 22.0, r * 38.0 - u_time * 0.75));
    float running = pow(0.5 + 0.5 * sin(a * 5.0 - u_time * 2.1), 12.0);  // light going round
    float power = 0.18 + 1.5 * lit + 0.9 * u_flash + 0.6 * u_pulse;
    colour += c_segmented_ring * across * segment * weave * power;
    colour += c_plasma_highlights * across * (segment * lit * lit * 0.9 + running * (0.2 + 0.6 * lively));

    // -- middle formation -----------------------------------------------------------------
    vec2 fast = turn(-u_time * 0.27) * dir;
    float ripple = (fbm(vec3(fast * 5.0, u_time * 0.24 + 7.0 + u_seed)) - 0.5) * 0.03 * (0.6 + u_mid) * u_motion;
    float e = r - u_frame - ripple;
    float thin = 0.010 + 0.010 * u_high + 0.006 * lively;
    float inner = exp(-e * e / (thin * thin));
    float fine = ridged(vec3(fast * 15.0, r * 28.0 + u_time * 0.65 + 2.0));
    colour += c_primary_ring * inner * (0.3 + 1.5 * fine * fine) * (0.4 + 0.9 * u_high + 0.45 * lively);
    colour += c_plasma_highlights * inner * pow(fine, 5.0) * (0.7 + 1.2 * u_high);

    // -- outer formation ------------------------------------------------------------------
    float heavy = max(u_low, u_bass.x * u_bass.y);
    vec2 slow = turn(u_time * 0.10) * dir;
    float buckle = (fbm(vec3(slow * 2.4, u_time * 0.09 + u_seed)) - 0.5) * 0.11 * (0.5 + 1.3 * heavy) * u_motion;
    float d = r - u_rim - buckle - 0.02 * heavy * u_motion;
    float thick = 0.030 + 0.024 * lively + 0.012 * heavy;
    float shell = exp(-d * d / (thick * thick));
    float ridge = ridged(vec3(slow * 6.5, r * 8.0 - u_time * 0.22 + u_seed));
    float grain = fbm(vec3(slow * 13.0, r * 21.0 + u_time * 0.35));
    float tendril = pow(ridged(vec3(slow * 4.6, r * 2.8 - u_time * 0.32 + 4.0)), 4.0)
                  * smoothstep(0.0, 0.035, d) * exp(-max(d, 0.0) * 6.0);
    float branch = pow(ridged(vec3(slow * 11.0, r * 6.0 - u_time * 0.5 + 9.0)), 5.0)
                 * smoothstep(0.02, 0.08, d) * exp(-max(d, 0.0) * 9.0);
    colour += c_primary_ring * shell * (0.3 + 1.7 * ridge * ridge + 0.7 * grain) * (0.45 + 0.95 * lively);
    colour += c_plasma_highlights * (shell * pow(ridge, 4.0) * 1.5 + (tendril + branch) * (0.5 + 1.6 * lively));
    colour += c_primary_ring * (tendril + 0.6 * branch) * 0.9 * (0.4 + lively);

    fragment = vec4(colour * u_shine, 1.0);
}
