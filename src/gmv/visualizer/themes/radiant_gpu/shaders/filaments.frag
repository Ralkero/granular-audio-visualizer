#version 330
#include "common.glsl"
// Electricity: thin, crooked, branching lines of light.
//
// A standing filament is a contour line of a noise field: where the noise crosses its
// middle the line is bright, and such lines wander, fork and close on themselves the way
// arcs do. The field is laid out round the rings (direction, radius), so the lines run
// along them, and it turns and evolves with audio time, so they travel, split and reform.
// A second, slower field decides which stretches are live at the moment, so a filament
// fades in one place and comes up in another instead of flickering.
//   two layers: wide slow arcs between the rings, fine quick ones in the corona
//   highs and intensity: how many stretches are live and how bright
//
// A bolt is different: it is one reported event (an onset of the whole mix, a guitar
// attack, a bass note, a drum hit, the start of a vocal phrase), struck across its own
// band along a crooked path with a branch, and gone in a third of a second. The list of
// bolts comes from the same events as phase 1's streaks: nothing is struck here that was
// not reported.

uniform vec3 c_filament_core, c_filament_glow;
uniform int u_bolt_count;
uniform vec4 u_bolts[16];        // angle, inner radius, outer radius, strength now
uniform vec4 u_bolt_colours[16]; // colour, seed

void main() {
    vec2 p = position();
    float r = length(p);
    float a = angle_of(p);
    vec2 dir = vec2(sin(a), -cos(a));
    float lively = energy();
    vec3 colour = vec3(0.0);

    // Wide arcs between the rings and just beyond the rim.
    vec2 round_a = turn(u_time * 0.16) * dir;
    float field_a = fbm(vec3(round_a * 3.6, r * 5.5 + u_seed) + vec3(0.0, 0.0, u_time * 0.20));
    float live_a = smoothstep(0.50, 0.72, fbm(vec3(round_a * 1.8 + 3.0, u_time * 0.45 + u_seed)));
    float zone_a = smoothstep(u_frame - 0.03, u_frame + 0.05, r) * (1.0 - smoothstep(u_rim + 0.05, u_rim + 0.22, r));
    float wide = arc(field_a, 0.0045) * live_a * zone_a * (0.35 + 0.6 * lively + 0.9 * u_high);

    // Fine quick arcs in the corona, going the other way.
    vec2 round_b = turn(-u_time * 0.31) * dir;
    float field_b = fbm(vec3(round_b * 8.0, r * 12.0 + 11.0 + u_seed) + vec3(0.0, 0.0, u_time * 0.42));
    float live_b = smoothstep(0.52, 0.70, fbm(vec3(round_b * 3.0 + 8.0, r * 3.0 + u_time * 0.8)));
    float zone_b = smoothstep(u_rim - 0.06, u_rim + 0.02, r) * (1.0 - smoothstep(u_rim + 0.25, u_rim + 0.65, r));
    float quick = arc(field_b, 0.003) * live_b * zone_b * (0.28 + 0.55 * lively + 1.1 * u_high);

    // And a few inside the inner composition, round the contour, kept faint.
    vec2 round_c = turn(u_time * 0.22) * dir;
    float field_c = fbm(vec3(round_c * 6.0, r * 16.0 + 23.0) + vec3(0.0, 0.0, u_time * 0.3));
    float zone_c = smoothstep(u_tips - 0.04, u_tips, r) * (1.0 - smoothstep(u_contour + 0.02, u_seg_low, r));
    float faint = arc(field_c, 0.002) * zone_c * smoothstep(0.55, 0.75, fbm(vec3(round_c * 2.0, u_time * 0.6 + 4.0))) * (0.1 + 0.8 * u_high);

    float standing = wide + quick + faint;
    colour += c_filament_glow * standing * 0.9;
    colour += c_filament_core * pow(min(standing, 1.6), 3.0) * 0.8;  // nodes where a line is hottest

    // Bolts.
    for (int i = 0; i < u_bolt_count; i++) {
        vec4 bolt = u_bolts[i];
        float seed = u_bolt_colours[i].w;
        float along = clamp((r - bolt.y) / max(bolt.z - bolt.y, 0.001), 0.0, 1.0);
        float ends = smoothstep(0.0, 0.08, along) * (1.0 - smoothstep(0.85, 1.0, along));
        if (r < bolt.y || r > bolt.z) ends = 0.0;
        float wander = (fbm(vec3(r * 11.0, seed * 13.7, u_time * 1.2)) - 0.5) * 0.30 * along
                     + (vnoise(vec3(r * 60.0, seed * 3.1, u_time * 2.0)) - 0.5) * 0.03;
        float away = a - bolt.x - wander;
        away = abs(mod(away + TAU * 0.5, TAU) - TAU * 0.5) * r;  // distance along the circle
        float width = 0.0022 * (0.7 + bolt.w);
        float main_path = width / (away + width);
        // A branch that leaves the path part of the way out.
        float fork = (fbm(vec3(r * 9.0, seed * 29.3 + 7.0, u_time * 1.2)) - 0.5) * 0.5 * max(along - 0.35, 0.0);
        float off = a - bolt.x - wander - fork;
        off = abs(mod(off + TAU * 0.5, TAU) - TAU * 0.5) * r;
        float branch = 0.6 * width / (off + width) * step(0.35, along);
        float struck = (main_path + branch) * ends * bolt.w * u_impact;
        colour += u_bolt_colours[i].rgb * struck * 2.2 + c_filament_core * pow(min(struck, 1.5), 3.0) * 1.5;
    }

    fragment = vec4(colour * u_shine, 1.0);
}
