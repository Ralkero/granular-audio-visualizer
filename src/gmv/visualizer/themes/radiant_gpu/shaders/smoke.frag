#version 330
#include "common.glsl"
// Smoke: clouds round the outside of the composition that drift and change on their own,
// and that every beat pushes about.
//
// It is a procedural field, not a simulation: the picture is worked out afresh for each
// frame from audio time and the list of recent beats, and nothing is carried from one
// frame to the next. That is what makes it the same at any frame rate and in replay.
//
// THE CLOUDS (cloud()). Density is built at three scales, and each scale turns round the
// centre at its own rate and in its own direction, and is looked up through a warp that
// is itself a slowly changing noise field:
//     masses   the big shapes: where there is smoke at all            slow
//     billows  the rounded lumps and curls on them                    faster, the other way
//     wisps    fine texture that eats into thin smoke and frays edges fastest
// Because the scales slide past one another and the warps keep changing, a cloud does not
// travel as a picture: it leans, opens, closes, thins out in one place and gathers in
// another. Two such layers at different depths are added, the far one dimmer and shaded.
// All of it is looked up at a position in the plane (not at an angle), so there is no
// seam anywhere round the circle.
//
// LIGHT. The smoke is lit from the centre. A second look at the density a little way
// towards the centre says how much smoke the light had to pass: thick smoke shades what
// is behind it (the shadow colour), thin smoke and the side facing the centre are lit
// (the primary colour), and the thin fringe of a cloud catches the highlight colour.
//
// INTENSITY. How intense the music is as a whole (u_musical, which changes slowly) makes
// the smoke more turbulent, not just brighter: the warps reach further, the billows curl
// more and the wisps fray the edges more, so loud passages look stirred and quiet ones
// calm. The rates at which everything moves do not change (a rate that followed the music
// would make the picture depend on its history, which a field worked out afresh each frame
// cannot do). Brightness follows intensity only a little.
//
// A BEAT. Each beat the tracker predicted and audio time has reached is a pressure wave
// leaving the centre at WAVE_SPEED. Where its front is, and for a while behind it, the
// smoke is moved: the smoke drawn at a point is looked up somewhere else.
//     outwards    strongest towards one side of the circle (each beat its own side), and
//                 in a pattern of its own whose lobes are broad for some beats and fine
//                 for others
//     sideways    one way here and the other way there
//     in eddies   a small-scale warp that the waves switch on and leave behind them
//     churned     the wisps eat further into the clouds, so edges break up
// The front also throws a brief light on the smoke it is passing. Then the wake dies away
// (longer after a strong beat than a weak one) and reaches exactly nothing at u_beat_life,
// when the beat leaves the list, so nothing jumps; the idle drift is all that is left.
// Several beats add up; the sum is bounded, so fast beats agitate the smoke without tearing
// it apart. The smoke never reaches inside the outer formation, where the instruments are.
//
// LIGHT FROM THE FORMATION. Smoke close to the outer formation catches its light, more when
// the music is intense or the lows are strong.

uniform vec3 c_smoke_primary, c_smoke_highlight, c_smoke_shadow;
uniform int u_beat_count;
uniform vec4 u_beats[16];  // age (s), strength, a turn of its own 0..1, unused
uniform float u_beat_life;  // seconds a beat is in the list (radiant_geometry.BEAT_LIFE)
uniform float u_musical;    // how intense the music is as a whole, 0..1, slowly changing

const float WAVE_FROM = 0.25;   // radius the wave starts at
const float WAVE_SPEED = 3.2;   // reaches the outer formation in a quarter of a second
const float MOST_PUSH = 0.30;   // furthest smoke is ever moved, in units of the reach

float fbm3(vec3 p) {
    float sum = 0.0, weight = 0.5;
    for (int octave = 0; octave < 3; octave++) {
        sum += weight * vnoise(p);
        p = p * 2.07 + vec3(11.3, 5.9, 2.2);
        weight *= 0.5;
    }
    return sum / 0.875;
}

// How much smoke there is at q (0..1). ``churn``: how stirred up it is by beats (0 at
// rest). ``agitation``: how stirred up it is by the music as a whole (0 calm .. 1).
float cloud(vec2 q, float t, float churn, float agitation, float layer) {
    vec2 big = turn(t * 0.021 + layer * 1.7) * q;
    vec2 mid = turn(-t * 0.034 - layer) * q;
    vec2 small = turn(t * 0.058 + layer * 0.6) * q;
    vec2 lean = vec2(fbm3(vec3(big * 1.1, t * 0.045 + layer * 9.0)),
                     fbm3(vec3(big * 1.1 + 4.7, t * 0.045 + 3.1 + layer * 9.0))) - 0.5;
    vec2 curl = vec2(fbm3(vec3(mid * 2.6 + lean, t * 0.09 + layer * 5.0)),
                     fbm3(vec3(mid * 2.6 + lean + 8.3, t * 0.09 + 1.7 + layer * 5.0))) - 0.5;
    float warp = 1.0 + 0.55 * agitation;
    float masses = fbm(vec3(big * 1.25 + 1.5 * warp * lean, t * 0.035 + layer * 13.0));
    float billows = fbm(vec3(mid * 2.9 + 1.1 * warp * lean + 0.75 * warp * curl * (1.0 + churn), t * 0.08 + layer * 7.0));
    float wisps = fbm3(vec3(small * 6.5 + 1.6 * warp * curl * (1.0 + churn), t * 0.17 + layer * 3.0));
    float d = smoothstep(0.44, 0.74, masses * (0.62 - 0.12 * agitation) + billows * (0.38 + 0.12 * agitation));
    return clamp(d * (0.55 + 0.9 * wisps) - (0.10 + 0.10 * agitation + 0.27 * churn) * (1.0 - wisps), 0.0, 1.0);
}

void main() {
    vec2 p = position();
    float r = length(p);
    if (r < u_rim - 0.14 - MOST_PUSH || r > 3.2) {  // no smoke here: the rest is not worked out
        fragment = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    float a = angle_of(p);
    vec2 dir = vec2(sin(a), -cos(a));
    vec2 out_ = p / max(r, 1e-4);
    vec2 along = vec2(-out_.y, out_.x);
    float lively = energy();

    // What the beats still in the air do here.
    vec2 push = vec2(0.0);
    float churn = 0.0, glint = 0.0, stir = 0.0;
    for (int i = 0; i < u_beat_count; i++) {
        float age = u_beats[i].x, strength = u_beats[i].y, own = u_beats[i].z;
        float ending = 1.0 - smoothstep(u_beat_life * 0.6, u_beat_life, age);  // nothing left when it leaves
        float front = WAVE_FROM + WAVE_SPEED * age;
        float passing = exp(-pow((r - front) / (0.10 + 0.16 * age), 2.0)) * exp(-age * 0.9) * ending;
        float lasting = 0.6 + 0.4 * strength;  // a strong beat's wake lingers longer
        float wake = smoothstep(front + 0.05, front - 0.45, r) * exp(-age / lasting) * smoothstep(0.0, 0.25, age) * ending;
        float side = 0.5 + 0.5 * cos(a - own * TAU);  // the side of the circle this beat hits hardest
        float lobes = 1.6 + 1.8 * fract(own * 7.31);  // broad lobes for some beats, fine for others
        float uneven = (0.3 + 1.4 * vnoise(vec3(dir * lobes + own * 37.0, own * 11.0))) * (0.45 + 1.1 * side * side);
        float sideways = (vnoise(vec3(dir * 3.1 + own * 53.0, 4.0 + own * 7.0)) - 0.5) * 2.0;  // one way or the other
        push += strength * (out_ * uneven * (0.16 * passing + 0.08 * wake) + along * sideways * (0.18 * passing + 0.11 * wake));
        stir += strength * (0.10 * passing + 0.16 * wake) * uneven;
        churn += strength * (0.9 * passing + 0.7 * wake);
        glint += strength * passing;
    }
    if (stir > 0.0005) {  // eddies: one field of its own, switched on where the wakes are
        vec2 eddy = vec2(fbm3(vec3(p * 3.4, u_time * 0.5 + u_seed)),
                         fbm3(vec3(p * 3.4 + 6.1, u_time * 0.5 + 2.0 + u_seed))) - 0.5;
        push += eddy * min(stir, 0.3) * 2.0;
    }
    push *= u_impact;
    float moved = length(push);
    push *= min(1.0, MOST_PUSH / max(moved, 1e-4));
    churn = min(churn * u_impact, 1.2);
    glint = min(glint, 1.5);
    vec2 q = p - push;  // the smoke that is here now was there
    float rq = length(q);

    // Where there is smoke at all: outside the outer formation, further out in some
    // directions than in others, and those directions drift.
    vec2 lumps_at = turn(u_time * 0.012) * (q / max(rq, 1e-4));
    float lumps = fbm3(vec3(lumps_at * 1.6, u_time * 0.02 + u_seed));
    float amount = 0.3 + 0.7 * smoothstep(0.3, 0.68, lumps);
    float reach = u_rim + 0.30 + 0.95 * lumps * (0.55 + 0.5 * lively + 0.3 * u_low);
    float where = smoothstep(u_rim - 0.02, u_rim + 0.12, rq) * (1.0 - smoothstep(reach - 0.4, reach, rq))
                * smoothstep(u_rim - 0.02, u_rim + 0.05, r);  // pushed or not, never over the instruments
    if (where <= 0.001) {
        fragment = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float t = u_time;
    float agitation = clamp(u_musical, 0.0, 1.0) * clamp(u_plasma, 0.0, 1.5);
    float near = cloud(q, t, churn, agitation, 0.0) * where * amount;
    float far = cloud(q * 0.82 + 3.7, t * 0.8, churn * 0.6, agitation, 1.0) * where;
    float between = cloud(q - out_ * 0.075, t, churn, agitation, 0.0);  // the smoke the centre's light has to pass
    float lit = exp(-2.4 * between);
    float fringe = near * (1.0 - near) * 4.0;
    float by_formation = exp(-max(rq - u_rim, 0.0) / 0.16) * (0.3 + 0.5 * lively + 0.4 * u_low);

    float strength = (0.8 + 0.45 * lively) * u_shine;
    vec3 colour = mix(c_smoke_shadow, c_smoke_primary, lit) * near * strength
                + mix(c_smoke_shadow, c_smoke_primary, 0.3) * far * 0.42 * strength
                + mix(c_smoke_primary, c_smoke_highlight, 0.6) * near * by_formation * 0.45 * u_shine
                + c_smoke_highlight * (fringe * lit * 0.42 * strength + glint * (near + 0.4 * far) * 0.85 * u_shine);
    fragment = vec4(colour, 1.0);
}
