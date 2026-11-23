// Guitar: articulated electrical strands.
//
// Five thin live wires run round the zone. A stroke plucks the ones it covered: they
// flash, and swing as a standing wave that starts again at every stroke and dies away
// with the ringing envelope (slowly under a held chord, quickly after a muted stroke),
// with a fine electrical jitter on top. While strokes repeat regularly, arcs jump between
// the strands. With nothing ringing the wires are faint and still.
uniform vec4 u_guitar;       // weight, level, ringing envelope, strength of the last stroke
uniform vec4 u_guitar_more;  // seconds since that stroke, where it sat 0..1, how wide, steady repetition 0..1

vec3 guitar(float r, float a, vec2 dir) {
    float weight = u_guitar.x;
    if (weight <= 0.001) return vec3(0.0);
    float level = u_guitar.y, ringing = u_guitar.z, struck = u_guitar.w;
    float since = u_guitar_more.x, centre = u_guitar_more.y, spread = u_guitar_more.z, steady = u_guitar_more.w;
    float height = u_band_guitar.y - u_band_guitar.x;
    float flash = struck * exp(-since / 0.11) * u_impact;
    float light = 0.0, hot = 0.0;
    for (int k = 0; k < 5; k++) {
        float n = float(k);
        float place = n / 4.0;
        float covered = exp(-pow((place - centre) / (0.14 + 0.7 * spread), 2.0));
        float swing = cos(TAU * (4.0 + 1.2 * n) * since);
        float wave = height * 0.11 * ringing * covered * swing * sin((9.0 + 3.0 * n) * a);
        float jitter = height * 0.05 * (ringing * covered + 0.25 * level)
                     * (vnoise(vec3(dir * 28.0 + n * 5.0, u_time * 5.0)) - 0.5) * 2.0;
        float wire = mix(u_band_guitar.x + 0.008, u_band_guitar.y - 0.008, place) + (wave + jitter) * u_motion;
        float width = 0.0016 * (1.0 + 2.2 * flash * covered);
        float away = abs(r - wire);
        float line = width / (away + width) * exp(-pow(away / (height * 0.16), 2.0));  // thin, and gone a little way off
        float alive = 0.10 + 0.6 * max(min(1.0, ringing * covered * 1.5), 0.35 * level);
        light += line * alive;
        hot += line * flash * covered;
    }
    float zone = smoothstep(u_band_guitar.x - 0.02, u_band_guitar.x - 0.004, r)
               * (1.0 - smoothstep(u_band_guitar.y + 0.004, u_band_guitar.y + 0.02, r));
    float jump = arc(fbm(vec3(dir * 11.0, r * 46.0 + u_time * 0.9)), 0.0028)
               * smoothstep(0.5, 0.75, fbm(vec3(dir * 4.0 + 2.0, u_time * 1.4)))
               * zone * steady * (0.5 + min(1.0, ringing)) * 1.6;
    return (c_guitar * (light * 1.5 + jump + hot * 1.4) + c_plasma_highlights * hot * 0.9) * zone * weight;
}
