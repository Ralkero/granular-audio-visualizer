// Vocals: flowing ribbons of plasma.
//
// Four ribbons run round the zone, weaving past one another. How far they open and how
// bright they are is the level of the voice now; where the whole bundle sits in the zone
// is the pitch (higher: further out); the weave drifts all the while a voice sounds. The
// start of a phrase sends a pulse of light round from the top, both ways. Without a
// readable pitch, or while the presence model is unsure, they are paler.
uniform vec4 u_vocals;       // weight, level, place of the pitch 0..1, bloom of a phrase start
uniform vec4 u_vocals_more;  // the voice's own time (s), pitched, tentative, age of the newest phrase start

vec3 vocals(float r, float a, vec2 dir) {
    float weight = u_vocals.x;
    if (weight <= 0.001) return vec3(0.0);
    float level = u_vocals.y, bloom = u_vocals.w, t = u_vocals_more.x;
    float open = min(1.3, level * (1.0 + 0.6 * bloom * u_impact));
    float half_zone = 0.5 * (u_band_vocals.y - u_band_vocals.x);
    float middle = mix(u_band_vocals.x, u_band_vocals.y, 0.5 + 0.3 * (u_vocals.z - 0.5));
    float light = 0.0;
    for (int k = 0; k < 4; k++) {
        float n = float(k);
        float weave = sin(a * (3.0 + 2.0 * n) + t * (0.9 + 0.35 * n) + n * 2.1)
                    + 0.5 * sin(a * (7.0 + n) - t * 0.6 + n);
        float centre = middle + half_zone * open * u_motion * (0.42 * (n - 1.5) / 1.5 + 0.42 * weave);
        float width = half_zone * (0.05 + 0.10 * open);
        float across = (r - centre) / width;
        float current = 0.5 + 0.7 * fbm(vec3(dir * 6.0 + n, t * 0.5 - n * 3.0));  // light running along it
        light += exp(-across * across) * current * (0.2 + 1.2 * open);
    }
    // A phrase begins: a pulse leaves the top and runs down both sides.
    float age = u_vocals_more.w;
    float front = age * 2.4;
    float pulse = exp(-pow((down(a) * PI - front) / 0.22, 2.0)) * exp(-age * 1.6)
                * exp(-pow((r - middle) / (half_zone * 1.3), 2.0)) * min(1.0, bloom + step(age, 2.0) * 0.3);
    float sure = 1.0 - 0.4 * max(1.0 - u_vocals_more.y, u_vocals_more.z);
    float zone = smoothstep(u_band_vocals.x - half_zone, u_band_vocals.x, r)
               * (1.0 - smoothstep(u_band_vocals.y, u_band_vocals.y + half_zone, r));
    return (c_vocals * (light + pulse * 1.4) + c_plasma_highlights * pulse * 0.5) * zone * sure * weight;
}
