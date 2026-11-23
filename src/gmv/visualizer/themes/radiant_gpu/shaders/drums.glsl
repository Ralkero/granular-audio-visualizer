// Drums: impacts.
//
// Nothing stands here between hits but a faint line. Each hit the analyzer reported is
// drawn for as long as it lasts and is gone, and any number overlap. A hit has a strength
// in three regions of the kit's range (low, mid, high: frequency regions, not named
// drums), and each region has its own kind of impact:
//   low   a broad shockwave that leaves the zone outwards and thins as it goes
//   mid   a burst of crooked sparks all the way round, quickly gone
//   high  a scatter of fine points at the outer edge, gone almost at once
// Where the sparks and points of a hit fall is fixed by the hit's own time.
uniform vec4 u_drums;            // weight, level, unused, unused
uniform vec2 u_drum_hits[8];     // age (s), a turn 0..1 of its own
uniform vec3 u_drum_regions[8];  // strength low, mid, high

vec3 drums(float r, float a, vec2 dir) {
    float weight = u_drums.x;
    if (weight <= 0.001) return vec3(0.0);
    float height = u_band_drums.y - u_band_drums.x;
    float rest = u_band_drums.x + height * 0.3;
    float low = 0.0, mid = 0.0, high = 0.0;
    for (int i = 0; i < 8; i++) {
        float age = u_drum_hits[i].x;
        if (age > 0.6) continue;
        float own = u_drum_hits[i].y;
        vec3 power = u_drum_regions[i] * u_impact;
        // low: a shockwave
        float gone = clamp(age / 0.5, 0.0, 1.0);
        float front = rest + height * (0.2 + 1.5 * (1.0 - (1.0 - gone) * (1.0 - gone)));
        low += power.x * exp(-pow((r - front) / (0.010 + 0.035 * gone), 2.0)) * pow(1.0 - gone, 1.6);
        // mid: sparks
        float short_life = clamp(age / 0.2, 0.0, 1.0);
        float sparks = pow(ridged(vec3(dir * 17.0 + own * 31.0, r * 3.5 + own * 7.0)), 5.0);
        float burst = exp(-pow((r - (rest + height * 0.4)) / (height * (0.35 + 0.5 * short_life)), 2.0));
        mid += power.y * sparks * burst * pow(1.0 - short_life, 1.5) * 2.2;
        // high: points
        float brief = clamp(age / 0.13, 0.0, 1.0);
        vec3 cell = floor(vec3(a * 46.0, r * 120.0, own * 97.0));
        float point = step(0.90, hash13(cell)) * exp(-pow((r - (u_band_drums.y - height * 0.12)) / (height * 0.3), 2.0));
        high += power.z * point * (1.0 - brief) * 1.8;
    }
    float idle = exp(-pow((r - rest) / 0.003, 2.0)) * 0.10;
    vec3 heavy = c_drums * 0.85;                         // the low region a shade darker
    vec3 bright = mix(c_drums, c_plasma_highlights, 0.45);  // the high region lighter
    return (heavy * low * 1.3 + c_drums * (mid + idle) + bright * high) * weight;
}
