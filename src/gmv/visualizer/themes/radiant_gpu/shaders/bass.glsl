// Bass: a heavy ring of pressure.
//
// One thick annulus, the broadest thing an instrument draws. Its thickness and light are
// the energy of the low register now; the note sets how far out it sits; slow lobes travel
// round it at a rate that grows with the energy (a phase the analyzer's rows integrate in
// audio time, the same one the bass layer uses), so a held note keeps moving. Every note
// onset the analyzer reported swells it and sends a pressure wave outwards through the
// plasma. Without a readable note it is paler.
uniform vec4 u_bass;        // weight, energy, place of the note 0..1, swell of recent note onsets
uniform vec4 u_bass_more;   // running phase (radians), pitched, tentative, unused
uniform vec2 u_bass_pulses[4];  // age (s), strength of the recent note onsets

vec3 bass(float r, float a, vec2 dir) {
    float weight = u_bass.x;
    if (weight <= 0.001) return vec3(0.0);
    float power = u_bass.y, swell = min(u_bass.w, 1.5) * u_impact, phase = u_bass_more.x;
    float half_zone = 0.5 * (u_band_bass.y - u_band_bass.x);
    float middle = mix(u_band_bass.x, u_band_bass.y, 0.32 + 0.36 * u_bass.z);
    float lobes = 0.55 * sin(a * 3.0 + phase) + 0.3 * sin(a * 5.0 - phase * 0.7);
    float ring = middle + half_zone * (0.30 * power * lobes * u_motion + 0.16 * swell);
    float thick = half_zone * (0.22 + 0.62 * power + 0.2 * swell);
    float across = (r - ring) / thick;
    float body = exp(-across * across * 1.4);
    vec2 drift = vec2(cos(phase), sin(phase)) * 1.4;
    float grain = 0.45 + 0.85 * fbm(vec3(dir * 5.0 + drift, r * 11.0));
    float veins = pow(ridged(vec3(dir * 9.0 - drift, r * 16.0)), 3.0);
    float light = body * (grain * (0.22 + 1.0 * power + 0.5 * swell) + veins * power * 0.8);
    // Pressure waves of the note onsets.
    float waves = 0.0;
    for (int i = 0; i < 4; i++) {
        float age = u_bass_pulses[i].x;
        float front = ring + age * 0.75;
        waves += u_bass_pulses[i].y * exp(-pow((r - front) / (0.018 + 0.06 * age), 2.0)) * exp(-age * 3.2);
    }
    float sure = 1.0 - 0.3 * max(1.0 - u_bass_more.y, u_bass_more.z);
    return (c_bass * (light + waves * 0.85 * u_impact) + c_plasma_highlights * body * swell * 0.25) * sure * weight;
}
