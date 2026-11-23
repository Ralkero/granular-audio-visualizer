#version 330
#include "common.glsl"
// The core: a source of light with depth, in the colours it is given.
//
//   a small centre, the brightest thing in the picture
//   a body shaded like a ball, with cells and slow turbulence that stay visible
//   a bright limb where the body ends
//   a corona that falls away in layers that drift outwards
//   rays of two kinds, fine and broad, turning slowly
//
// Brightness is kept in check here, and the composite brings what is left down without
// changing its hue: a pink core is pink to its middle, not a white disc. Its radius, the
// reach of its corona and of its rays are functions of the level, the intensity, the onset
// flash and the beat pulse (radiant_geometry.core).

uniform vec3 c_core_center, c_core_halo, c_core_rays;
uniform float u_core, u_halo, u_rays, u_ray_turn;  // in units of the reach; the turn in radians

void main() {
    vec2 p = position();
    float r = length(p);
    float a = angle_of(p);
    vec2 dir = vec2(sin(a), -cos(a));
    float x = r / u_core;
    float alive = 0.45 + 0.55 * u_intensity + 0.5 * u_flash;

    // The body: a ball lit from within, darker towards its edge, with cells.
    float ball = sqrt(max(0.0, 1.0 - min(x * x, 1.0)));
    float churn = fbm(vec3(p / u_core * 2.6, u_time * 0.30 + u_seed));
    float cells = ridged(vec3(p / u_core * 4.5, u_time * 0.22 + 6.0));
    vec3 colour = mix(c_core_halo, c_core_center, 0.25 + 0.55 * ball)
                * ball * (0.35 + 0.55 * churn + 0.6 * cells * cells) * (0.7 + 0.5 * alive);
    // The centre: small, and only so bright.
    colour += c_core_center * 1.5 * exp(-x * x * 9.0) * (0.6 + 0.5 * alive);
    // The limb.
    colour += c_core_halo * exp(-pow((x - 1.03) / 0.06, 2.0)) * (0.7 + 0.7 * u_flash + 0.5 * u_pulse);

    float layers = 0.6 + 0.4 * sin(r / u_core * 5.0 - u_time * 1.1 + 2.5 * fbm(vec3(dir * 2.0, u_time * 0.2)));
    float corona = exp(-max(r - u_core, 0.0) / (u_halo * 0.26));
    float drift = 0.5 + 0.8 * fbm(vec3(dir * 2.6, r / u_core * 1.1 - u_time * 0.45));
    colour += c_core_halo * corona * drift * layers * 0.42 * alive * step(1.0, x);

    vec2 rays_at = turn(u_ray_turn) * dir;
    float fine = pow(vnoise(vec3(rays_at * 21.0, u_time * 0.12 + u_seed)), 3.0);
    float broad = pow(vnoise(vec3(rays_at * 4.5, 3.0 + u_time * 0.07)), 2.0);
    float reach = exp(-r / (u_rays * 0.30));
    colour += mix(c_core_rays, c_core_halo, 0.35) * (fine * 1.1 + broad * 0.45) * reach
            * smoothstep(0.8, 1.6, x) * (0.3 + 0.8 * u_intensity + 0.9 * u_flash);

    fragment = vec4(colour * u_shine, 1.0);
}
