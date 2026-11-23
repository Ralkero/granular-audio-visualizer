#version 330
// The picture the view shows: the background colour, the scene, and the light the bright
// parts of it throw (a tight bloom and a wide one), brought down from open-ended
// brightness to what a display shows and encoded for it.
//
// Everything up to here is linear light, where adding two lights is adding two numbers.
// Brightness is brought down by the curve 1 - exp(-x) applied to the brightest channel,
// and the other channels are scaled with it: dark things stay dark, bright things stop
// short of clipping, and a colour keeps its hue and saturation however bright it is.
// Only the very brightest light is let a little way towards white. Outside the visualizer's own area (the header above, the sections
// below) the picture fades to the plain background.
in vec2 v_uv;
out vec4 fragment;
uniform sampler2D u_scene, u_bloom_tight, u_bloom_wide;
uniform vec3 c_background_base;
uniform float u_bloom;      // the response's gain on the bloom
uniform float u_exposure;
uniform vec2 u_area;        // top and bottom of the visualizer's area, as shares of the height
uniform int u_show;         // 0 the picture; 1 the scene alone; 2 the bloom alone (for tests)

vec3 encoded(vec3 linear) {
    vec3 low = linear * 12.92;
    vec3 high = 1.055 * pow(max(linear, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(low, high, step(vec3(0.0031308), linear));
}

void main() {
    vec3 scene = texture(u_scene, v_uv).rgb;
    vec3 bloom = (texture(u_bloom_tight, v_uv).rgb * 0.45 + texture(u_bloom_wide, v_uv).rgb * 0.6) * u_bloom;
    float y = 1.0 - v_uv.y;
    float inside = smoothstep(u_area.x - 0.02, u_area.x + 0.03, y) * (1.0 - smoothstep(u_area.y - 0.02, u_area.y + 0.04, y));
    vec3 light = u_show == 1 ? scene : u_show == 2 ? bloom : (scene + bloom) * (0.25 + 0.75 * inside);
    light = c_background_base + light * u_exposure;
    float brightest = max(light.r, max(light.g, light.b));
    float brought = 1.0 - exp(-brightest);
    vec3 kept = light * (brought / max(brightest, 1e-5));
    kept = mix(kept, vec3(brought), 0.3 * smoothstep(2.5, 9.0, brightest));
    fragment = vec4(encoded(kept), 1.0);
}
