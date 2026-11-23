#version 330
// What is bright enough to light its surroundings: everything over a threshold, eased in
// over a knee so that nothing switches on at an edge. Read from the scene at half the
// view's size, written at a quarter.
in vec2 v_uv;
out vec4 fragment;
uniform sampler2D u_source;
uniform float u_threshold, u_knee;

void main() {
    vec3 colour = texture(u_source, v_uv).rgb;
    float brightest = max(colour.r, max(colour.g, colour.b));
    fragment = vec4(colour * smoothstep(u_threshold, u_threshold + u_knee, brightest), 1.0);
}
