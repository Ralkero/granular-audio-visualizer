#version 330
// One direction of a Gaussian blur: run once across and once down. Five look-ups placed
// between texels stand for nine, so a pass stays cheap.
in vec2 v_uv;
out vec4 fragment;
uniform sampler2D u_source;
uniform vec2 u_step;  // one texel of the source, in the direction of this pass

void main() {
    vec3 colour = texture(u_source, v_uv).rgb * 0.2270270270;
    colour += texture(u_source, v_uv + u_step * 1.3846153846).rgb * 0.3162162162;
    colour += texture(u_source, v_uv - u_step * 1.3846153846).rgb * 0.3162162162;
    colour += texture(u_source, v_uv + u_step * 3.2307692308).rgb * 0.0702702703;
    colour += texture(u_source, v_uv - u_step * 3.2307692308).rgb * 0.0702702703;
    fragment = vec4(colour, 1.0);
}
