#version 330
// One triangle that covers the target, made from the vertex's number alone (no vertex
// data); v_uv runs 0..1 across the target, origin bottom left.
out vec2 v_uv;

void main() {
    v_uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(v_uv * 2.0 - 1.0, 0.0, 1.0);
}
