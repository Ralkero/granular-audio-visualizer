#version 330
#include "common.glsl"
// The four instruments, each a structure of its own kind, in its own zone, in its own
// colour, drawn from its own live state (radiant_gpu/scene.py: instruments). None of this
// is the QPainter instrument layers; the only thing shared with them is the data.
//
// An instrument that is hidden, or that the presence model does not hear, has weight zero
// and draws nothing. Hiding one moves no other.

uniform vec3 c_vocals, c_guitar, c_bass, c_drums, c_plasma_highlights;
uniform vec2 u_band_vocals, u_band_guitar, u_band_bass, u_band_drums;

#include "vocals.glsl"
#include "guitar.glsl"
#include "bass.glsl"
#include "drums.glsl"

void main() {
    vec2 p = position();
    float r = length(p);
    float a = angle_of(p);
    vec2 dir = vec2(sin(a), -cos(a));
    vec3 colour = vocals(r, a, dir) + guitar(r, a, dir) + bass(r, a, dir) + drums(r, a, dir);
    fragment = vec4(colour * u_shine, 1.0);
}
