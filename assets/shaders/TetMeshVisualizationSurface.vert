#version 150
uniform mat4	ciModelViewProjection;
in vec4			ciPosition;

out vec3 posWorld;

void main(void) {
    posWorld = ciPosition.xyz;
    gl_Position = ciModelViewProjection * vec4(ciPosition.xyz, 1);
}