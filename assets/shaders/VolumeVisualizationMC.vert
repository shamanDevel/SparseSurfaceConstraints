#version 430

uniform mat4	ciModelViewProjection;

in int gl_VertexID;
in ivec2 in_NodeIndices;
in float in_InterpWeight;

layout(std430, binding = 3) buffer nodePositionLayout
{
    vec4 nodePositions[];
};

out vec3 posWorld;

void main(void) {
	posWorld = mix(nodePositions[in_NodeIndices.x].xyz, nodePositions[in_NodeIndices.y].xyz, in_InterpWeight);
    gl_Position = ciModelViewProjection * vec4(posWorld, 1);
	//gl_Position /= gl_Position.w;
}