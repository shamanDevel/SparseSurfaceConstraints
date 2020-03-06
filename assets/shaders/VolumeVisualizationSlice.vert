#version 150

uniform mat4    ciModelMatrix;
uniform mat4	ciModelViewProjection;
in vec4			ciPosition;

out vec4 pos;

void main( void ) {
    pos = ciModelMatrix * ciPosition;
	gl_Position	= ciModelViewProjection * ciPosition;
}