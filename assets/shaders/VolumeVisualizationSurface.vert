#version 150

in vec4			ciPosition;

out vec4 pos;

void main( void ) {
    pos = ciPosition;
	gl_Position	= ciPosition;
}