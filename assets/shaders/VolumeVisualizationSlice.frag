#version 150

uniform sampler1D tfTex;
uniform sampler3D volTex;
uniform float tfMin;
uniform float tfMax;
uniform vec3 boxMin;
uniform vec3 boxSize;

in vec4 pos;
out vec4 oColor;

void main() {
    //convert pos to volume space
    vec3 volPos = (pos.xyz - boxMin) / boxSize;
    if (volPos.x<0 || volPos.y<0 || volPos.z<0 
        || volPos.x>1 || volPos.y>1 || volPos.z>1) {
        discard;
        return;
    }

    //sample into the 3d volume
    float val = texture(volTex, volPos).r;
    val = -val; //level sets are positive outside

    //get color from transfer function
    vec4 col = texture(tfTex, (val - tfMin) / (tfMax - tfMin));

    oColor = col;
}