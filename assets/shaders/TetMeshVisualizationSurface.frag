#version 150

in vec3 normal;
in vec3 edgeCoordinates;
out vec4 oColor;

uniform vec3 lightDir = vec3(0.2, 0.5, 0.3);
uniform vec4 g_edgeColor = vec4(0.2, 0.2, 0.2, 1.0);

// In GLSL 4.5, we have better versions for derivatives
// use them if available
#ifdef GLSL_VERSION_450
#define dFdxFinest dFdxFine
#define dFdyFinest dFdyFine
#define fwidthFinest fwidthFine
#else
// fallback to the possible less precise dFdx / dFdy
#define dFdxFinest dFdx
#define dFdyFinest dFdy
#define fwidthFinest fwidth
#endif

void main(void) {

    //BASE COLOR
    vec4 col = vec4(1.0, 1.0, 0.0, 1.0);

    //EDGES
    //smoothing
    float isEdgeSmoothed = 1;
    vec3 dx = dFdxFinest(edgeCoordinates);
    vec3 dy = dFdyFinest(edgeCoordinates);
    for (int i=0; i<3; ++i) {
        //Distance to the line
        float d = abs(edgeCoordinates[i]-1) / length(vec2(dx[i], dy[i]));
        float fraction = edgeCoordinates[i]<1 ? (1-(0.5*d + 0.5)) : (0.5*d+0.5);
        isEdgeSmoothed *= 1 - clamp(fraction, 0, 1);
    }
    isEdgeSmoothed = 1 - isEdgeSmoothed;
    //blend in edge color
    col.rgb = mix(col.rgb, g_edgeColor.rgb, isEdgeSmoothed*min(1,g_edgeColor.a));

    //LIGHT
    vec3 light = vec3(abs(dot(normal, lightDir)));
    col.rgb = col.rgb * light;

    col.a = 1;
	oColor = col;
}
