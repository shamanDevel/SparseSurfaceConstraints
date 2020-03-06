#version 430

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

uniform ivec2 ciWindowSize;
uniform float g_edgeSize = 2;

in vec3 posWorld[];

out vec3 normal;
out vec3 edgeCoordinates;

void main()
{
    //Compute normals
	normal = normalize(cross(
		posWorld[0] - posWorld[2],
		posWorld[1] - posWorld[2]));

    //Compute edges
    //vertices coordinates in pixel space
    vec2 screenA = 0.5 * vec2(ciWindowSize) * gl_in[0].gl_Position.xy / gl_in[0].gl_Position.w;
    vec2 screenB = 0.5 * vec2(ciWindowSize) * gl_in[1].gl_Position.xy / gl_in[1].gl_Position.w;
    vec2 screenC = 0.5 * vec2(ciWindowSize) * gl_in[2].gl_Position.xy / gl_in[2].gl_Position.w;
    //side lengths in pixel coordinates
    float ab = length(screenB - screenA);
    float ac = length(screenC - screenA);
    float bc = length(screenC - screenB);
    //cosines angles at the vertices
    float angleACos = dot((screenB - screenA) / ab, (screenC - screenA) / ac);
    float angleBCos = dot((screenA - screenB) / ab, (screenC - screenB) / bc);
    float angleCCos = dot((screenA - screenC) / ac, (screenB - screenC) / bc);
    //sines at the vertices
    float angleASin = sqrt(1 - angleACos*angleACos);
    float angleBSin = sqrt(1 - angleBCos*angleBCos);
    float angleCSin = sqrt(1 - angleCCos*angleCCos);

    //desired edge width in pixels
    vec3 edgeWidth = vec3(
        g_edgeSize / length(gl_in[0].gl_Position.z),
        g_edgeSize / length(gl_in[1].gl_Position.z),
        g_edgeSize / length(gl_in[2].gl_Position.z)
    );
    //compute edge coordinates
    vec3 edgeCoords[3];
    edgeCoords[0] = vec3(
        0,
        1 / (1 - min(0.99999, edgeWidth.x / (ab * angleASin))),
        1 / (1 - min(0.99999, edgeWidth.x / (ac * angleASin)))
    );
    edgeCoords[1] = vec3(
        1 / (1 - min(0.99999, edgeWidth.y / (ab * angleBSin))),
        0,
        1 / (1 - min(0.99999, edgeWidth.y / (bc * angleBSin)))
    );
    edgeCoords[2] = vec3(
        1 / (1 - min(0.99999, edgeWidth.z / (ac * angleCSin))),
        1 / (1 - min(0.99999, edgeWidth.z / (bc * angleCSin))),
        0
    );

    //Send to pixel shader
    gl_Position = gl_in[0].gl_Position;
    edgeCoordinates = edgeCoords[0];
    EmitVertex();

    gl_Position = gl_in[1].gl_Position;
    edgeCoordinates = edgeCoords[1];
    EmitVertex();

	gl_Position = gl_in[2].gl_Position;
    edgeCoordinates = edgeCoords[2];
    EmitVertex();

    EndPrimitive();
}