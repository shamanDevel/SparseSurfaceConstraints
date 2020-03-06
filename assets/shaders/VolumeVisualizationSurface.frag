#version 150

uniform mat4 ciModelViewProjection;
uniform mat4 ciModelViewProjectionInverse;
uniform mat4 ciModelViewInverse;

uniform sampler3D volTex;
uniform vec3 boxMin;
uniform vec3 boxSize;
uniform float stepSize;

uniform vec3 directionalLightDir;
uniform vec4 directionalLightColor;
uniform vec4 ambientLightColor;
uniform bool showNormals = false;

in vec4 pos;
out vec4 oColor;
out float gl_FragDepth;

void main() {

    //camera / eye position
    vec3 eyePos = ciModelViewInverse[3].xyz / ciModelViewInverse[3].w;

    //world position
    vec4 screenPos = vec4(pos.x, pos.y, 0.999999, 1);
    vec4 worldPos = ciModelViewProjectionInverse * screenPos;
    worldPos.xyz /= worldPos.w;

    //ray direction
    vec3 rayDir = normalize(worldPos.xyz - eyePos);
    float depth = length(worldPos.xyz - eyePos);
    vec3 invRayDir = vec3(1) / rayDir;

    //entry, exit points
    float t1 = (boxMin.x - eyePos.x) * invRayDir.x;
	float t2 = (boxMin.x + boxSize.x - eyePos.x) * invRayDir.x;
	float t3 = (boxMin.y - eyePos.y) * invRayDir.y;
	float t4 = (boxMin.y + boxSize.y - eyePos.y) * invRayDir.y;
	float t5 = (boxMin.z - eyePos.z) * invRayDir.z;
	float t6 = (boxMin.z + boxSize.z - eyePos.z) * invRayDir.z;
	float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
	float tmax = min(depth, min(min(max(t1, t2), max(t3, t4)), max(t5, t6)));
	if (tmax < 0 || tmin > tmax)
	{
		discard;
    }

    //perform stepping
    vec3 pos = eyePos + tmin * rayDir;
    vec3 normal;
    bool found = false;

	//TODO: adaptive step size based on the SDF value
    for (float sampleDepth = max(0, tmin); sampleDepth < tmax && !found; sampleDepth += stepSize)
    {
        pos = eyePos + sampleDepth * rayDir;
        vec3 volPos = (pos.xyz - boxMin) / boxSize;
        float val = texture(volTex, volPos).r;
        if (val < 0) {
            //we are inside
            //TODO: binary search to improve precision
            found = true;
            //get gradient / normal with central differences
            normal.x = 0.5 * (textureOffset(volTex, volPos, ivec3(1,0,0)).r - textureOffset(volTex, volPos, ivec3(-1,0,0)).r);
            normal.y = 0.5 * (textureOffset(volTex, volPos, ivec3(0,1,0)).r - textureOffset(volTex, volPos, ivec3(0,-1,0)).r);
            normal.z = 0.5 * (textureOffset(volTex, volPos, ivec3(0,0,1)).r - textureOffset(volTex, volPos, ivec3(0,0,-1)).r);
        }
    }
    if (!found) {
        discard;
    }

    //light
    normal = normalize(normal);
    vec4 color = vec4(1,1,1,1);
    if (showNormals) {
        color = vec4((normal + vec3(1))/2, 1);
    }
    //else: color from texture

    //Phong shading
    color = color * (ambientLightColor + directionalLightColor * max(0, dot(normal, -directionalLightDir)));
    color = clamp(color, 0, 1);
    color.a = 1;

    //write result
    vec4 posScreen = ciModelViewProjection * vec4(pos, 1.0);
    float ndcDepth = posScreen.z / posScreen.w;
    gl_FragDepth = (ndcDepth + 1) / 2;
    oColor = color;
}