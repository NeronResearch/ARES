#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=2) in vec2 aUV;
layout(location=3) in vec4 aTangent; // xyz = tangent, w = handedness

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out VS_OUT {
    vec3 WorldPos;
    vec3 N;
    vec2 UV;
    mat3 TBN;
} vs;

void main(){
    vec4 wp = uModel * vec4(aPos,1.0);
    vs.WorldPos = wp.xyz;
    // Build TBN. If no tangents are meaningful, fragment will fallback to normal space.
    vec3 N = normalize(mat3(uModel) * aNrm);
    vec3 T = normalize(mat3(uModel) * aTangent.xyz);
    vec3 B = normalize(cross(N, T) * aTangent.w);
    vs.TBN = mat3(T, B, N);
    vs.N   = N;
    vs.UV  = aUV;
    gl_Position = uProj * uView * wp;
}
