#version 330 core
in VS_OUT {
    vec3 WorldPos;
    vec3 N;
    vec2 UV;
    mat3 TBN;
} fs;

out vec4 FragColor;

// Directional light + environment ambient
uniform vec3 uSunDir;      // direction from sun toward scene (normalize)
uniform vec3 uSunColor;
uniform vec3 uAmbientColor;
uniform vec3 uCameraPos;

// glTF metallic-roughness material
uniform sampler2D uBaseColorTex;         // RGBA
uniform sampler2D uMetallicRoughnessTex; // G = roughness, B = metallic (glTF spec)
uniform sampler2D uNormalTex;            // tangent space normal
uniform sampler2D uOcclusionTex;         // R channel
uniform sampler2D uEmissiveTex;          // RGB

uniform vec4 uBaseColorFactor;           // RGBA
uniform vec2 uMetallicRoughnessFactor;   // x = metallic, y = roughness
uniform vec3 uEmissiveFactor;

// Helpers
vec3 srgb_to_linear(vec3 c){ return pow(c, vec3(2.2)); }
vec3 linear_to_srgb(vec3 c){ return pow(clamp(c,0.0,1.0), vec3(1.0/2.2)); }

// Normal map sampling with fallback
vec3 getNormal(){
    vec3 n = fs.N;
    vec3 tnorm = texture(uNormalTex, fs.UV).xyz * 2.0 - 1.0;
    // If the normal map is not bound (texture id 0) driver reads zeros; detect degenerate:
    if (length(tnorm) < 0.1) return normalize(n);
    return normalize(fs.TBN * normalize(tnorm));
}

// Fresnel (Schlick), Geometry (Schlick-GGX), Distribution (GGX)
float D_GGX(float NdotH, float alpha){
    float a2 = alpha*alpha;
    float d = (NdotH*NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * d * d);
}
float G_SchlickGGX(float NdotV, float k){ return NdotV / (NdotV*(1.0-k) + k); }
float G_Smith(float NdotV, float NdotL, float k){ return G_SchlickGGX(NdotV,k) * G_SchlickGGX(NdotL,k); }
vec3  F_Schlick(vec3 F0, float HdotV){ return F0 + (1.0 - F0) * pow(1.0 - HdotV, 5.0); }

void main(){
    // Material inputs
    vec4 bcTex = texture(uBaseColorTex, fs.UV);
    vec4 baseColor = bcTex * uBaseColorFactor;  // glTF multiplies baseColorTexture by factor
    vec3 albedo = srgb_to_linear(baseColor.rgb);

    vec2 mrTex = texture(uMetallicRoughnessTex, fs.UV).gb; // .g=roughness, .b=metallic
    float metallic  = clamp(mrTex.y * uMetallicRoughnessFactor.x, 0.0, 1.0);
    float roughness = clamp(mrTex.x * uMetallicRoughnessFactor.y, 0.04, 1.0);

    float ao = texture(uOcclusionTex, fs.UV).r;
    vec3  emissive = srgb_to_linear(texture(uEmissiveTex, fs.UV).rgb) * uEmissiveFactor;

    // Shading setup
    vec3 N = getNormal();
    vec3 V = normalize(uCameraPos - fs.WorldPos);
    vec3 L = normalize(-uSunDir);
    vec3 H = normalize(V + L);

    float NdotL = max(dot(N,L), 0.0);
    float NdotV = max(dot(N,V), 0.0);
    float NdotH = max(dot(N,H), 0.0);
    float HdotV = max(dot(H,V), 0.0);

    // Energy-conserving Fresnel at normal incidence
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // Microfacet BRDF
    float alpha = roughness * roughness;
    float D = D_GGX(NdotH, alpha);
    float k = (roughness + 1.0); k = (k*k) / 8.0; // Schlick-GGX for IBL-less direct lights
    float G = G_Smith(NdotV, NdotL, k);
    vec3  F = F_Schlick(F0, HdotV);

    vec3 spec = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);

    vec3 kd = (1.0 - F) * (1.0 - metallic);
    vec3 diffuse = kd * albedo / 3.14159265;

    vec3 direct = (diffuse + spec) * uSunColor * NdotL;

    // Very small constant ambient as placeholder (until IBL/skybox)
    vec3 ambient = uAmbientColor * albedo * ao;

    vec3 color = ambient + direct + emissive;

    FragColor = vec4(linear_to_srgb(color), baseColor.a);
}
