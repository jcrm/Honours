cbuffer cbuf
{
  float4 g_vQuadRect;
  int g_UseCase;
}
Texture2D g_Texture2D;
Texture3D g_Texture3D;
TextureCube g_TextureCube;

SamplerState samLinear{
    Filter = MIN_MAG_LINEAR_MIP_POINT;
};
struct VertexInputType
{
    uint SV_VertexID : POSITION;
};
struct PixelInputType{  
    float4 Pos : SV_POSITION; 
    float3 Tex : TEXCOORD0;
}; 
 
PixelInputType SimpleVertexShader(VertexInputType vertexId) 
{ 
    PixelInputType output; 
    output.Tex = float3( 0.f, 0.f, 0.f);  
    if (vertexId.SV_VertexID == 1) output.Tex.x = 1.f;  
    else if (vertexId.SV_VertexID == 2) output.Tex.y = 1.f;  
    else if (vertexId.SV_VertexID == 3) output.Tex.xy = float2(1.f, 1.f);  
     
    output.Pos = float4( g_vQuadRect.xy + output.Tex * g_vQuadRect.zw, 0, 1); 
     
    if (g_UseCase == 1) {  
        if (vertexId.SV_VertexID == 1) output.Tex.z = 0.5f;  
        else if (vertexId.SV_VertexID == 2) output.Tex.z = 0.5f;  
        else if (vertexId.SV_VertexID == 3) output.Tex.z = 1.f;  
    }  
    else if (g_UseCase >= 2) {
        output.Tex.xy = output.Tex.xy * 2.f - 1.f;
    }
    return output;
}