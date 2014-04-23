////////////////////////////////////////////////////////////////////////////////
// Filename: terrain.vs
////////////////////////////////////////////////////////////////////////////////

/////////////
// GLOBALS //
/////////////
cbuffer MatrixBuffer
{
	matrix world_matrix;
	matrix viewMatrix;
	matrix projection_matrix;
};

//////////////
// TYPEDEFS //
//////////////
struct VertexInputType
{
	float4 position_ : POSITION;
	float2 tex: TEXCOORD0;
	float3 normal : NORMAL;
};

struct PixelInputType
{
	float4 position_ : SV_POSITION;
	float2 tex: TEXCOORD0;
	float3 normal : NORMAL;
};

////////////////////////////////////////////////////////////////////////////////
// Vertex Shader
////////////////////////////////////////////////////////////////////////////////
PixelInputType TerrainVertexShader(VertexInputType input)
{
    PixelInputType output;

	// Change the position vector to be 4 units for proper matrix calculations.
    input.position_.w = 1.0f;

	// Calculate the position of the vertex against the world, view, and projection matrices.
    output.position_ = mul(input.position_, world_matrix);
    output.position_ = mul(output.position_, viewMatrix);
    output.position_ = mul(output.position_, projection_matrix);
    
	output.tex = input.tex;
	// Calculate the normal vector against the world matrix only.
    output.normal = mul(input.normal, (float3x3)world_matrix);
    // Normalize the normal vector.
    output.normal = normalize(output.normal);
    return output;
}