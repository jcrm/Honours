////////////////////////////////////////////////////////////////////////////////
// Filename: merge.vs
////////////////////////////////////////////////////////////////////////////////

//////////////
// TYPEDEFS //
//////////////
struct VertexInputType
{
	float4 position_ : POSITION;
	float2 tex_ : TEXCOORD0;
};

struct PixelInputType
{
	float4 position_ : SV_POSITION;
	float2 tex_ : TEXCOORD0;
};

////////////////////////////////////////////////////////////////////////////////
// Vertex Shader
////////////////////////////////////////////////////////////////////////////////
PixelInputType MergeVertexShader(VertexInputType input)
{
	PixelInputType output;
	

	// Change the position vector to be 4 units for proper matrix calculations.
	input.position_.w = 1.0f;

	output.position_ = input.position_;
	output.position_.z = 0.0f;
	// Store the texture coordinates for the pixel shader.
	output.tex_ = input.tex_;
	
	return output;
}