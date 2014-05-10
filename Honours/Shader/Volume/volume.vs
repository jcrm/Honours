/////////////
// GLOBALS //
/////////////
cbuffer MatrixBuffer
{
	matrix world_matrix_;
	matrix view_matrix_;
	matrix projection_matrix_;
};
//////////////
// TYPEDEFS //
//////////////
struct VertexShaderInput
{
	float4 position_ : POSITION;
	float3 texcoord_ : TEXCOORD0;
};
struct PixelShaderInput
{
	float4 position_ : SV_POSITION;
	float3 tex_		: TEXCOORD0;
};
PixelShaderInput VolumeVS(VertexShaderInput input)
{
	PixelShaderInput output;

	input.position_.w = 1.0f;

	// Calculate the position of the vertex against the world, view, and projection matrices.
	output.position_ = mul(input.position_, world_matrix_);
	output.position_ = mul(output.position_, view_matrix_);
	output.position_ = mul(output.position_, projection_matrix_);

	output.tex_ = input.texcoord_;
	return output;
}