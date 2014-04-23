/////////////
// GLOBALS //
/////////////
cbuffer MatrixBuffer
{
	matrix world_matrix;
	matrix view_matrix_;
	matrix projection_matrix;
};
cbuffer ScaleBuffer
{
	float4 scale_;
	float3 step_size_;
	float iterations_;
};
//////////////
// TYPEDEFS //
//////////////
struct VertexShaderInput
{
	float4 position_ : POSITION;
	float2 texcoord : TEXCOORD0;
};

struct PixelShaderInput
{
    float4 position_ : SV_POSITION;
    float3 texC		: TEXCOORD0;
    float4 pos		: TEXCOORD1;
	float3 step_size_ : TEXCOORD2;
	int iterations_: TEXCOORD3;
};
PixelShaderInput VolumeVS(VertexShaderInput input)
{
	PixelShaderInput output;

	input.position_.w = 1.0f;
	input.position_ = input.position_ * float4(1, 1, 1, 1) * scale_;

	// Calculate the position of the vertex against the world, view, and projection matrices.
	output.position_ = mul(input.position_/scale_, world_matrix);
	output.position_ = mul(output.position_, view_matrix_);
	output.position_ = mul(output.position_, projection_matrix);

	output.texC = input.position_;
	output.pos = output.position_;
	output.step_size_ = step_size_;
	output.iterations_ = iterations_;
	return output;
}