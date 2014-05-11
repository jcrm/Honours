/////////////
// GLOBALS //
/////////////
cbuffer MatrixBuffer
{
	matrix world_matrix_;
	matrix view_matrix_;
	matrix projection_matrix_;
};
cbuffer ScaleBuffer
{
	float3 step_size_;
	float iterations_;
};
cbuffer CameraData
{
	float4 CameraPosition;
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
	float4 camera_position_ :CAMERA_POSITION;
	float3 step_size_ :STEP;
	float iterations_ :ITER;
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

	output.camera_position_ = CameraPosition;
	output.step_size_ = step_size_;
	output.iterations_ = iterations_;
	return output;
}