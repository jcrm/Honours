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
	float4 scale;
	float3 StepSize;
	float Iterations;
};
//////////////
// TYPEDEFS //
//////////////
struct VertexShaderInput
{
	float4 position : POSITION;
	float2 texcoord : TEXCOORD0;
};

struct PixelShaderInput
{
    float4 position : SV_POSITION;
    float3 texC		: TEXCOORD0;
    float4 pos		: TEXCOORD1;
	float3 StepSize : TEXCOORD2;
	int Iterations: TEXCOORD3;
};
PixelShaderInput VolumeVS(VertexShaderInput input)
{
	PixelShaderInput output;

	input.position.w = 1.0f;
	input.position = input.position * float4(1, 1, 1, 1) * scale;

	// Calculate the position of the vertex against the world, view, and projection matrices.
	output.position = mul(input.position/scale, world_matrix);
	output.position = mul(output.position, view_matrix_);
	output.position = mul(output.position, projection_matrix);

	output.texC = input.position;
	output.pos = output.position;
	output.StepSize = StepSize;
	output.Iterations = Iterations;
	return output;
}