cbuffer MatrixBuffer
{
	matrix world_matrix;
	matrix viewMatrix;
	matrix projection_matrix;
};
cbuffer ScaleBuffer
{
	float4 scale_;
};
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
};

PixelShaderInput FaceVS(VertexShaderInput input)
{
	PixelShaderInput output;

	input.position.w = 1.0f;
	input.position = input.position * float4(1, 1, 1, 1) * scale_;

	// Calculate the position of the vertex against the world, view, and projection matrices.
	output.position = mul(input.position/scale_, world_matrix);
	output.position = mul(output.position, viewMatrix);
	output.position = mul(output.position, projection_matrix);

	output.texC = input.position ;
	output.pos = output.position;

	return output;
}