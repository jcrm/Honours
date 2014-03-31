cbuffer MatrixBuffer
{
	matrix worldMatrix;
	matrix viewMatrix;
	matrix projectionMatrix;
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
	input.position = input.position * float4(1, 1, 1, 1);

	// Calculate the position of the vertex against the world, view, and projection matrices.
	output.position = mul(input.position, worldMatrix);
	output.position = mul(output.position, viewMatrix);
	output.position = mul(output.position, projectionMatrix);

	output.texC = input.position;
	output.pos = output.position;

	return output;
}