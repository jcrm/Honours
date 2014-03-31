/////////////
// GLOBALS //
/////////////
cbuffer MatrixBuffer
{
	matrix worldMatrix;
	matrix viewMatrix;
	matrix projectionMatrix;
};

//////////////
// TYPEDEFS //
//////////////
struct VertexInputType
{
    float4 Position : POSITION0;
    float2 texC		: TEXCOORD0;
};

struct PixelInputType
{
	float4 Position		: POSITION0;
	float3 texC			: TEXCOORD0;
	float4 pos			: TEXCOORD1;
};

PixelInputType PositionVS(VertexInputType input){
	PixelInputType output;

	// Change the position vector to be 4 units for proper matrix calculations.
	input.Position.w = 1.0f;

	// Calculate the position of the vertex against the world, view, and projection matrices.
	output.Position = mul(input.Position*0.1f, worldMatrix);
	output.Position = mul(output.Position, viewMatrix);
	output.Position = mul(output.Position, projectionMatrix);

	output.texC = input.Position;
	output.pos = output.Position;

	return output;
}