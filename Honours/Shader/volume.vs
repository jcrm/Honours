SamplerState FrontS
{
	Texture = Front;
	MinFilter = LINEAR;
	MagFilter = LINEAR;
	MipFilter = LINEAR;
	
	AddressU = Border;				// border sampling in U
    AddressV = Border;				// border sampling in V
    BorderColor = float4(0,0,0,0);	// outside of border should be black
};

SamplerState BackS
{
	Texture = Back;
	MinFilter = LINEAR;
	MagFilter = LINEAR;
	MipFilter = LINEAR;
	
	AddressU = Border;				// border sampling in U
	AddressV = Border;				// border sampling in V
	BorderColor = float4(0,0,0,0);	// outside of border should be black
};

SamplerState VolumeS
{
	Texture = Volume;
	MinFilter = LINEAR;
	MagFilter = LINEAR;
	MipFilter = LINEAR;
	
	AddressU = Border;				// border sampling in U
	AddressV = Border;				// border sampling in V
	AddressW = Border;
	BorderColor = float4(0,0,0,0);	// outside of border should be black
};

cbuffer MatrixBuffer
{
float4x4 World;
float4x4 WorldViewProj;
float4x4 WorldInvTrans;
};

float4 ScaleFactor;

Texture2D Front;
Texture2D Back;
Texture2D Volume;

struct VertexShaderInput
{
    float4 Position : POSITION0;
    float2 texC		: TEXCOORD0;
};

struct VertexShaderOutput
{
	float4 Position		: POSITION0;
	float3 texC			: TEXCOORD0;
	float4 pos			: TEXCOORD1;
};

VertexShaderOutput PositionVS(VertexShaderInput input){
	VertexShaderOutput output;
	
	output.Position = mul(input.Position * ScaleFactor, WorldViewProj);

	output.texC = input.Position;
	output.pos = output.Position;

	return output;
}