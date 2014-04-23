////////////////////////////////////////////////////////////////////////////////
// Filename: particle.vs
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
    float2 tex : TEXCOORD0;
	float4 color : COLOR;
};

struct PixelInputType
{
    float4 position_ : SV_POSITION;
    float2 tex : TEXCOORD0;
	float4 color : COLOR;
};


////////////////////////////////////////////////////////////////////////////////
// Vertex Shader
////////////////////////////////////////////////////////////////////////////////
PixelInputType ParticleVertexShader(VertexInputType input)
{
    PixelInputType output;
    

	// Change the position vector to be 4 units for proper matrix calculations.
    input.position_.w = 1.0f;

	// Calculate the position of the vertex against the world, view, and projection matrices.
    output.position_ = mul(input.position_, world_matrix);
    output.position_ = mul(output.position_, viewMatrix);
    output.position_ = mul(output.position_, projection_matrix);
    
	// Store the texture coordinates for the pixel shader.
	output.tex = input.tex;
    
	// Store the particle color for the pixel shader. 
    output.color = input.color;

    return output;
}