////////////////////////////////////////////////////////////////////////////////
// Filename: particle.vs
////////////////////////////////////////////////////////////////////////////////

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
struct VertexInputType
{
	float4 position_ : POSITION;
	float2 tex_ : TEXCOORD0;
	float4 color_ : COLOR;
	float4 instance_position_ : TEXCOORD1;
};

struct PixelInputType
{
	float4 position_ : SV_POSITION;
	float2 tex_ : TEXCOORD0;
	float4 color_ : COLOR;
};

////////////////////////////////////////////////////////////////////////////////
// Vertex Shader
////////////////////////////////////////////////////////////////////////////////
PixelInputType ParticleVertexShader(VertexInputType input)
{
	PixelInputType output;

	// Update the position of the vertices based on the data for this particular instance.
	input.position_.x += input.instance_position_.x;
	input.position_.y += input.instance_position_.y;
	input.position_.z += input.instance_position_.z;
	// Change the position vector to be 4 units for proper matrix calculations.
	input.position_.w = 1.0f;

	// Calculate the position of the vertex against the world, view, and projection matrices.
	output.position_ = mul(input.position_, world_matrix_);
	output.position_ = mul(output.position_, view_matrix_);
	output.position_ = mul(output.position_, projection_matrix_);

	// Store the texture coordinates for the pixel shader.
	output.tex_ = input.tex_;

	// Store the particle color for the pixel shader. 
	output.color_ = input.color_;
	if(input.instance_position_.w == -1){
		output.color_ = float4(0.f, 0.f, 0.f, 0.f);
	}
	return output;
}