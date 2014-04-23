#include "shader.h"
ShaderClass::ShaderClass(void): vertex_shader_(0), pixel_shader_(0), layout_(0), 
	sample_state_(0), matrix_buffer_(0)
{
}
ShaderClass::~ShaderClass(void){
}
bool ShaderClass::Initialize(ID3D11Device*, HWND){
	return false;
}
void ShaderClass::Shutdown(){
		// Shutdown the vertex and pixel shaders as well as the related objects.
	ShutdownShader();
	return;
}
bool ShaderClass::Render(ID3D11DeviceContext*, int, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, ID3D11ShaderResourceView*, float, float){
	return false;
}
bool ShaderClass::Render(ID3D11DeviceContext*, int, D3DXMATRIX, ID3D11ShaderResourceView*, float, float){
	return false;
}
bool ShaderClass::InitializeShader(ID3D11Device*, HWND, WCHAR*, WCHAR*){
	return false;
}
void ShaderClass::ShutdownShader(){
		// Release the sampler state.
	if(sample_state_){
		sample_state_->Release();
		sample_state_ = 0;
	}
	// Release the matrix constant buffer.
	if(matrix_buffer_){
		matrix_buffer_->Release();
		matrix_buffer_ = 0;
	}
	// Release the layout.
	if(layout_){
		layout_->Release();
		layout_ = 0;
	}
	// Release the pixel shader.
	if(pixel_shader_){
		pixel_shader_->Release();
		pixel_shader_ = 0;
	}
	// Release the vertex shader.
	if(vertex_shader_){
		vertex_shader_->Release();
		vertex_shader_ = 0;
	}
	return;
}
void ShaderClass::OutputShaderErrorMessage(ID3D10Blob* error_message, HWND hwnd, WCHAR* shader_filename){
	char* compile_errors;
	unsigned long buffer_size, i;
	ofstream fout;
	// Get a pointer to the error message text buffer.
	compile_errors = (char*)(error_message->GetBufferPointer());
	// Get the length of the message.
	buffer_size = error_message->GetBufferSize();
	// Open a file to write the error message to.
	fout.open("shader-error.txt");
	// Write out the error message.
	for(i=0; i<buffer_size; i++)	{
		fout << compile_errors[i];
	}
	// Close the file.
	fout.close();
	// Release the error message.
	error_message->Release();
	error_message = 0;
	// Pop a message up on the screen to notify the user to check the text file for compile errors.
	MessageBox(hwnd, L"Error compiling shader.  Check shader-error.txt for message.", shader_filename, MB_OK);
	return;
}
bool ShaderClass::SetShaderParameters(ID3D11DeviceContext*, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, ID3D11ShaderResourceView*){
	return false;
}
void ShaderClass::RenderShader(ID3D11DeviceContext*, int){
}
