#include "volume_shader.h"
VolumeShader::VolumeShader(void){
}
VolumeShader::~VolumeShader(void){
}
bool VolumeShader::Initialize(ID3D11Device* device, HWND hwnd){
	bool result;
	// Initialize the vertex and pixel shaders.
	result = InitializeShader(device, hwnd, L"Shader/Volume/volume.vs", L"Shader/Volume/volume.ps");
	if(!result){
		return false;
	}
	return true;
}
bool VolumeShader::Render(ID3D11DeviceContext* device_context, int index_count_, D3DXMATRIX world_matrix, D3DXMATRIX viewMatrix, 
						  D3DXMATRIX projection_matrix, ID3D11ShaderResourceView* frontTexture , ID3D11ShaderResourceView* backTexture,
						  ID3D11ShaderResourceView* volumeTexture, float scale){
	// Set the shader parameters that it will use for rendering.
	bool result = SetShaderParameters(device_context, world_matrix, viewMatrix, projection_matrix, frontTexture, backTexture, volumeTexture, scale);
	if(!result){
		return false;
	}
	// Now render the prepared buffers with the shader.
	RenderShader(device_context, index_count_);
	return true;
}
bool VolumeShader::InitializeShader(ID3D11Device* device, HWND hwnd, WCHAR* vs_filename, WCHAR* ps_filename){
	HRESULT result;
	ID3D10Blob* error_message = 0;
	ID3D10Blob* vertex_shader_buffer = 0;
	ID3D10Blob* pixel_shader_buffer = 0;;
	D3D11_INPUT_ELEMENT_DESC polygon_layout[2];
	unsigned int num_elements;
	D3D11_BUFFER_DESC matrix_buffer_desc;
	D3D11_BUFFER_DESC volume_buffer_desc;
	D3D11_SAMPLER_DESC sampler_desc;
	// Compile the vertex shader code.
	result = D3DX11CompileFromFile(vs_filename, NULL, NULL, "VolumeVS", "vs_5_0", D3D10_SHADER_ENABLE_STRICTNESS, 0, NULL, 
		&vertex_shader_buffer, &error_message, NULL);
	if(FAILED(result)){
		// If the shader failed to compile it should have writen something to the error message.
		if(error_message){
			OutputShaderErrorMessage(error_message, hwnd, vs_filename);
		}else{		// If there was nothing in the error message then it simply could not find the shader file itself.
			MessageBox(hwnd, vs_filename, L"Missing Shader File", MB_OK);
		}
		return false;
	}
	// Compile the pixel shader code.
	result = D3DX11CompileFromFile(ps_filename, NULL, NULL, "RayCastSimplePS", "ps_5_0", D3D10_SHADER_ENABLE_STRICTNESS, 0, NULL, 
		&pixel_shader_buffer, &error_message, NULL);
	if(FAILED(result)){
		// If the shader failed to compile it should have writen something to the error message.
		if(error_message){
			OutputShaderErrorMessage(error_message, hwnd, ps_filename);
		}else{
			// If there was  nothing in the error message then it simply could not find the file itself.
			MessageBox(hwnd, ps_filename, L"Missing Shader File", MB_OK);
		}
		return false;
	}
	// Create the vertex shader from the buffer.
	result = device->CreateVertexShader(vertex_shader_buffer->GetBufferPointer(), vertex_shader_buffer->GetBufferSize(), NULL, &vertex_shader_);
	if(FAILED(result)){
		return false;
	}
	// Create the pixel shader from the buffer.
	result = device->CreatePixelShader(pixel_shader_buffer->GetBufferPointer(), pixel_shader_buffer->GetBufferSize(), NULL, &pixel_shader_);
	if(FAILED(result)){
		return false;
	}
	// Create the vertex input layout description.
	polygon_layout[0].SemanticName = "POSITION";
	polygon_layout[0].SemanticIndex = 0;
	polygon_layout[0].Format = DXGI_FORMAT_R32G32B32_FLOAT;
	polygon_layout[0].InputSlot = 0;
	polygon_layout[0].AlignedByteOffset = 0;
	polygon_layout[0].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	polygon_layout[0].InstanceDataStepRate = 0;
	polygon_layout[1].SemanticName = "TEXCOORD";
	polygon_layout[1].SemanticIndex = 0;
	polygon_layout[1].Format = DXGI_FORMAT_R32G32_FLOAT;
	polygon_layout[1].InputSlot = 0;
	polygon_layout[1].AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
	polygon_layout[1].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	polygon_layout[1].InstanceDataStepRate = 0;
	// Get a count of the elements in the layout.
	num_elements = sizeof(polygon_layout) / sizeof(polygon_layout[0]);
	// Create the vertex input layout.
	result = device->CreateInputLayout(polygon_layout, num_elements, vertex_shader_buffer->GetBufferPointer(), vertex_shader_buffer->GetBufferSize(), 
		&layout_);
	if(FAILED(result)){
		return false;
	}
	// Release the vertex shader buffer and pixel shader buffer since they are no longer needed.
	vertex_shader_buffer->Release();
	vertex_shader_buffer = 0;
	pixel_shader_buffer->Release();
	pixel_shader_buffer = 0;
	// Setup the description of the dynamic matrix constant buffer that is in the vertex shader.
	matrix_buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
	matrix_buffer_desc.ByteWidth = sizeof(MatrixBufferType);
	matrix_buffer_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	matrix_buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	matrix_buffer_desc.MiscFlags = 0;
	matrix_buffer_desc.StructureByteStride = 0;
	// Create the constant buffer pointer so we can access the vertex shader constant buffer from within this class.
	result = device->CreateBuffer(&matrix_buffer_desc, NULL, &matrix_buffer_);
	if(FAILED(result)){
		return false;
	}
	// Create a texture sampler state description.
	sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
	sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
	sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	sampler_desc.MipLODBias = 0.0f;
	sampler_desc.MaxAnisotropy = 1;
	sampler_desc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
	sampler_desc.BorderColor[0] = 0;
	sampler_desc.BorderColor[1] = 0;
	sampler_desc.BorderColor[2] = 0;
	sampler_desc.BorderColor[3] = 0;
	sampler_desc.MinLOD = 0;
	sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;
	// Create the texture sampler state.
	result = device->CreateSamplerState(&sampler_desc, &sample_state_);
	if(FAILED(result)){
		return false;
	}
	// Setup the description of the light dynamic constant buffer that is in the pixel shader.
	// Note that ByteWidth always needs to be a multiple of 16 if using D3D11_BIND_CONSTANT_BUFFER or CreateBuffer will fail.
	volume_buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
	volume_buffer_desc.ByteWidth = sizeof(VolumeBufferType);
	volume_buffer_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	volume_buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	volume_buffer_desc.MiscFlags = 0;
	volume_buffer_desc.StructureByteStride = 0;
	// Create the constant buffer pointer so we can access the vertex shader constant buffer from within this class.
	result = device->CreateBuffer(&volume_buffer_desc, NULL, &volume_buffer_);
	if(FAILED(result)){
		return false;
	}
	return true;
}
bool VolumeShader::SetShaderParameters(ID3D11DeviceContext* device_context, D3DXMATRIX world_matrix, D3DXMATRIX viewMatrix, 
									   D3DXMATRIX projection_matrix, ID3D11ShaderResourceView* frontTexture, 
									   ID3D11ShaderResourceView* backTexture, ID3D11ShaderResourceView* volumeTexture, float scale){
	HRESULT result;
	D3D11_MAPPED_SUBRESOURCE mapped_resource;
	MatrixBufferType* data_ptr;
	VolumeBufferType* data_ptr_two;
	unsigned int buffer_number;
	// Transpose the matrices to prepare them for the shader.
	D3DXMatrixTranspose(&world_matrix, &world_matrix);
	D3DXMatrixTranspose(&viewMatrix, &viewMatrix);
	D3DXMatrixTranspose(&projection_matrix, &projection_matrix);
	// Lock the constant buffer so it can be written to.
	result = device_context->Map(matrix_buffer_, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped_resource);
	if(FAILED(result)){
		return false;
	}
	// Get a pointer to the data in the constant buffer.
	data_ptr = (MatrixBufferType*)mapped_resource.pData;
	// Copy the matrices into the constant buffer.
	data_ptr->world_ = world_matrix;
	data_ptr->view_ = viewMatrix;
	data_ptr->projection_ = projection_matrix;
	// Unlock the constant buffer.
	device_context->Unmap(matrix_buffer_, 0);
	// Set the position of the constant buffer in the vertex shader.
	buffer_number = 0;
	// Now set the constant buffer in the vertex shader with the updated values.
	device_context->VSSetConstantBuffers(buffer_number, 1, &matrix_buffer_);
	// Set shader texture resource in the pixel shader.
	device_context->PSSetShaderResources(0, 1, &frontTexture);
	device_context->PSSetShaderResources(1, 1, &backTexture);
	device_context->PSSetShaderResources(2, 1, &volumeTexture);
	
	result = device_context->Map(volume_buffer_, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped_resource);
	if(FAILED(result)){
		return false;
	}
	// Get a pointer to the data in the constant buffer.
	data_ptr_two = (VolumeBufferType*)mapped_resource.pData;
	float maxSize = 64.f;
	float mStepScale = 1.0f;
	// Copy the lighting variables into the constant buffer.
	data_ptr_two->scale_ = D3DXVECTOR4(scale,scale,scale,1.0f);
	data_ptr_two->iterations_ = (int)maxSize * (1.0f / mStepScale);
	data_ptr_two->step_size_ = D3DXVECTOR3(1.0f / 64.f, 1.0f / 64.f, 1.0f / 64.f);;
	
	// Unlock the constant buffer.
	device_context->Unmap(volume_buffer_, 0);
	// Set the position of the light constant buffer in the pixel shader.
	buffer_number = 1;
	// Finally set the light constant buffer in the pixel shader with the updated values.
	device_context->VSSetConstantBuffers(buffer_number, 1, &volume_buffer_);
	return true;
}
void VolumeShader::RenderShader(ID3D11DeviceContext* device_context, int index_count_){
	// Set the vertex input layout.
	device_context->IASetInputLayout(layout_);
	// Set the vertex and pixel shaders that will be used to render this triangle.
	device_context->VSSetShader(vertex_shader_, NULL, 0);
	device_context->PSSetShader(pixel_shader_, NULL, 0);
	// Set the sampler state in the pixel shader.
	device_context->PSSetSamplers(0, 1, &sample_state_);
	// Render the triangle.
	device_context->DrawIndexed(index_count_, 0, 0);
	return;
}