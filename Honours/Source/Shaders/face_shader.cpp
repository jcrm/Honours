#include "face_shader.h"
FaceShader::FaceShader(void){
}
FaceShader::~FaceShader(void){
}
bool FaceShader::Initialize(ID3D11Device* device, HWND hwnd){
	bool result;
	// Initialize the vertex and pixel shaders.
	result = InitializeShader(device, hwnd, L"Shader/Face/face.vs", L"Shader/Face/face.ps");
	if(!result){
		return false;
	}
	return true;
}
bool FaceShader::Render(ID3D11DeviceContext* device_context, int indexCount, D3DXMATRIX worldMatrix, D3DXMATRIX viewMatrix, 
						  D3DXMATRIX projectionMatrix, float scale){
	bool result;
	// Set the shader parameters that it will use for rendering.
	result = SetShaderParameters(device_context, worldMatrix, viewMatrix, projectionMatrix, scale);
	if(!result){
		return false;
	}
	// Now render the prepared buffers with the shader.
	RenderShader(device_context, indexCount);
	return true;
}
bool FaceShader::InitializeShader(ID3D11Device* device, HWND hwnd, WCHAR* vsFilename, WCHAR* psFilename){
	HRESULT result;
	ID3D10Blob* errorMessage;
	ID3D10Blob* vertexShaderBuffer;
	ID3D10Blob* pixelShaderBuffer;
	D3D11_INPUT_ELEMENT_DESC polygonLayout[2];
	unsigned int numElements;
	D3D11_BUFFER_DESC matrixBufferDesc;
	D3D11_BUFFER_DESC scaleBufferDesc;
	D3D11_SAMPLER_DESC samplerDesc;
	// Initialize the pointers this function will use to null.
	errorMessage = 0;
	vertexShaderBuffer = 0;
	pixelShaderBuffer = 0;
	// Compile the vertex shader code.
	result = D3DX11CompileFromFile(vsFilename, NULL, NULL, "FaceVS", "vs_5_0", D3D10_SHADER_ENABLE_STRICTNESS, 0, NULL, 
		&vertexShaderBuffer, &errorMessage, NULL);
	if(FAILED(result)){
		// If the shader failed to compile it should have writen something to the error message.
		if(errorMessage){
			OutputShaderErrorMessage(errorMessage, hwnd, vsFilename);
		}else{		// If there was nothing in the error message then it simply could not find the shader file itself.
			MessageBox(hwnd, vsFilename, L"Missing Shader File", MB_OK);
		}
		return false;
	}
	// Compile the pixel shader code.
	result = D3DX11CompileFromFile(psFilename, NULL, NULL, "FacePS", "ps_5_0", D3D10_SHADER_ENABLE_STRICTNESS, 0, NULL, 
		&pixelShaderBuffer, &errorMessage, NULL);
	if(FAILED(result)){
		// If the shader failed to compile it should have writen something to the error message.
		if(errorMessage){
			OutputShaderErrorMessage(errorMessage, hwnd, psFilename);
		}else{
			// If there was  nothing in the error message then it simply could not find the file itself.
			MessageBox(hwnd, psFilename, L"Missing Shader File", MB_OK);
		}
		return false;
	}
	// Create the vertex shader from the buffer.
	result = device->CreateVertexShader(vertexShaderBuffer->GetBufferPointer(), vertexShaderBuffer->GetBufferSize(), NULL, &vertex_shader_);
	if(FAILED(result)){
		return false;
	}
	// Create the pixel shader from the buffer.
	result = device->CreatePixelShader(pixelShaderBuffer->GetBufferPointer(), pixelShaderBuffer->GetBufferSize(), NULL, &pixel_shader_);
	if(FAILED(result)){
		return false;
	}
	// Create the vertex input layout description.
	polygonLayout[0].SemanticName = "POSITION";
	polygonLayout[0].SemanticIndex = 0;
	polygonLayout[0].Format = DXGI_FORMAT_R32G32B32_FLOAT;
	polygonLayout[0].InputSlot = 0;
	polygonLayout[0].AlignedByteOffset = 0;
	polygonLayout[0].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	polygonLayout[0].InstanceDataStepRate = 0;
	polygonLayout[1].SemanticName = "TEXCOORD";
	polygonLayout[1].SemanticIndex = 0;
	polygonLayout[1].Format = DXGI_FORMAT_R32G32_FLOAT;
	polygonLayout[1].InputSlot = 0;
	polygonLayout[1].AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
	polygonLayout[1].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	polygonLayout[1].InstanceDataStepRate = 0;
	// Get a count of the elements in the layout.
	numElements = sizeof(polygonLayout) / sizeof(polygonLayout[0]);
	// Create the vertex input layout.
	result = device->CreateInputLayout(polygonLayout, numElements, vertexShaderBuffer->GetBufferPointer(), vertexShaderBuffer->GetBufferSize(), &layout_);
	if(FAILED(result)){
		return false;
	}
	// Release the vertex shader buffer and pixel shader buffer since they are no longer needed.
	vertexShaderBuffer->Release();
	vertexShaderBuffer = 0;
	pixelShaderBuffer->Release();
	pixelShaderBuffer = 0;
	// Setup the description of the dynamic matrix constant buffer that is in the vertex shader.
	matrixBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
	matrixBufferDesc.ByteWidth = sizeof(MatrixBufferType);
	matrixBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	matrixBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	matrixBufferDesc.MiscFlags = 0;
	matrixBufferDesc.StructureByteStride = 0;
	// Create the constant buffer pointer so we can access the vertex shader constant buffer from within this class.
	result = device->CreateBuffer(&matrixBufferDesc, NULL, &matrix_buffer_);
	if(FAILED(result)){
		return false;
	}
	// Create a texture sampler state description.
	samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	samplerDesc.MipLODBias = 0.0f;
	samplerDesc.MaxAnisotropy = 1;
	samplerDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
	samplerDesc.BorderColor[0] = 0;
	samplerDesc.BorderColor[1] = 0;
	samplerDesc.BorderColor[2] = 0;
	samplerDesc.BorderColor[3] = 0;
	samplerDesc.MinLOD = 0;
	samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
	// Create the texture sampler state.
	result = device->CreateSamplerState(&samplerDesc, &sample_state_);
	if(FAILED(result)){
		return false;
	}
	// Setup the description of the light dynamic constant buffer that is in the pixel shader.
	// Note that ByteWidth always needs to be a multiple of 16 if using D3D11_BIND_CONSTANT_BUFFER or CreateBuffer will fail.
	scaleBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
	scaleBufferDesc.ByteWidth = sizeof(ScaleBufferType);
	scaleBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	scaleBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	scaleBufferDesc.MiscFlags = 0;
	scaleBufferDesc.StructureByteStride = 0;
	// Create the constant buffer pointer so we can access the vertex shader constant buffer from within this class.
	result = device->CreateBuffer(&scaleBufferDesc, NULL, &m_scaleBuffer);
	if(FAILED(result)){
		return false;
	}
	return true;
}
bool FaceShader::SetShaderParameters(ID3D11DeviceContext* device_context, D3DXMATRIX worldMatrix, D3DXMATRIX viewMatrix, 
									   D3DXMATRIX projectionMatrix, float scale){
	HRESULT result;
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	MatrixBufferType* dataPtr;
	ScaleBufferType* dataPtr2;
	unsigned int bufferNumber;
	// Transpose the matrices to prepare them for the shader.
	D3DXMatrixTranspose(&worldMatrix, &worldMatrix);
	D3DXMatrixTranspose(&viewMatrix, &viewMatrix);
	D3DXMatrixTranspose(&projectionMatrix, &projectionMatrix);
	// Lock the constant buffer so it can be written to.
	result = device_context->Map(matrix_buffer_, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	if(FAILED(result)){
		return false;
	}
	// Get a pointer to the data in the constant buffer.
	dataPtr = (MatrixBufferType*)mappedResource.pData;
	// Copy the matrices into the constant buffer.
	dataPtr->world = worldMatrix;
	dataPtr->view = viewMatrix;
	dataPtr->projection = projectionMatrix;
	// Unlock the constant buffer.
	device_context->Unmap(matrix_buffer_, 0);
	// Set the position of the constant buffer in the vertex shader.
	bufferNumber = 0;
	// Now set the constant buffer in the vertex shader with the updated values.
	device_context->VSSetConstantBuffers(bufferNumber, 1, &matrix_buffer_);
	
	result = device_context->Map(m_scaleBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	if(FAILED(result)){
		return false;
	}
	// Get a pointer to the data in the constant buffer.
	dataPtr2 = (ScaleBufferType*)mappedResource.pData;
	// Copy the lighting variables into the constant buffer.
	dataPtr2->scale = D3DXVECTOR4(scale,scale,scale,1.0f);
	// Unlock the constant buffer.
	device_context->Unmap(m_scaleBuffer, 0);
	// Set the position of the light constant buffer in the pixel shader.
	bufferNumber = 1;
	// Finally set the light constant buffer in the pixel shader with the updated values.
	device_context->VSSetConstantBuffers(bufferNumber, 1, &m_scaleBuffer);
	return true;
}
void FaceShader::RenderShader(ID3D11DeviceContext* device_context, int indexCount){
	// Set the vertex input layout.
	device_context->IASetInputLayout(layout_);
	// Set the vertex and pixel shaders that will be used to render this triangle.
	device_context->VSSetShader(vertex_shader_, NULL, 0);
	device_context->PSSetShader(pixel_shader_, NULL, 0);
	// Set the sampler state in the pixel shader.
	device_context->PSSetSamplers(0, 1, &sample_state_);
	// Render the triangle.
	device_context->DrawIndexed(indexCount, 0, 0);
	return;
}