////////////////////////////////////////////////////////////////////////////////
// Filename: orthowindowclass.cpp
////////////////////////////////////////////////////////////////////////////////
#include "ortho_window.h"
OrthoWindowClass::OrthoWindowClass(){
	vertex_buffer_ = 0;
	index_buffer_ = 0;
}
OrthoWindowClass::OrthoWindowClass(const OrthoWindowClass& other){
}
OrthoWindowClass::~OrthoWindowClass(){
}
bool OrthoWindowClass::Initialize(ID3D11Device* device, int windowWidth, int windowHeight){
	bool result;
	// Initialize the vertex and index buffer that hold the geometry for the ortho window model.
	result = InitializeBuffers(device, windowWidth, windowHeight);
	if(!result){
		return false;
	}
	return true;
}
void OrthoWindowClass::Shutdown(){
	// Release the vertex and index buffers.
	ShutdownBuffers();
	return;
}
void OrthoWindowClass::Render(ID3D11DeviceContext* deviceContext){
	// Put the vertex and index buffers on the graphics pipeline to prepare them for drawing.
	RenderBuffers(deviceContext);
	return;
}
bool OrthoWindowClass::InitializeBuffers(ID3D11Device* device, int windowWidth, int windowHeight){
	float left, right, top, bottom;
	VertexType* vertices;
	unsigned long* indices;
	D3D11_BUFFER_DESC vertex_buffer_desc, index_buffer_desc;
	D3D11_SUBRESOURCE_DATA vertex_data, index_data;
	HRESULT result;
	/*
		Changed from using windowWidth and windowHeight to using +-1.
		This is because when adding convolution shader the projection,
		and view matrices caused problems with displaying textures to the screen.
	*/
	/*
	// Calculate the screen coordinates of the left side of the window.
	left = (float)((windowWidth / 2) * -1);
	// Calculate the screen coordinates of the right side of the window.
	right = left + (float)windowWidth;
	// Calculate the screen coordinates of the top of the window.
	top = (float)(windowHeight / 2);
	// Calculate the screen coordinates of the bottom of the window.
	bottom = top - (float)windowHeight;*/
	
	// Calculate the screen coordinates of the left side of the window.
	left = -1;
	// Calculate the screen coordinates of the right side of the window.
	right = 1;
	// Calculate the screen coordinates of the top of the window.
	top = 1;
	// Calculate the screen coordinates of the bottom of the window.
	bottom = -1;
	// Set the number of vertices in the vertex array.
	vertex_count_ = 6;
	// Set the number of indices in the index array.
	index_count_ = vertex_count_;
	// Create the vertex array.
	vertices = new VertexType[vertex_count_];
	if(!vertices){
		return false;
	}
	// Create the index array.
	indices = new unsigned long[index_count_];
	if(!indices){
		return false;
	}
	// Load the vertex array with data.
	// First triangle.
	vertices[0].position_ = D3DXVECTOR3(left, top, 0.0f);  // Top left.
	vertices[0].texture_ = D3DXVECTOR2(0.0f, 0.0f);
	vertices[1].position_ = D3DXVECTOR3(right, bottom, 0.0f);  // Bottom right.
	vertices[1].texture_ = D3DXVECTOR2(1.0f, 1.0f);
	vertices[2].position_ = D3DXVECTOR3(left, bottom, 0.0f);  // Bottom left.
	vertices[2].texture_ = D3DXVECTOR2(0.0f, 1.0f);
	// Second triangle.
	vertices[3].position_ = D3DXVECTOR3(left, top, 0.0f);  // Top left.
	vertices[3].texture_ = D3DXVECTOR2(0.0f, 0.0f);
	vertices[4].position_ = D3DXVECTOR3(right, top, 0.0f);  // Top right.
	vertices[4].texture_ = D3DXVECTOR2(1.0f, 0.0f);
	vertices[5].position_ = D3DXVECTOR3(right, bottom, 0.0f);  // Bottom right.
	vertices[5].texture_ = D3DXVECTOR2(1.0f, 1.0f);
	// Load the index array with data.
	for(int i=0; i<index_count_; i++){
		indices[i] = i;
	}
	// Set up the description of the vertex buffer.
    vertex_buffer_desc.Usage = D3D11_USAGE_DEFAULT;
    vertex_buffer_desc.ByteWidth = sizeof(VertexType) * vertex_count_;
    vertex_buffer_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    vertex_buffer_desc.CPUAccessFlags = 0;
    vertex_buffer_desc.MiscFlags = 0;
	vertex_buffer_desc.StructureByteStride = 0;
	// Give the subresource structure a pointer to the vertex data.
    vertex_data.pSysMem = vertices;
	vertex_data.SysMemPitch = 0;
	vertex_data.SysMemSlicePitch = 0;
	// Now finally create the vertex buffer.
    result = device->CreateBuffer(&vertex_buffer_desc, &vertex_data, &vertex_buffer_);
	if(FAILED(result)){
		return false;
	}
	// Set up the description of the index buffer.
    index_buffer_desc.Usage = D3D11_USAGE_DEFAULT;
    index_buffer_desc.ByteWidth = sizeof(unsigned long) * index_count_;
    index_buffer_desc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    index_buffer_desc.CPUAccessFlags = 0;
    index_buffer_desc.MiscFlags = 0;
	index_buffer_desc.StructureByteStride = 0;
	// Give the subresource structure a pointer to the index data.
    index_data.pSysMem = indices;
	index_data.SysMemPitch = 0;
	index_data.SysMemSlicePitch = 0;
	// Create the index buffer.
	result = device->CreateBuffer(&index_buffer_desc, &index_data, &index_buffer_);
	if(FAILED(result)){
		return false;
	}
	// Release the arrays now that the vertex and index buffers have been created and loaded.
	delete [] vertices;
	vertices = 0;
	delete [] indices;
	indices = 0;
	return true;
}
void OrthoWindowClass::ShutdownBuffers(){
	// Release the index buffer.
	if(index_buffer_){
		index_buffer_->Release();
		index_buffer_ = 0;
	}
	// Release the vertex buffer.
	if(vertex_buffer_){
		vertex_buffer_->Release();
		vertex_buffer_ = 0;
	}
	return;
}
void OrthoWindowClass::RenderBuffers(ID3D11DeviceContext* deviceContext){
	unsigned int stride;
	unsigned int offset;
	// Set vertex buffer stride and offset.
	stride = sizeof(VertexType); 
	offset = 0;
	// Set the vertex buffer to active in the input assembler so it can be rendered.
	deviceContext->IASetVertexBuffers(0, 1, &vertex_buffer_, &stride, &offset);
	// Set the index buffer to active in the input assembler so it can be rendered.
	deviceContext->IASetIndexBuffer(index_buffer_, DXGI_FORMAT_R32_UINT, 0);
	// Set the type of primitive that should be rendered from this vertex buffer, in this case triangles.
	deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	return;
}