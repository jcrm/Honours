////////////////////////////////////////////////////////////////////////////////
// Filename: terrainclass.cpp
////////////////////////////////////////////////////////////////////////////////
#include "cLoud_box.h"
#include <cmath>
#include <math.h>
CloudClass::CloudClass(): vertex_buffer_(0), index_buffer_(0)
{
	// Rotate the world matrix by the rotation value so that the cube will spin.
	D3DXMatrixRotationZ(&transform_, 1.57f);
}
CloudClass::CloudClass(const CloudClass& other): vertex_buffer_(0), index_buffer_(0)
{
}
CloudClass::~CloudClass(){
}
bool CloudClass::Initialize(ID3D11Device* device, int width, int height, float SCREEN_DEPTH, float SCREEN_NEAR){
	bool result;
	// Create the up sample render to texture object.
	m_FrontTexture = new RenderTextureClass;
	if(!m_FrontTexture){
		return false;
	}
	// Initialize the up sample render to texture object.
	result = m_FrontTexture->Initialize(device, width, height, SCREEN_DEPTH, SCREEN_NEAR);
	if(!result){
		return false;
	}
	m_BackTexture = new RenderTextureClass;
	if(!m_BackTexture){
		return false;
	}
	// Initialize the up sample render to texture object.
	result = m_BackTexture->Initialize(device, width, height, SCREEN_DEPTH, SCREEN_NEAR);
	if(!result){
		return false;
	}
	// Initialize the vertex and index buffer that hold the geometry for the terrain.
	result = InitializeBuffers(device);
	if(!result){
		return false;
	}
	return true;
}
void CloudClass::Shutdown(){
	ReleaseTexture();
	// Release the vertex and index buffer.
	ShutdownBuffers();
	return;
}
void CloudClass::Render(ID3D11DeviceContext* deviceContext){
	// Put the vertex and index buffers on the graphics pipeline to prepare them for drawing.
	RenderBuffers(deviceContext);
	return;
}
void CloudClass::ReleaseTexture(){
	// Release the texture object.
	if(m_FrontTexture){
		m_FrontTexture->Shutdown();
		delete m_FrontTexture;
		m_FrontTexture = 0;
	}
	if(m_BackTexture){
		m_BackTexture->Shutdown();
		delete m_BackTexture;
		m_BackTexture = 0;
	}
	return;
}
bool CloudClass::InitializeBuffers(ID3D11Device* device){
	VertexType* vertices;
	unsigned long* indices;
	int index;
	D3D11_BUFFER_DESC vertexBufferDesc, indexBufferDesc;
    D3D11_SUBRESOURCE_DATA vertexData, indexData;
	HRESULT result;
	// Calculate the number of vertices's in the terrain mesh.
	vertex_count_ = 36;
	// Set the index count to the same as the vertex count.
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
	// Initialize the index to the vertex buffer.
	index = 0;
	size = 1.f;
	// Load the vertex and index array with the terrain data using a quilt method.
	//front -0
	vertices[index].position_ = D3DXVECTOR3(0.f, 0.f, 0.f);
	vertices[index].texture = D3DXVECTOR3(0.f, 0.f, 0.f);
	indices[index] = index++;
	//1
	vertices[index].position_ = D3DXVECTOR3(0.f, size, 0.f);
	vertices[index].texture = D3DXVECTOR3(0.f, 1.f, 0.f);
	indices[index] = index++;
	//2
	vertices[index].position_ = D3DXVECTOR3(size, size, 0.f);
	vertices[index].texture = D3DXVECTOR3(1.f, 1.f, 0.f);
	indices[index] = index++;
	//2
	vertices[index].position_ = D3DXVECTOR3(size, size, 0.f);
	vertices[index].texture = D3DXVECTOR3(1.f, 1.f, 0.f);
	indices[index] = index++;
	//3
	vertices[index].position_ = D3DXVECTOR3(size, 0.f, 0.f);
	vertices[index].texture = D3DXVECTOR3(1.f, 0.f, 0.f);
	indices[index] = index++;
	//0
	vertices[index].position_ = D3DXVECTOR3(0.f, 0.f, 0.f);
	vertices[index].texture = D3DXVECTOR3(0.f, 0.f, 0.f);
	indices[index] = index++;
	//back -//7
	vertices[index].position_ = D3DXVECTOR3(size, 0.f, size);
	vertices[index].texture = D3DXVECTOR3(1.f, 0.f, 1.f);
	indices[index] = index++;
	//6
	vertices[index].position_ = D3DXVECTOR3(size, size, size);
	vertices[index].texture = D3DXVECTOR3(1.f, 1.f, 1.f);
	indices[index] = index++;
	//5
	vertices[index].position_ = D3DXVECTOR3(0.f, size, size);
	vertices[index].texture = D3DXVECTOR3(0.f, 1.f, 1.f);
	indices[index] = index++;
	//5
	vertices[index].position_ = D3DXVECTOR3(0.f, size, size);
	vertices[index].texture = D3DXVECTOR3(0.f, 1.f, 1.f);
	indices[index] = index++;
	//4
	vertices[index].position_ = D3DXVECTOR3(0.f, 0.f, size);
	vertices[index].texture = D3DXVECTOR3(0.f, 0.f, 1.f);
	indices[index] = index++;
	//7
	vertices[index].position_ = D3DXVECTOR3(size, 0.f, size);
	vertices[index].texture = D3DXVECTOR3(1.f, 0.f, 1.f);
	indices[index] = index++;
	//left -//4
	vertices[index].position_ = D3DXVECTOR3(0.f, 0.f, size);
	vertices[index].texture = D3DXVECTOR3(0.f, 0.f, 1.f);
	indices[index] = index++;
	//5
	vertices[index].position_ = D3DXVECTOR3(0.f, size, size);
	vertices[index].texture = D3DXVECTOR3(0.f, 1.f, 1.f);
	indices[index] = index++;
	//1
	vertices[index].position_ = D3DXVECTOR3(0.f, size, 0.f);
	vertices[index].texture = D3DXVECTOR3(0.f, 1.f, 0.f);
	indices[index] = index++;
	//1
	vertices[index].position_ =D3DXVECTOR3(0.f, size, 0.f);
	vertices[index].texture = D3DXVECTOR3(0.f, 1.f, 0.f);
	indices[index] = index++;
	//0
	vertices[index].position_ = D3DXVECTOR3(0.f, 0.f, 0.f);
	vertices[index].texture = D3DXVECTOR3(0.f, 0.f, 0.f);
	indices[index] = index++;
	//4
	vertices[index].position_ = D3DXVECTOR3(0.f, 0.f, size);
	vertices[index].texture = D3DXVECTOR3(0.f, 0.f, 1.f);
	indices[index] = index++;
	//right -//3
	vertices[index].position_ = D3DXVECTOR3(size, 0.f, 0.f);
	vertices[index].texture = D3DXVECTOR3(1.f, 0.f, 0.f);
	indices[index] = index++;
	//2
	vertices[index].position_ = D3DXVECTOR3(size, size, 0.f);
	vertices[index].texture = D3DXVECTOR3(1.f, 1.f, 0.f);
	indices[index] = index++;
	//6
	vertices[index].position_ = D3DXVECTOR3(size, size, size);
	vertices[index].texture = D3DXVECTOR3(1.f, 1.f, 1.f);
	indices[index] = index++;
	//6
	vertices[index].position_ = D3DXVECTOR3(size, size, size);
	vertices[index].texture = D3DXVECTOR3(1.f, 1.f, 1.f);
	indices[index] = index++;
	//7
	vertices[index].position_ = D3DXVECTOR3(size, 0.f, size);
	vertices[index].texture = D3DXVECTOR3(1.f, 0.f, 1.f);
	indices[index] = index++;
	//3
	vertices[index].position_ = D3DXVECTOR3(size, 0.f, 0.f);
	vertices[index].texture = D3DXVECTOR3(1.f, 0.f, 0.f);
	indices[index] = index++;
	//Bottom -//4
	vertices[index].position_ = D3DXVECTOR3(0.f, 0.f, size);
	vertices[index].texture = D3DXVECTOR3(0.f, 0.f, 1.f);
	indices[index] = index++;
	//0
	vertices[index].position_ =D3DXVECTOR3(0.f, 0.f, 0.f);
	vertices[index].texture = D3DXVECTOR3(0.f, 0.f, 0.f);
	indices[index] = index++;
	//3
	vertices[index].position_ = D3DXVECTOR3(size, 0.f, 0.f);
	vertices[index].texture = D3DXVECTOR3(1.f, 0.f, 0.f);
	indices[index] = index++;
	//3
	vertices[index].position_ = D3DXVECTOR3(size, 0.f, 0.f);
	vertices[index].texture = D3DXVECTOR3(1.f, 0.f, 0.f);
	indices[index] = index++;
	//7
	vertices[index].position_ = D3DXVECTOR3(size, 0.f, size);
	vertices[index].texture = D3DXVECTOR3(1.f, 0.f, 1.f);
	indices[index] = index++;
	//4
	vertices[index].position_ = D3DXVECTOR3(0.f, 0.f, size);
	vertices[index].texture = D3DXVECTOR3(0.f, 0.f, 1.f);
	indices[index] = index++;
	//Top -//1
	vertices[index].position_ = D3DXVECTOR3(0.f, size, 0.f);
	vertices[index].texture = D3DXVECTOR3(0.f, 1.f, 0.f);
	indices[index] = index++;
	//5
	vertices[index].position_ = D3DXVECTOR3(0.f, size, size);
	vertices[index].texture = D3DXVECTOR3(0.f, 1.f, 1.f);
	indices[index] = index++;
	//6
	vertices[index].position_ = D3DXVECTOR3(size, size, size);
	vertices[index].texture = D3DXVECTOR3(1.f, 1.f, 1.f);
	indices[index] = index++;
	//6
	vertices[index].position_ = D3DXVECTOR3(size, size, size);
	vertices[index].texture = D3DXVECTOR3(1.f, 1.f, 1.f);
	indices[index] = index++;
	//2
	vertices[index].position_ = D3DXVECTOR3(size, size, 0.f);
	vertices[index].texture = D3DXVECTOR3(1.f, 1.f, 0.f);
	indices[index] = index++;
	//1
	vertices[index].position_ = D3DXVECTOR3(0.f, size, 0.f);
	vertices[index].texture = D3DXVECTOR3(0.f, 1.f, 0.f);
	indices[index] = index++;
	// Set up the description of the static vertex buffer.
    vertexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
    vertexBufferDesc.ByteWidth = sizeof(VertexType) * vertex_count_;
    vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    vertexBufferDesc.CPUAccessFlags = 0;
    vertexBufferDesc.MiscFlags = 0;
	vertexBufferDesc.StructureByteStride = 0;
	// Give the subresource structure a pointer to the vertex data.
    vertexData.pSysMem = vertices;
	vertexData.SysMemPitch = 0;
	vertexData.SysMemSlicePitch = 0;
	
	// Now create the vertex buffer.
    result = device->CreateBuffer(&vertexBufferDesc, &vertexData, &vertex_buffer_);
	if(FAILED(result)){
		return false;
	}
	// Set up the description of the static index buffer.
    indexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
    indexBufferDesc.ByteWidth = sizeof(unsigned long) * index_count_;
    indexBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    indexBufferDesc.CPUAccessFlags = 0;
    indexBufferDesc.MiscFlags = 0;
	indexBufferDesc.StructureByteStride = 0;
	// Give the subresource structure a pointer to the index data.
    indexData.pSysMem = indices;
	indexData.SysMemPitch = 0;
	indexData.SysMemSlicePitch = 0;
	// Create the index buffer.
	result = device->CreateBuffer(&indexBufferDesc, &indexData, &index_buffer_);
	if(FAILED(result)){
		return false;
	}
	// Release the arrays now that the buffers have been created and loaded.
	delete [] vertices;
	vertices = 0;
	delete [] indices;
	indices = 0;
	return true;
}
void CloudClass::ShutdownBuffers(){
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
void CloudClass::RenderBuffers(ID3D11DeviceContext* deviceContext){
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