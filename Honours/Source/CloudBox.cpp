////////////////////////////////////////////////////////////////////////////////
// Filename: terrainclass.cpp
////////////////////////////////////////////////////////////////////////////////
#include "CLoudBox.h"
#include <cmath>
#include <math.h>

CloudClass::CloudClass(): m_vertexBuffer(0), m_indexBuffer(0)
{
}
CloudClass::CloudClass(const CloudClass& other): m_vertexBuffer(0), m_indexBuffer(0)
{
}
CloudClass::~CloudClass(){
}

bool CloudClass::Initialize(ID3D11Device* device){
	bool result;

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

	return;
}
bool CloudClass::InitializeBuffers(ID3D11Device* device){
	VertexType* vertices;
	unsigned long* indices;
	int index, i, j;
	D3D11_BUFFER_DESC vertexBufferDesc, indexBufferDesc;
    D3D11_SUBRESOURCE_DATA vertexData, indexData;
	HRESULT result;

	// Calculate the number of vertices's in the terrain mesh.
	m_vertexCount = 36;

	// Set the index count to the same as the vertex count.
	m_indexCount = m_vertexCount;

	// Create the vertex array.
	vertices = new VertexType[m_vertexCount];
	if(!vertices){
		return false;
	}

	// Create the index array.
	indices = new unsigned long[m_indexCount];
	if(!indices){
		return false;
	}

	// Initialize the index to the vertex buffer.
	index = 0;

	// Load the vertex and index array with the terrain data using a quilt method.
	//front -0
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(0.f, 0.f, 0.f);
	indices[index] = index++;
	//1
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(0.f, 1.f, 0.f);
	indices[index] = index++;
	//2
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(1.f, 1.f, 0.f);
	indices[index] = index++;
	//2
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(1.f, 1.f, 0.f);
	indices[index] = index++;
	//3
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(1.f, 0.f, 0.f);
	indices[index] = index++;
	//0
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(0.f, 0.f, 0.f);
	indices[index] = index++;

	//back -//7
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(1.f, 0.f, 1.f);
	indices[index] = index++;
	//6
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(1.f, 1.f, 1.f);
	indices[index] = index++;
	//5
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(0.f, 1.f, 1.f);
	indices[index] = index++;
	//5
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(0.f, 1.f, 1.f);
	indices[index] = index++;
	//4
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(0.f, 0.f, 1.f);
	indices[index] = index++;
	//7
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(1.f, 0.f, 1.f);
	indices[index] = index++;

	//left -//4
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(0.f, 0.f, 1.f);
	indices[index] = index++;
	//5
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(0.f, 1.f, 1.f);
	indices[index] = index++;
	//1
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(0.f, 1.f, 0.f);
	indices[index] = index++;
	//1
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(0.f, 1.f, 0.f);
	indices[index] = index++;
	//0
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(0.f, 0.f, 0.f);
	indices[index] = index++;
	//4
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(0.f, 0.f, 1.f);
	indices[index] = index++;

	//right -//3
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(1.f, 0.f, 0.f);
	indices[index] = index++;
	//2
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(1.f, 1.f, 0.f);
	indices[index] = index++;
	//6
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(1.f, 1.f, 1.f);
	indices[index] = index++;
	//6
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(1.f, 1.f, 1.f);
	indices[index] = index++;
	//7
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(1.f, 0.f, 1.f);
	indices[index] = index++;
	//3
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(1.f, 0.f, 0.f);
	indices[index] = index++;

	//Bottom -//4
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(0.f, 0.f, 1.f);
	indices[index] = index++;
	//0
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(0.f, 0.f, 0.f);
	indices[index] = index++;
	//3
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(1.f, 0.f, 0.f);
	indices[index] = index++;
	//3
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(1.f, 0.f, 0.f);
	indices[index] = index++;
	//7
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(1.f, 0.f, 1.f);
	indices[index] = index++;
	//4
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(0.f, 0.f, 1.f);
	indices[index] = index++;

	//Bottom -//1
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(0.f, 1.f, 0.f);
	indices[index] = index++;
	//5
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(0.f, 1.f, 1.f);
	indices[index] = index++;
	//6
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(1.f, 1.f, 1.f);
	indices[index] = index++;
	//6
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(1.f, 1.f, 1.f);
	indices[index] = index++;
	//2
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(1.f, 1.f, 0.f);
	indices[index] = index++;
	//1
	vertices[index].position = vertices[index].texture = D3DXVECTOR3(0.f, 1.f, 0.f);
	indices[index] = index++;

	// Set up the description of the static vertex buffer.
    vertexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
    vertexBufferDesc.ByteWidth = sizeof(VertexType) * m_vertexCount;
    vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    vertexBufferDesc.CPUAccessFlags = 0;
    vertexBufferDesc.MiscFlags = 0;
	vertexBufferDesc.StructureByteStride = 0;

	// Give the subresource structure a pointer to the vertex data.
    vertexData.pSysMem = vertices;
	vertexData.SysMemPitch = 0;
	vertexData.SysMemSlicePitch = 0;
	
	// Now create the vertex buffer.
    result = device->CreateBuffer(&vertexBufferDesc, &vertexData, &m_vertexBuffer);
	if(FAILED(result)){
		return false;
	}

	// Set up the description of the static index buffer.
    indexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
    indexBufferDesc.ByteWidth = sizeof(unsigned long) * m_indexCount;
    indexBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    indexBufferDesc.CPUAccessFlags = 0;
    indexBufferDesc.MiscFlags = 0;
	indexBufferDesc.StructureByteStride = 0;

	// Give the subresource structure a pointer to the index data.
    indexData.pSysMem = indices;
	indexData.SysMemPitch = 0;
	indexData.SysMemSlicePitch = 0;

	// Create the index buffer.
	result = device->CreateBuffer(&indexBufferDesc, &indexData, &m_indexBuffer);
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
	if(m_indexBuffer){
		m_indexBuffer->Release();
		m_indexBuffer = 0;
	}

	// Release the vertex buffer.
	if(m_vertexBuffer){
		m_vertexBuffer->Release();
		m_vertexBuffer = 0;
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
	deviceContext->IASetVertexBuffers(0, 1, &m_vertexBuffer, &stride, &offset);

    // Set the index buffer to active in the input assembler so it can be rendered.
	deviceContext->IASetIndexBuffer(m_indexBuffer, DXGI_FORMAT_R32_UINT, 0);

    // Set the type of primitive that should be rendered from this vertex buffer, in this case triangles.
	deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	return;
}