////////////////////////////////////////////////////////////////////////////////
// Filename: terrainclass.cpp
////////////////////////////////////////////////////////////////////////////////
#include "cLoud_box.h"
#include <cmath>
#include <math.h>
CloudClass::CloudClass(): vertex_buffer_(0), index_buffer_(0)
{
	//size_ = D3DXVECTOR3(64.f,256.f,256.f);
	size_ = D3DXVECTOR3(1.f,1.f,1.f);
	// Rotate the world matrix by the rotation value so that the cube will spin.
	D3DXMATRIX rot;
	D3DXMATRIX trans;
	D3DXMATRIX scale;
	D3DXMatrixIdentity(&transform_);
	
	D3DXMatrixTranslation(&trans, -size_.x/2.f, -size_.y/2.f, -size_.z/2.f);
	D3DXMatrixMultiply(&transform_,&transform_,&trans);

	//D3DXMatrixRotationZ(&rot, -1.57f);
	//D3DXMatrixMultiply(&transform_,&transform_,&rot);

	D3DXMatrixTranslation(&trans, size_.y/2.f, size_.x/2.f, size_.z/2.f);
	D3DXMatrixMultiply(&transform_,&transform_,&trans);
	D3DXMatrixTranslation(&trans, 0.0f, 2.f, 0.0f);
	D3DXMatrixMultiply(&transform_,&transform_,&trans);
}
CloudClass::CloudClass(const CloudClass& other): vertex_buffer_(0), index_buffer_(0)
{
}
CloudClass::~CloudClass(){
}
bool CloudClass::Initialize(ID3D11Device* device, int width, int height, float SCREEN_DEPTH, float SCREEN_NEAR){
	bool result;
	// Create the up sample render to texture object.
	front_texture_ = new RenderTextureClass;
	if(!front_texture_){
		return false;
	}
	// Initialize the up sample render to texture object.
	result = front_texture_->Initialize(device, width, height, SCREEN_DEPTH, SCREEN_NEAR);
	if(!result){
		return false;
	}
	back_texture_ = new RenderTextureClass;
	if(!back_texture_){
		return false;
	}
	// Initialize the up sample render to texture object.
	result = back_texture_->Initialize(device, width, height, SCREEN_DEPTH, SCREEN_NEAR);
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
void CloudClass::Render(ID3D11DeviceContext* device_context){
	// Put the vertex and index buffers on the graphics pipeline to prepare them for drawing.
	RenderBuffers(device_context);
	return;
}
void CloudClass::ReleaseTexture(){
	// Release the texture object.
	if(front_texture_){
		front_texture_->Shutdown();
		delete front_texture_;
		front_texture_ = 0;
	}
	if(back_texture_){
		back_texture_->Shutdown();
		delete back_texture_;
		back_texture_ = 0;
	}
	return;
}
bool CloudClass::InitializeBuffers(ID3D11Device* device){
	VertexType* vertices;
	unsigned long* indices;
	int index;
	D3D11_BUFFER_DESC vertex_buffer_desc, index_buffer_desc;
	D3D11_SUBRESOURCE_DATA vertex_data, index_data;
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
	
	// Load the vertex and index array with the terrain data using a quilt method.
	//front -0
	vertices[index].position_ = D3DXVECTOR3(0.f, size_.y, 0.f);
	vertices[index].texture_ = D3DXVECTOR3(0.f, 0.f, 0.f);
	indices[index] = index++;
	//1
	vertices[index].position_ = D3DXVECTOR3(size_.x, size_.y, 0.f);
	vertices[index].texture_ = D3DXVECTOR3(1.f, 0.f, 0.f);
	indices[index] = index++;
	//2
	vertices[index].position_ = D3DXVECTOR3(size_.x, 0, 0.f);
	vertices[index].texture_ = D3DXVECTOR3(1.f, 1.f, 0.f);
	indices[index] = index++;
	//2
	vertices[index].position_ = D3DXVECTOR3(size_.x, 0, 0.f);
	vertices[index].texture_ = D3DXVECTOR3(1.f, 1.f, 0.f);
	indices[index] = index++;
	//3
	vertices[index].position_ = D3DXVECTOR3(0, 0.f, 0.f);
	vertices[index].texture_ = D3DXVECTOR3(0.f, 1.f, 0.f);
	indices[index] = index++;
	//0
	vertices[index].position_ = D3DXVECTOR3(0.f, size_.y, 0.f);
	vertices[index].texture_ = D3DXVECTOR3(0.f, 0.f, 0.f);
	indices[index] = index++;
	//back -//4
	vertices[index].position_ = D3DXVECTOR3(size_.x, size_.y, size_.z);
	vertices[index].texture_ = D3DXVECTOR3(1.f, 0.f, 1.f);
	indices[index] = index++;
	//5
	vertices[index].position_ = D3DXVECTOR3(0, size_.y, size_.z);
	vertices[index].texture_ = D3DXVECTOR3(0.f, 0.f, 1.f);
	indices[index] = index++;
	//6
	vertices[index].position_ = D3DXVECTOR3(0, 0, size_.z);
	vertices[index].texture_ = D3DXVECTOR3(0.f, 1.f, 1.f);
	indices[index] = index++;
	//6
	vertices[index].position_ = D3DXVECTOR3(0, 0, size_.z);
	vertices[index].texture_ = D3DXVECTOR3(0.f, 1.f, 1.f);
	indices[index] = index++;
	//7
	vertices[index].position_ = D3DXVECTOR3(size_.x, 0, size_.z);
	vertices[index].texture_ = D3DXVECTOR3(1.f, 1.f, 1.f);
	indices[index] = index++;
	//4
	vertices[index].position_ = D3DXVECTOR3(size_.x, size_.y, size_.z);
	vertices[index].texture_ = D3DXVECTOR3(1.f, 0.f, 1.f);
	indices[index] = index++;
	//left	//5
	vertices[index].position_ = D3DXVECTOR3(0, size_.y, size_.z);
	vertices[index].texture_ = D3DXVECTOR3(0.f, 0.f, 1.f);
	indices[index] = index++;
	//0
	vertices[index].position_ = D3DXVECTOR3(0.f, size_.y, 0.f);
	vertices[index].texture_ = D3DXVECTOR3(0.f, 0.f, 0.f);
	indices[index] = index++;
	//3
	vertices[index].position_ = D3DXVECTOR3(0, 0.f, 0.f);
	vertices[index].texture_ = D3DXVECTOR3(0.f, 1.f, 0.f);
	indices[index] = index++;
	//3
	vertices[index].position_ = D3DXVECTOR3(0, 0.f, 0.f);
	vertices[index].texture_ = D3DXVECTOR3(0.f, 1.f, 0.f);
	indices[index] = index++;
	//6
	vertices[index].position_ = D3DXVECTOR3(0, 0, size_.z);
	vertices[index].texture_ = D3DXVECTOR3(0.f, 1.f, 1.f);
	indices[index] = index++;
	//5
	vertices[index].position_ = D3DXVECTOR3(0, size_.y, size_.z);
	vertices[index].texture_ = D3DXVECTOR3(0.f, 0.f, 1.f);
	indices[index] = index++;
	//right	 //1
	vertices[index].position_ = D3DXVECTOR3(size_.x, size_.y, 0.f);
	vertices[index].texture_ = D3DXVECTOR3(1.f, 0.f, 0.f);
	indices[index] = index++;
	//4
	vertices[index].position_ = D3DXVECTOR3(size_.x, size_.y, size_.z);
	vertices[index].texture_ = D3DXVECTOR3(1.f, 0.f, 1.f);
	indices[index] = index++;
	//7
	vertices[index].position_ = D3DXVECTOR3(size_.x, 0, size_.z);
	vertices[index].texture_ = D3DXVECTOR3(1.f, 1, 1.f);
	indices[index] = index++;
	//7
	vertices[index].position_ = D3DXVECTOR3(size_.x, 0, size_.z);
	vertices[index].texture_ = D3DXVECTOR3(1.f, 1, 1.f);
	indices[index] = index++;
	//2
	vertices[index].position_ = D3DXVECTOR3(size_.x, 0, 0.f);
	vertices[index].texture_ = D3DXVECTOR3(1.f, 1.f, 0.f);
	indices[index] = index++;
	//1
	vertices[index].position_ = D3DXVECTOR3(size_.x, size_.y, 0.f);
	vertices[index].texture_ = D3DXVECTOR3(1.f, 0.f, 0.f);
	indices[index] = index++;
	//top	//5
	vertices[index].position_ = D3DXVECTOR3(0, size_.y, size_.z);
	vertices[index].texture_ = D3DXVECTOR3(0.f, 0.f, 1.f);
	indices[index] = index++;
	//4
	vertices[index].position_ = D3DXVECTOR3(size_.x, size_.y, size_.z);
	vertices[index].texture_ = D3DXVECTOR3(1.f, 0.f, 1.f);
	indices[index] = index++;
	//1
	vertices[index].position_ = D3DXVECTOR3(size_.x, size_.y, 0.f);
	vertices[index].texture_ = D3DXVECTOR3(1.f, 0.f, 0.f);
	indices[index] = index++;
	//1
	vertices[index].position_ = D3DXVECTOR3(size_.x, size_.y, 0.f);
	vertices[index].texture_ = D3DXVECTOR3(1.f, 0.f, 0.f);
	indices[index] = index++;
	//0
	vertices[index].position_ = D3DXVECTOR3(0.f, size_.y, 0.f);
	vertices[index].texture_ = D3DXVECTOR3(0.f, 0.f, 0.f);
	indices[index] = index++;
	//5
	vertices[index].position_ = D3DXVECTOR3(0, size_.y, size_.z);
	vertices[index].texture_ = D3DXVECTOR3(0.f, 0.f, 1.f);
	indices[index] = index++;
	//back	//3
	vertices[index].position_ = D3DXVECTOR3(0, 0.f, 0.f);
	vertices[index].texture_ = D3DXVECTOR3(0.f, 1.f, 0.f);
	indices[index] = index++;
	//2
	vertices[index].position_ = D3DXVECTOR3(size_.x, 0, 0.f);
	vertices[index].texture_ = D3DXVECTOR3(1.f, 1.f, 0.f);
	indices[index] = index++;
	//7
	vertices[index].position_ = D3DXVECTOR3(size_.x, 0, size_.z);
	vertices[index].texture_ = D3DXVECTOR3(1.f, 1, 1.f);
	indices[index] = index++;
	//7
	vertices[index].position_ = D3DXVECTOR3(size_.x, 0, size_.z);
	vertices[index].texture_ = D3DXVECTOR3(1.f, 1, 1.f);
	indices[index] = index++;
	//6
	vertices[index].position_ = D3DXVECTOR3(0, 0, size_.z);
	vertices[index].texture_ = D3DXVECTOR3(0.f, 1.f, 1.f);
	indices[index] = index++;
	//3
	vertices[index].position_ = D3DXVECTOR3(0, 0.f, 0.f);
	vertices[index].texture_ = D3DXVECTOR3(0.f, 1.f, 0.f);
	indices[index] = index++;
	
	// Set up the description of the static vertex buffer.
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
	
	// Now create the vertex buffer.
	result = device->CreateBuffer(&vertex_buffer_desc, &vertex_data, &vertex_buffer_);
	if(FAILED(result)){
		return false;
	}
	// Set up the description of the static index buffer.
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
void CloudClass::RenderBuffers(ID3D11DeviceContext* device_context){
	unsigned int stride;
	unsigned int offset;
	// Set vertex buffer stride and offset.
	stride = sizeof(VertexType); 
	offset = 0;
	
	// Set the vertex buffer to active in the input assembler so it can be rendered.
	device_context->IASetVertexBuffers(0, 1, &vertex_buffer_, &stride, &offset);
	// Set the index buffer to active in the input assembler so it can be rendered.
	device_context->IASetIndexBuffer(index_buffer_, DXGI_FORMAT_R32_UINT, 0);
	// Set the type of primitive that should be rendered from this vertex buffer, in this case triangles.
	device_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	return;
}