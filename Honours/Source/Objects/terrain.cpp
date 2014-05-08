////////////////////////////////////////////////////////////////////////////////
// Filename: terrainclass.cpp
////////////////////////////////////////////////////////////////////////////////
#include "terrain.h"
TerrainClass::TerrainClass(){
	vertex_buffer_ = 0;
	index_buffer_ = 0;
	height_map_ = 0;
	texture_ = 0;
}
TerrainClass::TerrainClass(const TerrainClass& other){
}
TerrainClass::~TerrainClass(){
}
bool TerrainClass::Initialize(ID3D11Device* device, char* height_map_filename, WCHAR* texture_filename){
	bool result;
	// Load in the height map for the terrain.
	result = LoadHeightMap(height_map_filename);
	if(!result){
		return false;
	}
	// Normalize the height of the height map.
	NormalizeHeightMap();
	// Calculate the normals for the terrain data.
	result = CalculateNormals();
	if(!result){
		return false;
	}
	// Calculate the texture coordinates.
	CalculateTextureCoordinates();
	// Load the texture.
	result = LoadTexture(device, texture_filename);
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
void TerrainClass::Shutdown(){
	// Release the texture.
	ReleaseTexture();
	// Release the vertex and index buffer.
	ShutdownBuffers();
	// Release the height map data.
	ShutdownHeightMap();
	return;
}
void TerrainClass::Render(ID3D11DeviceContext* device_context){
	// Put the vertex and index buffers on the graphics pipeline to prepare them for drawing.
	RenderBuffers(device_context);
	return;
}
int TerrainClass::GetIndexCount(){
	return index_count_;
}
ID3D11ShaderResourceView* TerrainClass::GetTexture(){
	return texture_->GetTexture();
}
bool TerrainClass::LoadHeightMap(char* filename){
	FILE* filePtr;
	int error;
	unsigned int count;
	BITMAPFILEHEADER bitmapFileHeader;
	BITMAPINFOHEADER bitmapInfoHeader;
	int imageSize, k, index;
	unsigned char* bitmapImage;
	unsigned char height;
	// Open the height map file in binary.
	error = fopen_s(&filePtr, filename, "rb");
	if(error != 0){
		return false;
	}
	// Read in the file header.
	count = fread(&bitmapFileHeader, sizeof(BITMAPFILEHEADER), 1, filePtr);
	if(count != 1){
		return false;
	}
	// Read in the bitmap info header.
	count = fread(&bitmapInfoHeader, sizeof(BITMAPINFOHEADER), 1, filePtr);
	if(count != 1){
		return false;
	}
	// Save the dimensions of the terrain.
	terrain_width_ = bitmapInfoHeader.biWidth;
	terrain_height_ = bitmapInfoHeader.biHeight;
	// Calculate the size of the bitmap image data.
	imageSize = terrain_width_ * terrain_height_ * 3;
	// Allocate memory for the bitmap image data.
	bitmapImage = new unsigned char[imageSize];
	if(!bitmapImage){
		return false;
	}
	// Move to the beginning of the bitmap data.
	fseek(filePtr, bitmapFileHeader.bfOffBits, SEEK_SET);
	// Read in the bitmap image data.
	count = fread(bitmapImage, 1, imageSize, filePtr);
	if(count != imageSize){
		return false;
	}
	// Close the file.
	error = fclose(filePtr);
	if(error != 0){
		return false;
	}
	// Create the structure to hold the height map data.
	height_map_ = new HeightMapType[terrain_width_ * terrain_height_];
	if(!height_map_){
		return false;
	}
	// Initialize the position in the image data buffer.
	k=0;
	// Read the image data into the height map.
	for(int j=0; j<terrain_height_; j++){
		for(int i=0; i<terrain_width_; i++){
			height = bitmapImage[k];
			
			index = (terrain_height_ * j) + i;
			height_map_[index].x = (float)i;
			height_map_[index].y = (float)height;
			height_map_[index].z = (float)j;
			k+=3;
		}
	}
	// Release the bitmap image data.
	delete [] bitmapImage;
	bitmapImage = 0;
	return true;
}
void TerrainClass::NormalizeHeightMap(){
	for(int j=0; j<terrain_height_; j++){
		for(int i=0; i<terrain_width_; i++){
			height_map_[(terrain_height_ * j) + i].y /= 15.0f;
		}
	}
	return;
}
bool TerrainClass::CalculateNormals(){
	int index1, index2, index3, index, count;
	float vertex1[3], vertex2[3], vertex3[3], vector1[3], vector2[3], sum[3], length;
	VectorType* normals;
	// Create a temporary array to hold the un-normalized normal vectors.
	normals = new VectorType[(terrain_height_-1) * (terrain_width_-1)];
	if(!normals){
		return false;
	}
	// Go through all the faces in the mesh and calculate their normals.
	for(int j=0; j<(terrain_height_-1); j++){
		for(int i=0; i<(terrain_width_-1); i++){
			index1 = (j * terrain_height_) + i;
			index2 = (j * terrain_height_) + (i+1);
			index3 = ((j+1) * terrain_height_) + i;
			// Get three vertices from the face.
			vertex1[0] = height_map_[index1].x;
			vertex1[1] = height_map_[index1].y;
			vertex1[2] = height_map_[index1].z;
		
			vertex2[0] = height_map_[index2].x;
			vertex2[1] = height_map_[index2].y;
			vertex2[2] = height_map_[index2].z;
		
			vertex3[0] = height_map_[index3].x;
			vertex3[1] = height_map_[index3].y;
			vertex3[2] = height_map_[index3].z;
			// Calculate the two vectors for this face.
			vector1[0] = vertex1[0] - vertex3[0];
			vector1[1] = vertex1[1] - vertex3[1];
			vector1[2] = vertex1[2] - vertex3[2];
			vector2[0] = vertex3[0] - vertex2[0];
			vector2[1] = vertex3[1] - vertex2[1];
			vector2[2] = vertex3[2] - vertex2[2];
			index = (j * (terrain_height_-1)) + i;
			// Calculate the cross product of those two vectors to get the un-normalized value for this face normal.
			normals[index].x = (vector1[1] * vector2[2]) - (vector1[2] * vector2[1]);
			normals[index].y = (vector1[2] * vector2[0]) - (vector1[0] * vector2[2]);
			normals[index].z = (vector1[0] * vector2[1]) - (vector1[1] * vector2[0]);
		}
	}
	// Now go through all the vertices and take an average of each face normal 	
	// that the vertex touches to get the averaged normal for that vertex.
	for(int j=0; j<terrain_height_; j++){
		for(int i=0; i<terrain_width_; i++){
			// Initialize the sum.
			sum[0] = 0.0f;
			sum[1] = 0.0f;
			sum[2] = 0.0f;
			// Initialize the count.
			count = 0;
			// Bottom left face.
			if(((i-1) >= 0) && ((j-1) >= 0)){
				index = ((j-1) * (terrain_height_-1)) + (i-1);
				sum[0] += normals[index].x;
				sum[1] += normals[index].y;
				sum[2] += normals[index].z;
				count++;
			}
			// Bottom right face.
			if((i < (terrain_width_-1)) && ((j-1) >= 0)){
				index = ((j-1) * (terrain_height_-1)) + i;
				sum[0] += normals[index].x;
				sum[1] += normals[index].y;
				sum[2] += normals[index].z;
				count++;
			}
			// Upper left face.
			if(((i-1) >= 0) && (j < (terrain_height_-1))){
				index = (j * (terrain_height_-1)) + (i-1);
				sum[0] += normals[index].x;
				sum[1] += normals[index].y;
				sum[2] += normals[index].z;
				count++;
			}
			// Upper right face.
			if((i < (terrain_width_-1)) && (j < (terrain_height_-1))){
				index = (j * (terrain_height_-1)) + i;
				sum[0] += normals[index].x;
				sum[1] += normals[index].y;
				sum[2] += normals[index].z;
				count++;
			}
			
			// Take the average of the faces touching this vertex.
			sum[0] = (sum[0] / (float)count);
			sum[1] = (sum[1] / (float)count);
			sum[2] = (sum[2] / (float)count);
			// Calculate the length of this normal.
			length = sqrt((sum[0] * sum[0]) + (sum[1] * sum[1]) + (sum[2] * sum[2]));
			
			// Get an index to the vertex location in the height map array.
			index = (j * terrain_height_) + i;
			// Normalize the final shared normal for this vertex and store it in the height map array.
			height_map_[index].nx = (sum[0] / length);
			height_map_[index].ny = (sum[1] / length);
			height_map_[index].nz = (sum[2] / length);
		}
	}
	// Release the temporary normals.
	delete [] normals;
	normals = 0;
	return true;
}
void TerrainClass::ShutdownHeightMap(){
	if(height_map_){
		delete [] height_map_;
		height_map_ = 0;
	}
	return;
}
void TerrainClass::CalculateTextureCoordinates(){
	int incrementCount, tuCount, tvCount;
	float incrementValue, tuCoordinate, tvCoordinate;
	// Calculate how much to increment the texture coordinates by.
	incrementValue = (float)TEXTURE_REPEAT / (float)terrain_width_;
	// Calculate how many times to repeat the texture.
	incrementCount = terrain_width_ / TEXTURE_REPEAT;
	// Initialize the tu and tv coordinate values.
	tuCoordinate = 0.0f;
	tvCoordinate = 1.0f;
	// Initialize the tu and tv coordinate indexes.
	tuCount = 0;
	tvCount = 0;
	// Loop through the entire height map and calculate the tu and tv texture coordinates for each vertex.
	for(int j=0; j<terrain_height_; j++){
		for(int i=0; i<terrain_width_; i++){
			// Store the texture coordinate in the height map.
			height_map_[(terrain_height_ * j) + i].tu = tuCoordinate;
			height_map_[(terrain_height_ * j) + i].tv = tvCoordinate;
			// Increment the tu texture coordinate by the increment value and increment the index by one.
			tuCoordinate += incrementValue;
			tuCount++;
			// Check if at the far right end of the texture and if so then start at the beginning again.
			if(tuCount == incrementCount){
				tuCoordinate = 0.0f;
				tuCount = 0;
			}
		}
		// Increment the tv texture coordinate by the increment value and increment the index by one.
		tvCoordinate -= incrementValue;
		tvCount++;
		// Check if at the top of the texture and if so then start at the bottom again.
		if(tvCount == incrementCount){
			tvCoordinate = 1.0f;
			tvCount = 0;
		}
	}
	return;
}
bool TerrainClass::LoadTexture(ID3D11Device* device, WCHAR* filename){
	bool result;
	// Create the texture object.
	texture_ = new TextureClass;
	if(!texture_){
		return false;
	}
	// Initialize the texture object.
	result = texture_->Initialize(device, filename);
	if(!result){
		return false;
	}
	return true;
}
void TerrainClass::ReleaseTexture(){
	// Release the texture object.
	if(texture_){
		texture_->Shutdown();
		delete texture_;
		texture_ = 0;
	}
	return;
}
bool TerrainClass::InitializeBuffers(ID3D11Device* device){
	VertexType* vertices;
	unsigned long* indices;
	int index;
	D3D11_BUFFER_DESC vertex_buffer_desc, index_buffer_desc;
	D3D11_SUBRESOURCE_DATA vertex_data, index_data;
	HRESULT result;
	int index1, index2, index3, index4;
	float tu, tv;
	// Calculate the number of vertices in the terrain mesh.
	vertex_count_ = (terrain_width_ - 1) * (terrain_height_ - 1) * 6;
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
	// Load the vertex and index array with the terrain data.
	for(int j=0; j<(terrain_height_-1); j++){
		for(int i=0; i<(terrain_width_-1); i++){
			index1 = (terrain_height_ * j) + i;          // Bottom left.
			index2 = (terrain_height_ * j) + (i+1);      // Bottom right.
			index3 = (terrain_height_ * (j+1)) + i;      // Upper left.
			index4 = (terrain_height_ * (j+1)) + (i+1);  // Upper right.
			// Upper left.
			tv = height_map_[index3].tv;
			// Modify the texture coordinates to cover the top edge.
			if(tv == 1.0f){
				tv = 0.0f;
			}
			vertices[index].position_ = D3DXVECTOR3(height_map_[index3].x, height_map_[index3].y, height_map_[index3].z);
			vertices[index].texture_ = D3DXVECTOR2(height_map_[index3].tu, tv);
			vertices[index].normal_ = D3DXVECTOR3(height_map_[index3].nx, height_map_[index3].ny, height_map_[index3].nz);
			indices[index] = index;
			index++;
			// Upper right.
			tu = height_map_[index4].tu;
			tv = height_map_[index4].tv;
			// Modify the texture coordinates to cover the top and right edge.
			if(tu == 0.0f){
				tu = 1.0f;
			}
			if(tv == 1.0f){
				tv = 0.0f;
			}
			vertices[index].position_ = D3DXVECTOR3(height_map_[index4].x, height_map_[index4].y, height_map_[index4].z);
			vertices[index].texture_ = D3DXVECTOR2(tu, tv);
			vertices[index].normal_ = D3DXVECTOR3(height_map_[index4].nx, height_map_[index4].ny, height_map_[index4].nz);
			indices[index] = index;
			index++;
			// Bottom left.
			vertices[index].position_ = D3DXVECTOR3(height_map_[index1].x, height_map_[index1].y, height_map_[index1].z);
			vertices[index].texture_ = D3DXVECTOR2(height_map_[index1].tu, height_map_[index1].tv);
			vertices[index].normal_ = D3DXVECTOR3(height_map_[index1].nx, height_map_[index1].ny, height_map_[index1].nz);
			indices[index] = index;
			index++;
			// Bottom left.
			vertices[index].position_ = D3DXVECTOR3(height_map_[index1].x, height_map_[index1].y, height_map_[index1].z);
			vertices[index].texture_ = D3DXVECTOR2(height_map_[index1].tu, height_map_[index1].tv);
			vertices[index].normal_ = D3DXVECTOR3(height_map_[index1].nx, height_map_[index1].ny, height_map_[index1].nz);
			indices[index] = index;
			index++;
			// Upper right.
			tu = height_map_[index4].tu;
			tv = height_map_[index4].tv;
			// Modify the texture coordinates to cover the top and right edge.
			if(tu == 0.0f){
				tu = 1.0f;
			}
			if(tv == 1.0f){
				tv = 0.0f;
			}
			vertices[index].position_ = D3DXVECTOR3(height_map_[index4].x, height_map_[index4].y, height_map_[index4].z);
			vertices[index].texture_ = D3DXVECTOR2(tu, tv);
			vertices[index].normal_ = D3DXVECTOR3(height_map_[index4].nx, height_map_[index4].ny, height_map_[index4].nz);
			indices[index] = index;
			index++;
			// Bottom right.
			tu = height_map_[index2].tu;
			// Modify the texture coordinates to cover the right edge.
			if(tu == 0.0f){
				tu = 1.0f;
			}
			vertices[index].position_ = D3DXVECTOR3(height_map_[index2].x, height_map_[index2].y, height_map_[index2].z);
			vertices[index].texture_ = D3DXVECTOR2(tu, height_map_[index2].tv);
			vertices[index].normal_ = D3DXVECTOR3(height_map_[index2].nx, height_map_[index2].ny, height_map_[index2].nz);
			indices[index] = index;
			index++;
		}
	}
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
void TerrainClass::ShutdownBuffers(){
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
void TerrainClass::RenderBuffers(ID3D11DeviceContext* device_context){
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