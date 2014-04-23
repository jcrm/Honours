///////////////////////////////////////////////////////////////////////////////
// Filename: textclass.cpp
///////////////////////////////////////////////////////////////////////////////
#include "textclass.h"
TextClass::TextClass()
{
	font_ = 0;
	sentence_one_ = 0;
	sentence_two_ = 0;
	sentence_three_ = 0;
	sentence_four_ = 0;
	sentence_five_ = 0;
	sentence_six_ = 0;
	sentence_seven_ = 0;
	sentence_eight_ = 0;
	sentence_nine_ = 0;
	sentence_tem_ = 0;
}
TextClass::TextClass(const TextClass& other)
{
}
TextClass::~TextClass()
{
}
bool TextClass::Initialize(ID3D11Device* device, ID3D11DeviceContext* device_context, HWND hwnd, int screen_width, int screen_height, 
						   D3DXMATRIX baseViewMatrix)
{
	bool result;
	// Store the screen width and height for calculating pixel location during the sentence updates.
	screen_width_ = screen_width;
	screen_height_ = screen_height;
	// Store the base view matrix for 2D text rendering.
	base_view_matrix_ = baseViewMatrix;
	// Create the font object.
	font_ = new FontClass;
	if(!font_)
	{
		return false;
	}
	// Initialize the font object.
	result = font_->Initialize(device, "Data/font_data.txt", L"Data/font.dds");
	if(!result)
	{
		MessageBox(hwnd, L"Could not initialize the font object.", L"Error", MB_OK);
		return false;
	}
	// Initialize the first sentence.
	result = InitializeSentence(&sentence_one_, 150, device);
	if(!result)
	{
		return false;
	}
	// Initialize the second sentence.
	result = InitializeSentence(&sentence_two_, 32, device);
	if(!result)
	{
		return false;
	}
	// Initialize the third sentence.
	result = InitializeSentence(&sentence_three_, 16, device);
	if(!result)
	{
		return false;
	}
	// Initialize the fourth sentence.
	result = InitializeSentence(&sentence_four_, 16, device);
	if(!result)
	{
		return false;
	}
	// Initialize the fifth sentence.
	result = InitializeSentence(&sentence_five_, 16, device);
	if(!result)
	{
		return false;
	}
	// Initialize the sixth sentence.
	result = InitializeSentence(&sentence_six_, 16, device);
	if(!result)
	{
		return false;
	}
	// Initialize the seventh sentence.
	result = InitializeSentence(&sentence_seven_, 16, device);
	if(!result)
	{
		return false;
	}
	// Initialize the eighth sentence.
	result = InitializeSentence(&sentence_eight_, 16, device);
	if(!result)
	{
		return false;
	}
	// Initialize the ninth sentence.
	result = InitializeSentence(&sentence_nine_, 16, device);
	if(!result)
	{
		return false;
	}
	// Initialize the tenth sentence.
	result = InitializeSentence(&sentence_tem_, 16, device);
	if(!result)
	{
		return false;
	}
	return true;
}
void TextClass::Shutdown()
{
	// Release the font object.
	if(font_)
	{
		font_->Shutdown();
		delete font_;
		font_ = 0;
	}
	// Release the sentences.
	ReleaseSentence(&sentence_one_);
	ReleaseSentence(&sentence_two_);
	ReleaseSentence(&sentence_three_);
	ReleaseSentence(&sentence_four_);
	ReleaseSentence(&sentence_five_);
	ReleaseSentence(&sentence_six_);
	ReleaseSentence(&sentence_seven_);
	ReleaseSentence(&sentence_eight_);
	ReleaseSentence(&sentence_nine_);
	ReleaseSentence(&sentence_tem_);
	return;
}
bool TextClass::Render(ID3D11DeviceContext* device_context, FontShaderClass* FontShader, D3DXMATRIX world_matrix, D3DXMATRIX ortho_matrix)
{
	bool result;
	// Draw the sentences.
	result = RenderSentence(sentence_one_, device_context, FontShader, world_matrix, ortho_matrix);
	if(!result)
	{
		return false;
	}
	result = RenderSentence(sentence_two_, device_context, FontShader, world_matrix, ortho_matrix);
	if(!result)
	{
		return false;
	}
	result = RenderSentence(sentence_three_, device_context, FontShader, world_matrix, ortho_matrix);
	if(!result)
	{
		return false;
	}
	result = RenderSentence(sentence_four_, device_context, FontShader, world_matrix, ortho_matrix);
	if(!result)
	{
		return false;
	}
	result = RenderSentence(sentence_five_, device_context, FontShader, world_matrix, ortho_matrix);
	if(!result)
	{
		return false;
	}
	result = RenderSentence(sentence_six_, device_context, FontShader, world_matrix, ortho_matrix);
	if(!result)
	{
		return false;
	}
	result = RenderSentence(sentence_seven_, device_context, FontShader, world_matrix, ortho_matrix);
	if(!result)
	{
		return false;
	}
	result = RenderSentence(sentence_eight_, device_context, FontShader, world_matrix, ortho_matrix);
	if(!result)
	{
		return false;
	}
	result = RenderSentence(sentence_nine_, device_context, FontShader, world_matrix, ortho_matrix);
	if(!result)
	{
		return false;
	}
	result = RenderSentence(sentence_tem_, device_context, FontShader, world_matrix, ortho_matrix);
	if(!result)
	{
		return false;
	}
	return true;
}
bool TextClass::InitializeSentence(SentenceType** sentence, int max_length_, ID3D11Device* device)
{
	VertexType* vertices;
	unsigned long* indices;
	D3D11_BUFFER_DESC vertex_buffer_desc, index_buffer_desc;
    D3D11_SUBRESOURCE_DATA vertex_data, index_data;
	HRESULT result;
	// Create a new sentence object.
	*sentence = new SentenceType;
	if(!*sentence)
	{
		return false;
	}
	// Initialize the sentence buffers to null.
	(*sentence)->vertex_buffer_ = 0;
	(*sentence)->index_buffer_ = 0;
	// Set the maximum length of the sentence.
	(*sentence)->max_length_ = max_length_;
	// Set the number of vertices in the vertex array.
	(*sentence)->vertex_count_ = 6 * max_length_;
	// Set the number of indexes in the index array.
	(*sentence)->index_count_ = (*sentence)->vertex_count_;
	// Create the vertex array.
	vertices = new VertexType[(*sentence)->vertex_count_];
	if(!vertices)
	{
		return false;
	}
	// Create the index array.
	indices = new unsigned long[(*sentence)->index_count_];
	if(!indices)
	{
		return false;
	}
	// Initialize vertex array to zeros at first.
	memset(vertices, 0, (sizeof(VertexType) * (*sentence)->vertex_count_));
	// Initialize the index array.
	for(int i=0; i<(*sentence)->index_count_; i++)
	{
		indices[i] = i;
	}
	// Set up the description of the dynamic vertex buffer.
    vertex_buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
    vertex_buffer_desc.ByteWidth = sizeof(VertexType) * (*sentence)->vertex_count_;
    vertex_buffer_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    vertex_buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    vertex_buffer_desc.MiscFlags = 0;
	vertex_buffer_desc.StructureByteStride = 0;
	// Give the subresource structure a pointer to the vertex data.
    vertex_data.pSysMem = vertices;
	vertex_data.SysMemPitch = 0;
	vertex_data.SysMemSlicePitch = 0;
	// Create the vertex buffer.
    result = device->CreateBuffer(&vertex_buffer_desc, &vertex_data, &(*sentence)->vertex_buffer_);
	if(FAILED(result))
	{
		return false;
	}
	// Set up the description of the static index buffer.
    index_buffer_desc.Usage = D3D11_USAGE_DEFAULT;
    index_buffer_desc.ByteWidth = sizeof(unsigned long) * (*sentence)->index_count_;
    index_buffer_desc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    index_buffer_desc.CPUAccessFlags = 0;
    index_buffer_desc.MiscFlags = 0;
	index_buffer_desc.StructureByteStride = 0;
	// Give the subresource structure a pointer to the index data.
    index_data.pSysMem = indices;
	index_data.SysMemPitch = 0;
	index_data.SysMemSlicePitch = 0;
	// Create the index buffer.
	result = device->CreateBuffer(&index_buffer_desc, &index_data, &(*sentence)->index_buffer_);
	if(FAILED(result))
	{
		return false;
	}
	// Release the vertex array as it is no longer needed.
	delete [] vertices;
	vertices = 0;
	// Release the index array as it is no longer needed.
	delete [] indices;
	indices = 0;
	return true;
}
bool TextClass::UpdateSentence(SentenceType* sentence, char* text, int position_x_, int position_y_, float red, float green, float blue,
							   ID3D11DeviceContext* device_context)
{
	int numLetters;
	VertexType* vertices;
	float drawX, drawY;
	HRESULT result;
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	VertexType* verticesPtr;
	// Store the color of the sentence.
	sentence->red_ = red;
	sentence->green_ = green;
	sentence->blue_ = blue;
	// Get the number of letters in the sentence.
	numLetters = (int)strlen(text);
	// Check for possible buffer overflow.
	if(numLetters > sentence->max_length_)
	{
		return false;
	}
	// Create the vertex array.
	vertices = new VertexType[sentence->vertex_count_];
	if(!vertices)
	{
		return false;
	}
	// Initialize vertex array to zeros at first.
	memset(vertices, 0, (sizeof(VertexType) * sentence->vertex_count_));
	// Calculate the X and Y pixel position on the screen to start drawing to.
	drawX = (float)(((screen_width_ / 2) * -1) + position_x_);
	drawY = (float)((screen_height_ / 2) - position_y_);
	// Use the font class to build the vertex array from the sentence text and sentence draw location.
	font_->BuildVertexArray((void*)vertices, text, drawX, drawY);
	// Lock the vertex buffer so it can be written to.
	result = device_context->Map(sentence->vertex_buffer_, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	if(FAILED(result))
	{
		return false;
	}
	// Get a pointer to the data in the vertex buffer.
	verticesPtr = (VertexType*)mappedResource.pData;
	// Copy the data into the vertex buffer.
	memcpy(verticesPtr, (void*)vertices, (sizeof(VertexType) * sentence->vertex_count_));
	// Unlock the vertex buffer.
	device_context->Unmap(sentence->vertex_buffer_, 0);
	// Release the vertex array as it is no longer needed.
	delete [] vertices;
	vertices = 0;
	return true;
}
void TextClass::ReleaseSentence(SentenceType** sentence)
{
	if(*sentence)
	{
		// Release the sentence vertex buffer.
		if((*sentence)->vertex_buffer_)
		{
			(*sentence)->vertex_buffer_->Release();
			(*sentence)->vertex_buffer_ = 0;
		}
		// Release the sentence index buffer.
		if((*sentence)->index_buffer_)
		{
			(*sentence)->index_buffer_->Release();
			(*sentence)->index_buffer_ = 0;
		}
		// Release the sentence.
		delete *sentence;
		*sentence = 0;
	}
	return;
}
bool TextClass::RenderSentence(SentenceType* sentence, ID3D11DeviceContext* device_context, FontShaderClass* FontShader, D3DXMATRIX world_matrix, 
							   D3DXMATRIX ortho_matrix)
{
	unsigned int stride, offset;
	D3DXVECTOR4 pixelColor;
	bool result;
	// Set vertex buffer stride and offset.
    stride = sizeof(VertexType); 
	offset = 0;
	// Set the vertex buffer to active in the input assembler so it can be rendered.
	device_context->IASetVertexBuffers(0, 1, &sentence->vertex_buffer_, &stride, &offset);
    // Set the index buffer to active in the input assembler so it can be rendered.
	device_context->IASetIndexBuffer(sentence->index_buffer_, DXGI_FORMAT_R32_UINT, 0);
    // Set the type of primitive that should be rendered from this vertex buffer, in this case triangles.
	device_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	// Create a pixel color vector with the input sentence color.
	pixelColor = D3DXVECTOR4(sentence->red_, sentence->green_, sentence->blue_, 1.0f);
	// Render the text using the font shader.
	result = FontShader->Render(device_context, sentence->index_count_, world_matrix, base_view_matrix_, ortho_matrix, font_->GetTexture(), pixelColor);
	if(!result)
	{
		false;
	}
	return true;
}
bool TextClass::SetVideoCardInfo(char* videoCardName, int videoCardMemory, ID3D11DeviceContext* device_context)
{
	char dataString[150];
	bool result;
	char tempString[16];
	char memoryString[32];
	// Setup the video card name string.
	strcpy_s(dataString, "Video Card: ");
	strcat_s(dataString, videoCardName);
	// Update the sentence vertex buffer with the new string information.
	result = UpdateSentence(sentence_one_, dataString, 10, 10, 1.0f, 1.0f, 1.0f, device_context);
	if(!result)
	{
		return false;
	}
	// Truncate the memory value to prevent buffer over flow.
	if(videoCardMemory > 9999999)
	{
		videoCardMemory = 9999999;
	}
	// Convert the video memory integer value to a string format.
	_itoa_s(videoCardMemory, tempString, 10);
	// Setup the video memory string.
	strcpy_s(memoryString, "Video Memory: ");
	strcat_s(memoryString, tempString);
	strcat_s(memoryString, " MB");
	// Update the sentence vertex buffer with the new string information.
	result = UpdateSentence(sentence_two_, memoryString, 10, 30, 1.0f, 1.0f, 1.0f, device_context);
	if(!result)
	{
		return false;
	}
	return true;
}
bool TextClass::SetFps(int fps, ID3D11DeviceContext* device_context)
{
	char tempString[16];
	char fpsString[16];
	bool result;
	// Truncate the fps to prevent a buffer over flow.
	if(fps > 9999)
	{
		fps = 9999;
	}
	// Convert the fps integer to string format.
	_itoa_s(fps, tempString, 10);
	// Setup the fps string.
	strcpy_s(fpsString, "Fps: ");
	strcat_s(fpsString, tempString);
	// Update the sentence vertex buffer with the new string information.
	result = UpdateSentence(sentence_three_, fpsString, 10, 70, 0.0f, 1.0f, 0.0f, device_context);
	if(!result)
	{
		return false;
	}
	return true;
}
bool TextClass::SetCpu(int cpu, ID3D11DeviceContext* device_context)
{
	char tempString[16];
	char cpuString[16];
	bool result;
	// Convert the cpu integer to string format.
	_itoa_s(cpu, tempString, 10);
	// Setup the cpu string.
	strcpy_s(cpuString, "Cpu: ");
	strcat_s(cpuString, tempString);
	strcat_s(cpuString, "%");
	// Update the sentence vertex buffer with the new string information.
	result = UpdateSentence(sentence_four_, cpuString, 10, 90, 0.0f, 1.0f, 0.0f, device_context);
	if(!result)
	{
		return false;
	}
	return true;
}
bool TextClass::SetCameraPosition(float posX, float posY, float posZ, ID3D11DeviceContext* device_context)
{
	int position_x_, position_y_, position_z_;
	char tempString[16];
	char dataString[16];
	bool result;
	// Convert the position from floating point to integer.
	position_x_ = (int)posX;
	position_y_ = (int)posY;
	position_z_ = (int)posZ;
	// Truncate the position if it exceeds either 9999 or -9999.
	if(position_x_ > 9999) { position_x_ = 9999; }
	if(position_y_ > 9999) { position_y_ = 9999; }
	if(position_z_ > 9999) { position_z_ = 9999; }
	if(position_x_ < -9999) { position_x_ = -9999; }
	if(position_y_ < -9999) { position_y_ = -9999; }
	if(position_z_ < -9999) { position_z_ = -9999; }
	// Setup the X position string.
	_itoa_s(position_x_, tempString, 10);
	strcpy_s(dataString, "X: ");
	strcat_s(dataString, tempString);
	result = UpdateSentence(sentence_five_, dataString, 10, 130, 0.0f, 1.0f, 0.0f, device_context);
	if(!result)
	{
		return false;
	}
	
	// Setup the Y position string.
	_itoa_s(position_y_, tempString, 10);
	strcpy_s(dataString, "Y: ");
	strcat_s(dataString, tempString);
	result = UpdateSentence(sentence_six_, dataString, 10, 150, 0.0f, 1.0f, 0.0f, device_context);
	if(!result)
	{
		return false;
	}
	// Setup the Z position string.
	_itoa_s(position_z_, tempString, 10);
	strcpy_s(dataString, "Z: ");
	strcat_s(dataString, tempString);
	result = UpdateSentence(sentence_seven_, dataString, 10, 170, 0.0f, 1.0f, 0.0f, device_context);
	if(!result)
	{
		return false;
	}
	return true;
}
bool TextClass::SetCameraRotation(float rotX, float rotY, float rotZ, ID3D11DeviceContext* device_context)
{
	int rotationX, rotationY, rotationZ;
	char tempString[16];
	char dataString[16];
	bool result;
	// Convert the rotation from floating point to integer.
	rotationX = (int)rotX;
	rotationY = (int)rotY;
	rotationZ = (int)rotZ;
	// Setup the X rotation string.
	_itoa_s(rotationX, tempString, 10);
	strcpy_s(dataString, "rX: ");
	strcat_s(dataString, tempString);
	result = UpdateSentence(sentence_eight_, dataString, 10, 210, 0.0f, 1.0f, 0.0f, device_context);
	if(!result)
	{
		return false;
	}
	// Setup the Y rotation string.
	_itoa_s(rotationY, tempString, 10);
	strcpy_s(dataString, "rY: ");
	strcat_s(dataString, tempString);
	result = UpdateSentence(sentence_nine_, dataString, 10, 230, 0.0f, 1.0f, 0.0f, device_context);
	if(!result)
	{
		return false;
	}
	// Setup the Z rotation string.
	_itoa_s(rotationZ, tempString, 10);
	strcpy_s(dataString, "rZ: ");
	strcat_s(dataString, tempString);
	result = UpdateSentence(sentence_tem_, dataString, 10, 250, 0.0f, 1.0f, 0.0f, device_context);
	if(!result)
	{
		return false;
	}
	return true;
}