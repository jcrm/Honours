///////////////////////////////////////////////////////////////////////////////
// Filename: fontclass.cpp
///////////////////////////////////////////////////////////////////////////////
#include "fontclass.h"
FontClass::FontClass(){
	font_ = 0;
	texture_ = 0;
}
FontClass::FontClass(const FontClass& other){
}
FontClass::~FontClass(){
}
bool FontClass::Initialize(ID3D11Device* device, char* font_filename, WCHAR* texture_filename){
	bool result;
	// Load in the text file containing the font data.
	result = LoadFontData(font_filename);
	if(!result){
		return false;
	}
	// Load the texture that has the font characters on it.
	result = LoadTexture(device, texture_filename);
	if(!result){
		return false;
	}
	return true;
}
void FontClass::Shutdown(){
	// Release the font texture.
	ReleaseTexture();
	// Release the font data.
	ReleaseFontData();
	return;
}
bool FontClass::LoadFontData(char* filename){
	ifstream fin;
	char temp;
	// Create the font spacing buffer.
	font_ = new FontType[95];
	if(!font_)	{
		return false;
	}
	// Read in the font size and spacing between chars.
	fin.open(filename);
	if(fin.fail()){
		return false;
	}
	// Read in the 95 used ascii characters for text.
	for(int i=0; i<95; i++){
		fin.get(temp);
		while(temp != ' '){
			fin.get(temp);
		}
		fin.get(temp);
		while(temp != ' '){
			fin.get(temp);
		}
		fin >> font_[i].left_;
		fin >> font_[i].right_;
		fin >> font_[i].size_;
	}
	// Close the file.
	fin.close();
	return true;
}
void FontClass::ReleaseFontData(){
	// Release the font data array.
	if(font_){
		delete [] font_;
		font_ = 0;
	}
	return;
}
bool FontClass::LoadTexture(ID3D11Device* device, WCHAR* filename){
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
void FontClass::ReleaseTexture(){
	// Release the texture object.
	if(texture_){
		texture_->Shutdown();
		delete texture_;
		texture_ = 0;
	}
	return;
}
ID3D11ShaderResourceView* FontClass::GetTexture(){
	return texture_->GetTexture();
}
void FontClass::BuildVertexArray(void* vertices, char* sentence, float drawX, float drawY){
	VertexType* vertexPtr;
	int num_letters, index, letter;
	// Coerce the input vertices into a VertexType structure.
	vertexPtr = (VertexType*)vertices;
	// Get the number of letters in the sentence.
	num_letters = (int)strlen(sentence);
	// Initialize the index to the vertex array.
	index = 0;
	// Draw each letter onto a quad.
	for(int i=0; i<num_letters; i++){
		letter = ((int)sentence[i]) - 32;
		// If the letter is a space then just move over three pixels.
		if(letter == 0){
			drawX = drawX + 3.0f;
		}else{
			// First triangle in quad.
			vertexPtr[index].position_ = D3DXVECTOR3(drawX, drawY, 0.0f);  // Top left.
			vertexPtr[index].texture_ = D3DXVECTOR2(font_[letter].left_, 0.0f);
			index++;
			vertexPtr[index].position_ = D3DXVECTOR3((drawX + font_[letter].size_), (drawY - 16), 0.0f);  // Bottom right.
			vertexPtr[index].texture_ = D3DXVECTOR2(font_[letter].right_, 1.0f);
			index++;
			vertexPtr[index].position_ = D3DXVECTOR3(drawX, (drawY - 16), 0.0f);  // Bottom left.
			vertexPtr[index].texture_ = D3DXVECTOR2(font_[letter].left_, 1.0f);
			index++;
			// Second triangle in quad.
			vertexPtr[index].position_ = D3DXVECTOR3(drawX, drawY, 0.0f);  // Top left.
			vertexPtr[index].texture_ = D3DXVECTOR2(font_[letter].left_, 0.0f);
			index++;
			vertexPtr[index].position_ = D3DXVECTOR3(drawX + font_[letter].size_, drawY, 0.0f);  // Top right.
			vertexPtr[index].texture_ = D3DXVECTOR2(font_[letter].right_, 0.0f);
			index++;
			vertexPtr[index].position_ = D3DXVECTOR3((drawX + font_[letter].size_), (drawY - 16), 0.0f);  // Bottom right.
			vertexPtr[index].texture_ = D3DXVECTOR2(font_[letter].right_, 1.0f);
			index++;
			// Update the x location for drawing by the size of the letter and one pixel.
			drawX = drawX + font_[letter].size_ + 1.0f;
		}
	}
	return;
}