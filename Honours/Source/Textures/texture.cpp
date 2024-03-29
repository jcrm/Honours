////////////////////////////////////////////////////////////////////////////////
// Filename: textureclass.cpp
////////////////////////////////////////////////////////////////////////////////
#include "texture.h"
TextureClass::TextureClass(){
	texture_ = 0;
}
TextureClass::TextureClass(const TextureClass& other){
}
TextureClass::~TextureClass(){
}
bool TextureClass::Initialize(ID3D11Device* device, WCHAR* filename){
	HRESULT result;
	// Load the texture in.
	result = D3DX11CreateShaderResourceViewFromFile(device, filename, NULL, NULL, &texture_, NULL);
	if(FAILED(result)){
		return false;
	}
	return true;
}
void TextureClass::Shutdown(){
	// Release the texture resource.
	if(texture_){
		texture_->Release();
		texture_ = 0;
	}
	return;
}
ID3D11ShaderResourceView* TextureClass::GetTexture(){
	return texture_;
}