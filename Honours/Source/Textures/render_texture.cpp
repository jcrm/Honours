////////////////////////////////////////////////////////////////////////////////
// Filename: rendertextureclass.cpp
////////////////////////////////////////////////////////////////////////////////
#include "render_texture.h"
RenderTextureClass::RenderTextureClass(){
	render_target_texture_ = 0;
	render_target_view_ = 0;
	shader_resource_view_ = 0;
	depth_stencil_buffer_ = 0;
	depth_stencil_view_ = 0;
}
RenderTextureClass::RenderTextureClass(const RenderTextureClass& other){
}
RenderTextureClass::~RenderTextureClass(){
}
bool RenderTextureClass::Initialize(ID3D11Device* device, int textureWidth, int textureHeight, float screen_depth, float screen_near){
	D3D11_TEXTURE2D_DESC texture_desc;
	HRESULT result;
	D3D11_RENDER_TARGET_VIEW_DESC render_target_view_desc;
	D3D11_SHADER_RESOURCE_VIEW_DESC shader_resource_view_desc;
	D3D11_TEXTURE2D_DESC depth_buffer_desc;
	D3D11_DEPTH_STENCIL_VIEW_DESC depth_stencil_view_desc;
	// Store the width and height of the render texture.
	texture_width_ = textureWidth;
	texture_height_ = textureHeight;
	// Initialize the render target texture description.
	ZeroMemory(&texture_desc, sizeof(texture_desc));
	// Setup the render target texture description.
	texture_desc.Width = textureWidth;
	texture_desc.Height = textureHeight;
	texture_desc.MipLevels = 1;
	texture_desc.ArraySize = 1;
	texture_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	texture_desc.SampleDesc.Count = 1;
	texture_desc.Usage = D3D11_USAGE_DEFAULT;
	texture_desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	texture_desc.CPUAccessFlags = 0;
    texture_desc.MiscFlags = 0;
	// Create the render target texture.
	result = device->CreateTexture2D(&texture_desc, NULL, &render_target_texture_);
	if(FAILED(result)){
		return false;
	}
	// Setup the description of the render target view.
	render_target_view_desc.Format = texture_desc.Format;
	render_target_view_desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
	render_target_view_desc.Texture2D.MipSlice = 0;
	// Create the render target view.
	result = device->CreateRenderTargetView(render_target_texture_, &render_target_view_desc, &render_target_view_);
	if(FAILED(result)){
		return false;
	}
	// Setup the description of the shader resource view.
	shader_resource_view_desc.Format = texture_desc.Format;
	shader_resource_view_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	shader_resource_view_desc.Texture2D.MostDetailedMip = 0;
	shader_resource_view_desc.Texture2D.MipLevels = 1;
	// Create the shader resource view.
	result = device->CreateShaderResourceView(render_target_texture_, &shader_resource_view_desc, &shader_resource_view_);
	if(FAILED(result)){
		return false;
	}
	// Initialize the description of the depth buffer.
	ZeroMemory(&depth_buffer_desc, sizeof(depth_buffer_desc));
	// Set up the description of the depth buffer.
	depth_buffer_desc.Width = textureWidth;
	depth_buffer_desc.Height = textureHeight;
	depth_buffer_desc.MipLevels = 1;
	depth_buffer_desc.ArraySize = 1;
	depth_buffer_desc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	depth_buffer_desc.SampleDesc.Count = 1;
	depth_buffer_desc.SampleDesc.Quality = 0;
	depth_buffer_desc.Usage = D3D11_USAGE_DEFAULT;
	depth_buffer_desc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	depth_buffer_desc.CPUAccessFlags = 0;
	depth_buffer_desc.MiscFlags = 0;
	// Create the texture for the depth buffer using the filled out description.
	result = device->CreateTexture2D(&depth_buffer_desc, NULL, &depth_stencil_buffer_);
	if(FAILED(result)){
		return false;
	}
	// Initialize the depth stencil view.
	ZeroMemory(&depth_stencil_view_desc, sizeof(depth_stencil_view_desc));
	// Set up the depth stencil view description.
	depth_stencil_view_desc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	depth_stencil_view_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	depth_stencil_view_desc.Texture2D.MipSlice = 0;
	// Create the depth stencil view.
	result = device->CreateDepthStencilView(depth_stencil_buffer_, &depth_stencil_view_desc, &depth_stencil_view_);
	if(FAILED(result)){
		return false;
	}
	// Setup the viewport for rendering.
    viewport_.Width = (float)textureWidth;
    viewport_.Height = (float)textureHeight;
    viewport_.MinDepth = 0.0f;
    viewport_.MaxDepth = 1.0f;
    viewport_.TopLeftX = 0.0f;
    viewport_.TopLeftY = 0.0f;
	// Setup the projection matrix.
	D3DXMatrixPerspectiveFovLH(&projection_matrix_, ((float)D3DX_PI / 4.0f), ((float)textureWidth / (float)textureHeight), screen_near, screen_depth);
	// Create an orthographic projection matrix for 2D rendering.
	D3DXMatrixOrthoLH(&ortho_matrix_, (float)textureWidth, (float)textureHeight, screen_near, screen_depth);
	return true;
}
void RenderTextureClass::Shutdown(){
	if(depth_stencil_view_){
		depth_stencil_view_->Release();
		depth_stencil_view_ = 0;
	}
	if(depth_stencil_buffer_){
		depth_stencil_buffer_->Release();
		depth_stencil_buffer_ = 0;
	}
	if(shader_resource_view_){
		shader_resource_view_->Release();
		shader_resource_view_ = 0;
	}
	if(render_target_view_){
		render_target_view_->Release();
		render_target_view_ = 0;
	}
	if(render_target_texture_){
		render_target_texture_->Release();
		render_target_texture_ = 0;
	}
	return;
}
void RenderTextureClass::SetRenderTarget(ID3D11DeviceContext* device_context){
	// Bind the render target view and depth stencil buffer to the output render pipeline.
	device_context->OMSetRenderTargets(1, &render_target_view_, depth_stencil_view_);
	
	// Set the viewport.
    device_context->RSSetViewports(1, &viewport_);
	return;
}
void RenderTextureClass::ClearRenderTarget(ID3D11DeviceContext* device_context, float red, float green, float blue, float alpha){
	float color[4];
	// Setup the color to clear the buffer to.
	color[0] = red;
	color[1] = green;
	color[2] = blue;
	color[3] = alpha;
	// Clear the back buffer.
	device_context->ClearRenderTargetView(render_target_view_, color);
    
	// Clear the depth buffer.
	device_context->ClearDepthStencilView(depth_stencil_view_, D3D11_CLEAR_DEPTH, 1.0f, 0);
	return;
}
ID3D11ShaderResourceView* RenderTextureClass::GetShaderResourceView(){
	return shader_resource_view_;
}
void RenderTextureClass::GetProjectionMatrix(D3DXMATRIX& projection_matrix){
	projection_matrix = projection_matrix_;
	return;
}
void RenderTextureClass::GetOrthoMatrix(D3DXMATRIX& ortho_matrix){
	ortho_matrix = ortho_matrix_;
	return;
}
int RenderTextureClass::GetTextureWidth(){
	return texture_width_;
}
int RenderTextureClass::GetTextureHeight(){
	return texture_height_;
}