////////////////////////////////////////////////////////////////////////////////
// Filename: d3dclass.cpp
////////////////////////////////////////////////////////////////////////////////
#include "d3dclass.h"
D3DClass::D3DClass():swap_chain_(0), device_(0), device_context_(0),
	render_target_view_(0), depth_stencil_buffer_(0), depth_stencil_state_(0),
	depth_stencil_view_(0), raster_state_(0), depth_disabled_stencil_state_(0),
	alpha_enable_blending_state_(0), alpha_disable_blending_state_(0)
{
}
D3DClass::D3DClass(const D3DClass& other):swap_chain_(0), device_(0), device_context_(0),
	render_target_view_(0), depth_stencil_buffer_(0), depth_stencil_state_(0),
	depth_stencil_view_(0), raster_state_(0), depth_disabled_stencil_state_(0),
	alpha_enable_blending_state_(0), alpha_disable_blending_state_(0)
{
}
D3DClass::~D3DClass(){
}
bool D3DClass::Initialize(int screen_width, int screen_height, bool vsync, HWND hwnd, bool fullscreen, float screen_depth, float screen_near){
	HRESULT result;
	IDXGIFactory* factory;
	IDXGIAdapter* adapter;
	IDXGIOutput* adapter_output;
	unsigned int num_modes, i, numerator, denominator, string_length;
	DXGI_MODE_DESC* display_mode_list;
	DXGI_ADAPTER_DESC adapter_desc;
	int error;
	DXGI_SWAP_CHAIN_DESC swap_chain_desc;
	D3D_FEATURE_LEVEL feature_level;
	ID3D11Texture2D* back_buffer_ptr;
	D3D11_TEXTURE2D_DESC depth_buffer_desc;
	D3D11_DEPTH_STENCIL_DESC depth_stencil_desc;
	D3D11_DEPTH_STENCIL_VIEW_DESC depth_stencil_view_desc;
	D3D11_RASTERIZER_DESC raster_desc;
	D3D11_VIEWPORT viewport;
	float field_of_view, screen_aspect;
	D3D11_DEPTH_STENCIL_DESC depth_disabled_stencil_desc;
	D3D11_BLEND_DESC blend_state_desc;
	// Store the vsync setting.
	vsync_enabled_ = vsync;
	
	if(!InitDepthBuffer(screen_width, screen_height)){
		return false;
	}
	if(!InitSwapChain(hwnd,screen_width, screen_height,numerator,denominator,fullscreen)){
		return false;
	}
	// Get the pointer to the back buffer.
	result = swap_chain_->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&back_buffer_ptr);
	if(FAILED(result)){
		MessageBox(NULL,L"error11",NULL,NULL);
		return false;
	}
	// Create the render target view with the back buffer pointer.
	result = device_->CreateRenderTargetView(back_buffer_ptr, NULL, &render_target_view_);
	if(FAILED(result)){
		MessageBox(NULL,L"error12",NULL,NULL);
		return false;
	}
	// Release pointer to the back buffer as we no longer need it.
	back_buffer_ptr->Release();
	back_buffer_ptr = 0;
	if(!InitDepthBuffer(screen_width, screen_height)){
		return false;
	}
	if(!InitDepthStencil()){
		return false;
	}
	// Set the depth stencil state.
	device_context_->OMSetDepthStencilState(depth_stencil_state_, 1);
	// Initialize the depth stencil view.
	ZeroMemory(&depth_stencil_view_desc, sizeof(depth_stencil_view_desc));
	// Set up the depth stencil view description.
	depth_stencil_view_desc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	depth_stencil_view_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	depth_stencil_view_desc.Texture2D.MipSlice = 0;
	// Create the depth stencil view.
	result = device_->CreateDepthStencilView(depth_stencil_buffer_, &depth_stencil_view_desc, &depth_stencil_view_);
	if(FAILED(result)){
		MessageBox(NULL,L"error15",NULL,NULL);
		return false;
	}
	// Bind the render target view and depth stencil buffer to the output render pipeline.
	device_context_->OMSetRenderTargets(1, &render_target_view_, depth_stencil_view_);
	CreateRaster();
	// Setup the viewport for rendering.
	viewport.Width = (float)screen_width;
	viewport.Height = (float)screen_height;
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;
	viewport.TopLeftX = 0.0f;
	viewport.TopLeftY = 0.0f;
	// Create the viewport.
	device_context_->RSSetViewports(1, &viewport);
	// Setup the projection matrix.
	field_of_view = (float)D3DX_PI / 4.0f;
	screen_aspect = (float)screen_width / (float)screen_height;
	// Create the projection matrix for 3D rendering.
	D3DXMatrixPerspectiveFovLH(&projection_matrix_, field_of_view, screen_aspect, screen_near, screen_depth);
	// Initialize the world matrix to the identity matrix.
	D3DXMatrixIdentity(&world_matrix_);
	// Create an orthographic projection matrix for 2D rendering.
	D3DXMatrixOrthoLH(&ortho_matrix_, (float)screen_width, (float)screen_height, screen_near, screen_depth);
	if(!InitDepthDisableStencil()){
		exit(EXIT_SUCCESS);
	}
	if(!InitBlendState()){
		exit(EXIT_SUCCESS);
	}
	return true;
}
void D3DClass::Shutdown(){
	// Before shutting down set to windowed mode or when you release the swap chain it will throw an exception.
	if(swap_chain_){
		swap_chain_->SetFullscreenState(false, NULL);
	}
	if(alpha_enable_blending_state_){
		alpha_enable_blending_state_->Release();
		alpha_enable_blending_state_ = 0;
	}
	if(alpha_disable_blending_state_){
		alpha_disable_blending_state_->Release();
		alpha_disable_blending_state_ = 0;
	}
	if(depth_disabled_stencil_state_){
		depth_disabled_stencil_state_->Release();
		depth_disabled_stencil_state_ = 0;
	}
	if(raster_state_){
		raster_state_->Release();
		raster_state_ = 0;
	}
	if(depth_stencil_view_){
		depth_stencil_view_->Release();
		depth_stencil_view_ = 0;
	}
	if(depth_stencil_state_){
		depth_stencil_state_->Release();
		depth_stencil_state_ = 0;
	}
	if(depth_stencil_buffer_){
		depth_stencil_buffer_->Release();
		depth_stencil_buffer_ = 0;
	}
	if(render_target_view_){
		render_target_view_->Release();
		render_target_view_ = 0;
	}
	if(device_context_){
		device_context_->Release();
		device_context_ = 0;
	}
	if(device_){
		device_->Release();
		device_ = 0;
	}
	if(swap_chain_){
		swap_chain_->Release();
		swap_chain_ = 0;
	}
	return;
}
void D3DClass::BeginScene(float red, float green, float blue, float alpha){
	float color[4];
	// Setup the color to clear the buffer to.
	color[0] = red;
	color[1] = green;
	color[2] = blue;
	color[3] = alpha;
	// Clear the back buffer.
	device_context_->ClearRenderTargetView(render_target_view_, color);
	// Clear the depth buffer.
	device_context_->ClearDepthStencilView(depth_stencil_view_, D3D11_CLEAR_DEPTH, 1.0f, 0);
	return;
}
void D3DClass::EndScene(){
	// Present the back buffer to the screen since rendering is complete.
	if(vsync_enabled_){
		// Lock to screen refresh rate.
		swap_chain_->Present(1, 0);
	}else{
		// Present as fast as possible.
		swap_chain_->Present(0, 0);
	}
	return;
}
void D3DClass::GetProjectionMatrix(D3DXMATRIX& projection_matrix){
	projection_matrix = projection_matrix_;
}
void D3DClass::GetWorldMatrix(D3DXMATRIX& world_matrix){
	world_matrix = world_matrix_;
}
void D3DClass::GetOrthoMatrix(D3DXMATRIX& ortho_matrix){
	ortho_matrix = ortho_matrix_;
}
void D3DClass::GetVideoCardInfo(char* card_name, int& memory){
	strcpy_s(card_name, 128, video_card_description_);
	memory = video_card_memory_;
}
void D3DClass::TurnZBufferOn(){
	device_context_->OMSetDepthStencilState(depth_stencil_state_, 1);
}
void D3DClass::TurnZBufferOff(){
	device_context_->OMSetDepthStencilState(depth_disabled_stencil_state_, 1);
}
void D3DClass::TurnOnAlphaBlending(){
	float blend_factor[4];
	// Setup the blend factor.
	blend_factor[0] = 0.0f;
	blend_factor[1] = 0.0f;
	blend_factor[2] = 0.0f;
	blend_factor[3] = 0.0f;
	// Turn on the alpha blending.
	device_context_->OMSetBlendState(alpha_enable_blending_state_, blend_factor, 0xffffffff);
	return;
}
void D3DClass::TurnOffAlphaBlending(){
	float blend_factor[4];
	// Setup the blend factor.
	blend_factor[0] = 0.0f;
	blend_factor[1] = 0.0f;
	blend_factor[2] = 0.0f;
	blend_factor[3] = 0.0f;
	// Turn off the alpha blending.
	device_context_->OMSetBlendState(alpha_disable_blending_state_, blend_factor, 0xffffffff);
	return;
}
void D3DClass::EnableAlphaBlending(){
	float blend_factor[4];
	// Setup the blend factor.
	blend_factor[0] = 0.0f;
	blend_factor[1] = 0.0f;
	blend_factor[2] = 0.0f;
	blend_factor[3] = 0.0f;
	// Turn on the alpha blending.
	device_context_->OMSetBlendState(alpha_enable_additional_blending_state_, blend_factor, 0xffffffff);
	return;
}
void D3DClass::DisableAlphaBlending(){
	float blend_factor[4];
	// Setup the blend factor.
	blend_factor[0] = 0.0f;
	blend_factor[1] = 0.0f;
	blend_factor[2] = 0.0f;
	blend_factor[3] = 0.0f;
	// Turn off the alpha blending.
	device_context_->OMSetBlendState(alpha_disable_additional_blending_state_, blend_factor, 0xffffffff);
	return;
}
void D3DClass::SetBackBufferRenderTarget(){
	// Bind the render target view and depth stencil buffer to the output render pipeline.
	device_context_->OMSetRenderTargets(1, &render_target_view_, depth_stencil_view_);
	return;
}
void D3DClass::ResetViewport(){
	// Set the viewport.
	device_context_->RSSetViewports(1, &viewport_);
	return;
}
bool D3DClass::InitDisplayMode(int screen_width, int screen_height, unsigned int &numerator, unsigned int &denominator){
	HRESULT result;
	IDXGIOutput* adapter_output;
	unsigned int num_modes, string_length;
	DXGI_MODE_DESC* display_mode_list;
	DXGI_ADAPTER_DESC adapter_desc;
	IDXGIFactory* factory;
	IDXGIAdapter* adapter;
	// Create a DirectX graphics interface factory.
	result = CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&factory);
	if(FAILED(result)){
		MessageBox(NULL,L"error1",NULL,NULL);
		return false;
	}
	// Enumerate the primary adapter output (monitor).
	result = adapter->EnumOutputs(0, &adapter_output);
	if(FAILED(result)){
		return false;
	}
	// Get the number of modes that fit the DXGI_FORMAT_R8G8B8A8_UNORM display format for the adapter output (monitor).
	result = adapter_output->GetDisplayModeList(DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_ENUM_MODES_INTERLACED, &num_modes, NULL);
	if(FAILED(result)){
		return false;
	}
	// Create a list to hold all the possible display modes for this monitor/video card combination.
	display_mode_list = new DXGI_MODE_DESC[num_modes];
	if(!display_mode_list){
		return false;
	}
	// Now fill the display mode list structures.
	result = adapter_output->GetDisplayModeList(DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_ENUM_MODES_INTERLACED, &num_modes, display_mode_list);
	if(FAILED(result)){
		return false;
	}
	// Now go through all the display modes and find the one that matches the screen width and height.
	// When a match is found store the numerator and denominator of the refresh rate for that monitor.
	for(int i=0; i<num_modes; i++){
		if(display_mode_list[i].Width == (unsigned int)screen_width){
			if(display_mode_list[i].Height == (unsigned int)screen_height){
				numerator = display_mode_list[i].RefreshRate.Numerator;
				denominator = display_mode_list[i].RefreshRate.Denominator;
			}
		}
	}
	// Release the display mode list.
	delete [] display_mode_list;
	display_mode_list = 0;
	// Release the adapter output.
	adapter_output->Release();
	adapter_output = 0;
	// Release the adapter.
	adapter->Release();
	adapter = 0;
	// Release the factory.
	factory->Release();
	factory = 0;
	return true;
}
bool D3DClass::InitSwapChain(HWND hwnd, int screen_width, int screen_height, unsigned int& numerator, unsigned int& denominator, bool fullscreen){
	DXGI_SWAP_CHAIN_DESC swap_chain_desc;
	HRESULT result;
	// Initialize the swap chain description.
	ZeroMemory(&swap_chain_desc, sizeof(swap_chain_desc));
	// Set to a single back buffer.
	swap_chain_desc.BufferCount = 1;
	// Set the width and height of the back buffer.
	swap_chain_desc.BufferDesc.Width = screen_width;
	swap_chain_desc.BufferDesc.Height = screen_height;
	// Set regular 32-bit surface for the back buffer.
	swap_chain_desc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	// Set the refresh rate of the back buffer.
	if(vsync_enabled_){
		swap_chain_desc.BufferDesc.RefreshRate.Numerator = numerator;
		swap_chain_desc.BufferDesc.RefreshRate.Denominator = denominator;
	}else{
		swap_chain_desc.BufferDesc.RefreshRate.Numerator = 0;
		swap_chain_desc.BufferDesc.RefreshRate.Denominator = 1;
	}
	// Set the usage of the back buffer.
	swap_chain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	// Set the handle for the window to render to.
	swap_chain_desc.OutputWindow = hwnd;
	// Turn multisampling off.
	swap_chain_desc.SampleDesc.Count = 1;
	swap_chain_desc.SampleDesc.Quality = 0;
	// Set to full screen or windowed mode.
	if(fullscreen){
		swap_chain_desc.Windowed = false;
	}else{
		swap_chain_desc.Windowed = true;
	}
	// Set the scan line ordering and scaling to unspecified.
	swap_chain_desc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
	swap_chain_desc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
	// Discard the back buffer contents after presenting.
	swap_chain_desc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
	// Don't set the advanced flags.
	swap_chain_desc.Flags = 0;
	// Set the feature level to DirectX 11.
	D3D_FEATURE_LEVEL feature_level[] =
	{
		D3D_FEATURE_LEVEL_11_0,
		D3D_FEATURE_LEVEL_10_1,
		D3D_FEATURE_LEVEL_10_0
	};
	D3D_FEATURE_LEVEL feature_level_resource;
	result = D3D11CreateDeviceAndSwapChain(
		NULL,
		D3D_DRIVER_TYPE_UNKNOWN,
		NULL, 0, feature_level,
		3, D3D11_SDK_VERSION,
		&swap_chain_desc,
		&swap_chain_, &device_,
		&feature_level_resource, &device_context_);
	if(FAILED(result)){
		return false;
	}
	
	return true;
}
bool D3DClass::InitDepthBuffer(int screen_width, int screen_height){
	HRESULT result;
	D3D11_TEXTURE2D_DESC depth_buffer_desc;
	// Initialize the description of the depth buffer.
	ZeroMemory(&depth_buffer_desc, sizeof(depth_buffer_desc));
	// Set up the description of the depth buffer.
	depth_buffer_desc.Width = screen_width;
	depth_buffer_desc.Height = screen_height;
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
	result = device_->CreateTexture2D(&depth_buffer_desc, NULL, &depth_stencil_buffer_);
	if(FAILED(result)){
		return false;
	}
	return true;
}
bool D3DClass::InitDepthStencil(){
	HRESULT result;
	D3D11_DEPTH_STENCIL_DESC depth_stencil_desc;
	// Initialize the description of the stencil state.
	ZeroMemory(&depth_stencil_desc, sizeof(depth_stencil_desc));
	// Set up the description of the stencil state.
	depth_stencil_desc.DepthEnable = true;
	depth_stencil_desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	depth_stencil_desc.DepthFunc = D3D11_COMPARISON_LESS;
	depth_stencil_desc.StencilEnable = true;
	depth_stencil_desc.StencilReadMask = 0xFF;
	depth_stencil_desc.StencilWriteMask = 0xFF;
	// Stencil operations if pixel is front-facing.
	depth_stencil_desc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
	depth_stencil_desc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_INCR;
	depth_stencil_desc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
	depth_stencil_desc.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
	// Stencil operations if pixel is back-facing.
	depth_stencil_desc.BackFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
	depth_stencil_desc.BackFace.StencilDepthFailOp = D3D11_STENCIL_OP_DECR;
	depth_stencil_desc.BackFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
	depth_stencil_desc.BackFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
	// Create the depth stencil state.
	result = device_->CreateDepthStencilState(&depth_stencil_desc, &depth_stencil_state_);
	if(FAILED(result)){
		return false;
	}
	// Set the depth stencil state.
	device_context_->OMSetDepthStencilState(depth_stencil_state_, 1);
	return true;
}
bool D3DClass::InitDepthDisableStencil(){
	HRESULT result;
	D3D11_DEPTH_STENCIL_DESC depth_disabled_stencil_desc;
	// Clear the second depth stencil state before setting the parameters.
	ZeroMemory(&depth_disabled_stencil_desc, sizeof(depth_disabled_stencil_desc));
	// Now create a second depth stencil state which turns off the Z buffer for 2D rendering.  The only difference is 
	// that DepthEnable is set to false, all other parameters are the same as the other depth stencil state.
	depth_disabled_stencil_desc.DepthEnable = false;
	depth_disabled_stencil_desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	depth_disabled_stencil_desc.DepthFunc = D3D11_COMPARISON_LESS;
	depth_disabled_stencil_desc.StencilEnable = true;
	depth_disabled_stencil_desc.StencilReadMask = 0xFF;
	depth_disabled_stencil_desc.StencilWriteMask = 0xFF;
	depth_disabled_stencil_desc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
	depth_disabled_stencil_desc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_INCR;
	depth_disabled_stencil_desc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
	depth_disabled_stencil_desc.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
	depth_disabled_stencil_desc.BackFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
	depth_disabled_stencil_desc.BackFace.StencilDepthFailOp = D3D11_STENCIL_OP_DECR;
	depth_disabled_stencil_desc.BackFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
	depth_disabled_stencil_desc.BackFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
	// Create the state using the device.
	result = device_->CreateDepthStencilState(&depth_disabled_stencil_desc, &depth_disabled_stencil_state_);
	if(FAILED(result)){
		return false;
	}
	return true;
}
bool D3DClass::InitBlendState(){
	HRESULT result;
	D3D11_BLEND_DESC blend_state_desc;
	// Clear the blend state description.
	ZeroMemory(&blend_state_desc, sizeof(D3D11_BLEND_DESC));
	// Create an alpha enabled blend state description.
	blend_state_desc.RenderTarget[0].BlendEnable = TRUE;
	blend_state_desc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
	blend_state_desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
	blend_state_desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
	blend_state_desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
	blend_state_desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
	blend_state_desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
	blend_state_desc.RenderTarget[0].RenderTargetWriteMask = 0x0f;
	// Create the blend state using the description.
	result = device_->CreateBlendState(&blend_state_desc, &alpha_enable_blending_state_);
	if(FAILED(result)){
		return false;
	}
	// Modify the description to create an alpha disabled blend state description.
	blend_state_desc.RenderTarget[0].BlendEnable = FALSE;
	// Create the second blend state using the description.
	result = device_->CreateBlendState(&blend_state_desc, &alpha_disable_blending_state_);
	if(FAILED(result)){
		return false;
	}

	D3D11_BLEND_DESC blend_add_state_desc;
	// Clear the blend state description.
	ZeroMemory(&blend_add_state_desc, sizeof(D3D11_BLEND_DESC));
	// Create an alpha enabled blend state description.
	blend_add_state_desc.RenderTarget[0].BlendEnable = TRUE;
	blend_add_state_desc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
	blend_add_state_desc.RenderTarget[0].DestBlend = D3D11_BLEND_ONE;
	blend_add_state_desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
	blend_add_state_desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
	blend_add_state_desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
	blend_add_state_desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
	blend_add_state_desc.RenderTarget[0].RenderTargetWriteMask = 0x0f;
	// Create the blend state using the description.
	result = device_->CreateBlendState(&blend_add_state_desc, &alpha_enable_additional_blending_state_);
	if(FAILED(result)){
		return false;
	}
	// Modify the description to create an alpha disabled blend state description.
	blend_add_state_desc.RenderTarget[0].BlendEnable = FALSE;
	// Create the second blend state using the description.
	result = device_->CreateBlendState(&blend_add_state_desc, &alpha_disable_additional_blending_state_);
	if(FAILED(result)){
		return false;
	}
	return true;
}
bool D3DClass::CreateRaster(){
	D3D11_RASTERIZER_DESC raster_desc;
	// Setup the raster description which will determine how and what polygons will be drawn.
	raster_desc.AntialiasedLineEnable = false;
	raster_desc.CullMode = D3D11_CULL_BACK;
	raster_desc.DepthBias = 0;
	raster_desc.DepthBiasClamp = 0.0f;
	raster_desc.DepthClipEnable = true;
	raster_desc.FillMode = D3D11_FILL_SOLID;
	raster_desc.FrontCounterClockwise = false;
	raster_desc.MultisampleEnable = false;
	raster_desc.ScissorEnable = false;
	raster_desc.SlopeScaledDepthBias = 0.0f;
	// Create the rasterizer state from the description we just filled out.
	bool result = device_->CreateRasterizerState(&raster_desc, &raster_state_);
	if(FAILED(result)){
		return false;
	}
	// Now set the rasterizer state.
	device_context_->RSSetState(raster_state_);
	return true;
}
bool D3DClass::CreateBackFaceRaster(){
	D3D11_RASTERIZER_DESC raster_desc;
	// Setup the raster description which will determine how and what polygons will be drawn.
	raster_desc.AntialiasedLineEnable = false;
	raster_desc.CullMode = D3D11_CULL_BACK;
	raster_desc.DepthBias = 0;
	raster_desc.DepthBiasClamp = 0.0f;
	raster_desc.DepthClipEnable = true;
	raster_desc.FillMode = D3D11_FILL_SOLID;
	raster_desc.FrontCounterClockwise = true;
	raster_desc.MultisampleEnable = false;
	raster_desc.ScissorEnable = false;
	raster_desc.SlopeScaledDepthBias = 0.0f;
	// Create the rasterizer state from the description we just filled out.
	bool result = device_->CreateRasterizerState(&raster_desc, &raster_state_);
	if(FAILED(result)){
		return false;
	}
	// Now set the rasterizer state.
	device_context_->RSSetState(raster_state_);
	return true;
}