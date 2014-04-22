#include "cuda_d3d.h"
// This header inclues all the necessary D3D11 and CUDA includes
#include <dynlink_d3d11.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
// includes, project
#include <helper_functions.h>
#define NAME_LEN	512
CUDAD3D::CUDAD3D(void):D3DClass(), cuda_capable_adapter_(0)
{
}
CUDAD3D::CUDAD3D(const D3DClass& other):D3DClass(other), cuda_capable_adapter_(0)
{
}
CUDAD3D::~CUDAD3D(void){
}
bool CUDAD3D::InitDisplayMode(int screen_width, int screen_height, unsigned int &numerator, unsigned int &denominator){
	HRESULT result;
	IDXGIOutput* adapter_output;
	unsigned int num_modes;
	DXGI_MODE_DESC* display_mode_list;
	// Enumerate the primary adapter output (monitor).
	result = cuda_capable_adapter_->EnumOutputs(0, &adapter_output);
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
	for(int i=0; i<(int)num_modes; i++){
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
	return true;
}
bool CUDAD3D::InitSwapChain(HWND hwnd, int screen_width, int screen_height, unsigned int& numerator, unsigned int& denominator, bool fullscreen){
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
	if ( !cuda_capable_adapter_ ){
		MessageBox(NULL,L"errorcuda",NULL,NULL);
		return false;
	}
	D3D_FEATURE_LEVEL feature_level_resource;
	result = D3D11CreateDeviceAndSwapChain(
		cuda_capable_adapter_,
		D3D_DRIVER_TYPE_UNKNOWN,
		NULL, 0, feature_level,
		3, D3D11_SDK_VERSION,
		&swap_chain_desc,
		&swap_chain_, &device_,
		&feature_level_resource, &device_context_);
	if(FAILED(result)){
		return false;
	}
	// Release the adapter.
	cuda_capable_adapter_->Release();
	cuda_capable_adapter_ = 0;
	return true;
}
bool CUDAD3D::Initialize(int screen_width, int screen_height, bool vsync, HWND hwnd, bool fullscreen, float screen_depth, float screen_near){
	HRESULT result;
	ID3D11Texture2D* back_buffer_ptr;
	D3D11_DEPTH_STENCIL_VIEW_DESC depth_stencil_view_desc;
	unsigned int numerator, denominator;
	D3D11_VIEWPORT viewport;
	float field_of_view, screen_aspect;
	char device_name[256];
	
	// Store the vsync setting.
	vsync_enabled_ = vsync;
	if (!findCUDADevice() ){				// Search for CUDA GPU 
		exit(EXIT_SUCCESS);
	}
	if (!findDXDevice(device_name)){		// Search for D3D Hardware Device
		exit(EXIT_SUCCESS);
	}
	if(!InitDisplayMode(screen_width, screen_height, numerator, denominator)){
		exit(EXIT_SUCCESS);
	}
	if(!InitSwapChain(hwnd, screen_width, screen_height, numerator, denominator, fullscreen)){
		exit(EXIT_SUCCESS);
	}
	// Get the immediate DeviceContext
	device_->GetImmediateContext(&device_context_);
	// Get the pointer to the back buffer.
	result = swap_chain_->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&back_buffer_ptr);
	if(FAILED(result)){
		return false;
	}
	// Create the render target view with the back buffer pointer.
	result = device_->CreateRenderTargetView(back_buffer_ptr, NULL, &render_target_view_);
	if(FAILED(result)){
		return false;
	}
	// Release pointer to the back buffer as we no longer need it.
	back_buffer_ptr->Release();
	back_buffer_ptr = 0;
	if(!InitDepthBuffer(screen_width, screen_height)){
		exit(EXIT_SUCCESS);
	}
	if(!InitDepthStencil()){
		exit(EXIT_SUCCESS);
	}
	// Initialize the depth stencil view.
	ZeroMemory(&depth_stencil_view_desc, sizeof(depth_stencil_view_desc));
	// Set up the depth stencil view description.
	depth_stencil_view_desc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	depth_stencil_view_desc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	depth_stencil_view_desc.Texture2D.MipSlice = 0;
	// Create the depth stencil view.
	result = device_->CreateDepthStencilView(depth_stencil_buffer_, &depth_stencil_view_desc, &depth_stencil_view_);
	if(FAILED(result)){
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
bool CUDAD3D::findCUDADevice(){
	int num_graphics_GPU = 0;
	int device_count = 0;
	bool is_graphics_found = false;
	char first_graphics_name[NAME_LEN], dev_name[NAME_LEN];
	// This function call returns 0 if there are no CUDA capable devices.
	cudaError_t error_id = cudaGetDeviceCount(&device_count);
	if(error_id != cudaSuccess){
		//printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}
	if(device_count == 0){
		//printf("> There are no device(s) supporting CUDA\n");
		return false;
	}else{
		//printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
	}
	// Get CUDA device properties
	cudaDeviceProp deviceProp;
	for (int dev = 0; dev < device_count; ++dev){
		cudaGetDeviceProperties (&deviceProp, dev);
		STRCPY ( dev_name, NAME_LEN, deviceProp.name );
		//printf("> GPU %d: %s\n", dev, devname );
	}
	return true;
}
bool CUDAD3D::findDXDevice( char* dev_name ){
	HRESULT hr = S_OK;
	cudaError cuda_status;
	bool error;
	// Iterate through the candidate adapters
	IDXGIFactory *factory;
	// Create a DirectX graphics interface factory.
	hr = CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&factory);
	if(FAILED(hr)){
		return false;
	}
	UINT adapter = 0;
	for (; !cuda_capable_adapter_; ++adapter){
		// Get a candidate DXGI adapter
		IDXGIAdapter *pAdapter = NULL;
		hr = factory->EnumAdapters(adapter, &pAdapter);
		if (FAILED(hr)) { break; }		// no compatible adapters found
		// Query to see if there exists a corresponding compute device
		int cuDevice;
		cuda_status = cudaD3D11GetDevice(&cuDevice, pAdapter);
		if (cudaSuccess == cuda_status){
			// If so, mark it as the one against which to create our d3d10 device
			cuda_capable_adapter_ = pAdapter;
			cuda_capable_adapter_->AddRef();
		}
		pAdapter->Release();
	}
	factory->Release();
	if(!cuda_capable_adapter_){
		return false;
	}
	DXGI_ADAPTER_DESC adapter_desc;
	cuda_capable_adapter_->GetDesc (&adapter_desc);
	wcstombs (dev_name, adapter_desc.Description, 128);
	// Store the dedicated video card memory in megabytes.
	video_card_memory_ = (int)(adapter_desc.DedicatedVideoMemory / 1024 / 1024);
	unsigned int string_length;
	// Convert the name of the video card to a character array and store it.
	error = wcstombs_s(&string_length, video_card_description_, 128, adapter_desc.Description, 128);
	if(error){
		return false;
	}
	//printf("> Found 1 D3D11 Adapater(s) /w Compute capability.\n" );
	//printf("> %s\n", dev_name );
	return true;
}