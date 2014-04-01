#include "cuda_d3d.h"
// This header inclues all the necessary D3D11 and CUDA includes
#include <dynlink_d3d11.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>

// includes, project
#include <helper_functions.h>

#define NAME_LEN	512

CUDAD3D::CUDAD3D(void):D3DClass(), g_pCudaCapableAdapter(0){
	m_swapChain = 0;
	m_device = 0;
	m_deviceContext = 0;
	m_renderTargetView = 0;
	m_depthStencilBuffer = 0;
	m_depthStencilState = 0;
	m_depthStencilView = 0;
	m_rasterState = 0;
	m_depthDisabledStencilState = 0;
	m_alphaEnableBlendingState = 0;
	m_alphaDisableBlendingState = 0;
}
CUDAD3D::CUDAD3D(const D3DClass& other):D3DClass(other){
	m_swapChain = 0;
	m_device = 0;
	m_deviceContext = 0;
	m_renderTargetView = 0;
	m_depthStencilBuffer = 0;
	m_depthStencilState = 0;
	m_depthStencilView = 0;
	m_rasterState = 0;
	m_depthDisabledStencilState = 0;
	m_alphaEnableBlendingState = 0;
	m_alphaDisableBlendingState = 0;
}

CUDAD3D::~CUDAD3D(void){
}
bool CUDAD3D::InitDisplayMode(int screenWidth, int screenHeight, unsigned int &numerator, unsigned int &denominator){
	HRESULT result;
	IDXGIOutput* adapterOutput;
	unsigned int numModes;
	DXGI_MODE_DESC* displayModeList;
	// Enumerate the primary adapter output (monitor).
	result = g_pCudaCapableAdapter->EnumOutputs(0, &adapterOutput);
	if(FAILED(result)){
		return false;
	}
	// Get the number of modes that fit the DXGI_FORMAT_R8G8B8A8_UNORM display format for the adapter output (monitor).
	result = adapterOutput->GetDisplayModeList(DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_ENUM_MODES_INTERLACED, &numModes, NULL);
	if(FAILED(result)){
		return false;
	}

	// Create a list to hold all the possible display modes for this monitor/video card combination.
	displayModeList = new DXGI_MODE_DESC[numModes];
	if(!displayModeList){
		return false;
	}

	// Now fill the display mode list structures.
	result = adapterOutput->GetDisplayModeList(DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_ENUM_MODES_INTERLACED, &numModes, displayModeList);
	if(FAILED(result)){
		return false;
	}

	// Now go through all the display modes and find the one that matches the screen width and height.
	// When a match is found store the numerator and denominator of the refresh rate for that monitor.
	for(int i=0; i<(int)numModes; i++){
		if(displayModeList[i].Width == (unsigned int)screenWidth){
			if(displayModeList[i].Height == (unsigned int)screenHeight){
				numerator = displayModeList[i].RefreshRate.Numerator;
				denominator = displayModeList[i].RefreshRate.Denominator;
			}
		}
	}

	// Release the display mode list.
	delete [] displayModeList;
	displayModeList = 0;

	// Release the adapter output.
	adapterOutput->Release();
	adapterOutput = 0;

	return true;
}
bool CUDAD3D::InitSwapChain(HWND hwnd, int screenWidth, int screenHeight, unsigned int& numerator, unsigned int& denominator, bool fullscreen){
	DXGI_SWAP_CHAIN_DESC swapChainDesc;
	HRESULT result;
	// Initialize the swap chain description.
	ZeroMemory(&swapChainDesc, sizeof(swapChainDesc));

	// Set to a single back buffer.
	swapChainDesc.BufferCount = 1;

	// Set the width and height of the back buffer.
	swapChainDesc.BufferDesc.Width = screenWidth;
	swapChainDesc.BufferDesc.Height = screenHeight;

	// Set regular 32-bit surface for the back buffer.
	swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;

	// Set the refresh rate of the back buffer.
	if(m_vsync_enabled){
		swapChainDesc.BufferDesc.RefreshRate.Numerator = numerator;
		swapChainDesc.BufferDesc.RefreshRate.Denominator = denominator;
	}else{
		swapChainDesc.BufferDesc.RefreshRate.Numerator = 0;
		swapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
	}

	// Set the usage of the back buffer.
	swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;

	// Set the handle for the window to render to.
	swapChainDesc.OutputWindow = hwnd;

	// Turn multisampling off.
	swapChainDesc.SampleDesc.Count = 1;
	swapChainDesc.SampleDesc.Quality = 0;

	// Set to full screen or windowed mode.
	if(fullscreen){
		swapChainDesc.Windowed = false;
	}else{
		swapChainDesc.Windowed = true;
	}

	// Set the scan line ordering and scaling to unspecified.
	swapChainDesc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
	swapChainDesc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;

	// Discard the back buffer contents after presenting.
	swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

	// Don't set the advanced flags.
	swapChainDesc.Flags = 0;

	// Set the feature level to DirectX 11.
	D3D_FEATURE_LEVEL featureLevel[] =
	{
		D3D_FEATURE_LEVEL_11_0,
		D3D_FEATURE_LEVEL_10_1,
		D3D_FEATURE_LEVEL_10_0
	};
	if ( !g_pCudaCapableAdapter ){
		MessageBox(NULL,L"errorcuda",NULL,NULL);
		return false;
	}
	D3D_FEATURE_LEVEL flRes;
	result = D3D11CreateDeviceAndSwapChain(
		g_pCudaCapableAdapter,
		D3D_DRIVER_TYPE_UNKNOWN,
		NULL, 0, featureLevel,
		3, D3D11_SDK_VERSION,
		&swapChainDesc,
		&m_swapChain, &m_device,
		&flRes, &m_deviceContext);
	if(FAILED(result)){
		return false;
	}
	// Release the adapter.
	g_pCudaCapableAdapter->Release();
	g_pCudaCapableAdapter = 0;

	return true;
}
bool CUDAD3D::Initialize(int screenWidth, int screenHeight, bool vsync, HWND hwnd, bool fullscreen, 
						 float screenDepth, float screenNear){
	HRESULT result;
	ID3D11Texture2D* backBufferPtr;
	D3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc;
	unsigned int numerator, denominator;
	D3D11_VIEWPORT viewport;
	float fieldOfView, screenAspect;
	char device_name[256];
	
	// Store the vsync setting.
	m_vsync_enabled = vsync;
	if (!findCUDADevice() ){				// Search for CUDA GPU 
		exit(EXIT_SUCCESS);
	}
	if (!findDXDevice(device_name)){		// Search for D3D Hardware Device
		exit(EXIT_SUCCESS);
	}
	if(!InitDisplayMode(screenWidth, screenHeight, numerator, denominator)){
		exit(EXIT_SUCCESS);
	}
	if(!InitSwapChain(hwnd, screenWidth, screenHeight, numerator, denominator, fullscreen)){
		exit(EXIT_SUCCESS);
	}
	// Get the immediate DeviceContext
	m_device->GetImmediateContext(&m_deviceContext);
	// Get the pointer to the back buffer.
	result = m_swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&backBufferPtr);
	if(FAILED(result)){
		return false;
	}
	// Create the render target view with the back buffer pointer.
	result = m_device->CreateRenderTargetView(backBufferPtr, NULL, &m_renderTargetView);
	if(FAILED(result)){
		return false;
	}
	// Release pointer to the back buffer as we no longer need it.
	backBufferPtr->Release();
	backBufferPtr = 0;

	if(!InitDepthBuffer(screenWidth, screenHeight)){
		exit(EXIT_SUCCESS);
	}
	if(!InitDepthStencil()){
		exit(EXIT_SUCCESS);
	}

	// Initialize the depth stencil view.
	ZeroMemory(&depthStencilViewDesc, sizeof(depthStencilViewDesc));

	// Set up the depth stencil view description.
	depthStencilViewDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	depthStencilViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	depthStencilViewDesc.Texture2D.MipSlice = 0;

	// Create the depth stencil view.
	result = m_device->CreateDepthStencilView(m_depthStencilBuffer, &depthStencilViewDesc, &m_depthStencilView);
	if(FAILED(result)){
		return false;
	}

	// Bind the render target view and depth stencil buffer to the output render pipeline.
	m_deviceContext->OMSetRenderTargets(1, &m_renderTargetView, m_depthStencilView);

	CreateRaster();

	// Setup the viewport for rendering.
	viewport.Width = (float)screenWidth;
	viewport.Height = (float)screenHeight;
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;
	viewport.TopLeftX = 0.0f;
	viewport.TopLeftY = 0.0f;

	// Create the viewport.
	m_deviceContext->RSSetViewports(1, &viewport);

	// Setup the projection matrix.
	fieldOfView = (float)D3DX_PI / 4.0f;
	screenAspect = (float)screenWidth / (float)screenHeight;

	// Create the projection matrix for 3D rendering.
	D3DXMatrixPerspectiveFovLH(&m_projectionMatrix, fieldOfView, screenAspect, screenNear, screenDepth);

	// Initialize the world matrix to the identity matrix.
	D3DXMatrixIdentity(&m_worldMatrix);

	// Create an orthographic projection matrix for 2D rendering.
	D3DXMatrixOrthoLH(&m_orthoMatrix, (float)screenWidth, (float)screenHeight, screenNear, screenDepth);

	if(!InitDepthDisableStencil()){
		exit(EXIT_SUCCESS);
	}
	if(!InitBlendState()){
		exit(EXIT_SUCCESS);
	}
	return true;
}


bool CUDAD3D::findCUDADevice(){
	int nGraphicsGPU = 0;
	int deviceCount = 0;
	bool bFoundGraphics = false;
	char firstGraphicsName[NAME_LEN], devname[NAME_LEN];

	// This function call returns 0 if there are no CUDA capable devices.
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if(error_id != cudaSuccess){
		//printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}

	if(deviceCount == 0){
		//printf("> There are no device(s) supporting CUDA\n");
		return false;
	}else{
		//printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
	}

	// Get CUDA device properties
	cudaDeviceProp deviceProp;
	for (int dev = 0; dev < deviceCount; ++dev){
		cudaGetDeviceProperties (&deviceProp, dev);
		STRCPY ( devname, NAME_LEN, deviceProp.name );
		//printf("> GPU %d: %s\n", dev, devname );
	}
	return true;
}

bool CUDAD3D::findDXDevice( char* dev_name ){
	HRESULT hr = S_OK;
	cudaError cuStatus;
	bool error;
	// Iterate through the candidate adapters
	IDXGIFactory *pFactory;
	// Create a DirectX graphics interface factory.
	hr = CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&pFactory);
	if(FAILED(hr)){
		return false;
	}

	UINT adapter = 0;
	for (; !g_pCudaCapableAdapter; ++adapter){
		// Get a candidate DXGI adapter
		IDXGIAdapter *pAdapter = NULL;
		hr = pFactory->EnumAdapters(adapter, &pAdapter);
		if (FAILED(hr)) { break; }		// no compatible adapters found
		// Query to see if there exists a corresponding compute device
		int cuDevice;
		cuStatus = cudaD3D11GetDevice(&cuDevice, pAdapter);
		if (cudaSuccess == cuStatus){
			// If so, mark it as the one against which to create our d3d10 device
			g_pCudaCapableAdapter = pAdapter;
			g_pCudaCapableAdapter->AddRef();
		}
		pAdapter->Release();
	}
	pFactory->Release();

	if(!g_pCudaCapableAdapter){
		return false;
	}

	DXGI_ADAPTER_DESC adapterDesc;
	g_pCudaCapableAdapter->GetDesc (&adapterDesc);
	wcstombs (dev_name, adapterDesc.Description, 128);

	// Store the dedicated video card memory in megabytes.
	m_videoCardMemory = (int)(adapterDesc.DedicatedVideoMemory / 1024 / 1024);
	unsigned int stringLength;
	// Convert the name of the video card to a character array and store it.
	error = wcstombs_s(&stringLength, m_videoCardDescription, 128, adapterDesc.Description, 128);
	if(error){
		return false;
	}
	//printf("> Found 1 D3D11 Adapater(s) /w Compute capability.\n" );
	//printf("> %s\n", dev_name );
	return true;
}