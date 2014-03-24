////////////////////////////////////////////////////////////////////////////////
// Filename: applicationclass.cpp
////////////////////////////////////////////////////////////////////////////////
#include "applicationclass.h"

#define NAME_LEN	512
ApplicationClass::ApplicationClass(): m_Input(0), m_Direct3D(0), m_Camera(0), m_Terrain(0),
	m_Timer(0), m_Position(0), m_Fps(0), m_Cpu(0), m_FontShader(0), m_Text(0),
	m_TerrainShader(0), m_Light(0), m_TextureShader(0), m_TextureToTextureShader(0),
	m_RenderFullSizeTexture(0), m_DownSampleHalfSizeTexure(0), m_FullSizeTexure(0), m_FullScreenWindow(0),
	m_HalfSizeTexture(0), mMergerShader(0),	m_MergeFullSizeTexture(0)
{
	g_pInputLayout = NULL;
	//g_pConstantBuffer = NULL;
}

ApplicationClass::ApplicationClass(const ApplicationClass& other): m_Input(0), m_Direct3D(0), m_Camera(0), m_Terrain(0),
	m_Timer(0), m_Position(0), m_Fps(0), m_Cpu(0), m_FontShader(0), m_Text(0),
	m_TerrainShader(0), m_Light(0), m_TextureShader(0), m_TextureToTextureShader(0),
	m_RenderFullSizeTexture(0), m_DownSampleHalfSizeTexure(0), m_FullSizeTexure(0), m_FullScreenWindow(0),
	m_HalfSizeTexture(0), mMergerShader(0), m_MergeFullSizeTexture(0)
{
	g_bDone   = false;
	g_bPassed = true;

	pArgc = NULL;
	pArgv = NULL;
	g_WindowWidth = 720;
	g_WindowHeight = 720;
}
ApplicationClass::~ApplicationClass(){
}
bool ApplicationClass::Initialize(HINSTANCE hinstance, HWND hwnd, int screenWidth, int screenHeight){
	bool result;
	int downSampleWidth, downSampleHeight;
    char *ref_file = NULL;

	// Set the size to sample down to.
	downSampleWidth = screenWidth / 2;
	downSampleHeight = screenHeight / 2;
	
	// Create the input object.  The input object will be used to handle reading the keyboard and mouse input from the user.
	m_Input = new InputClass;
	if(!m_Input){
		return false;
	}
	// Initialize the input object.
	result = m_Input->Initialize(hinstance, hwnd, screenWidth, screenHeight);
	if(!result){
		MessageBox(hwnd, L"Could not initialize the input object.", L"Error", MB_OK);
		return false;
	}
	// Create the Direct3D object.
	//m_Direct3D = new D3DClass;
	m_Direct3D = new CUDAD3D;
	if(!m_Direct3D){
		return false;
	}
	// Initialize the Direct3D object.
	result = m_Direct3D->Initialize(screenWidth, screenHeight, VSYNC_ENABLED, hwnd, FULL_SCREEN, SCREEN_DEPTH, SCREEN_NEAR);
	if(!result){
		MessageBox(hwnd, L"Could not initialize DirectX 11.", L"Error", MB_OK);
		return false;
	}
	//Initialize the camera and position objects
	InitCamera();
	//Initialize the terrain, water, and cubes
	InitObjects(hwnd);
	//Initialize the text on screen. 
	InitText(hwnd, screenWidth, screenHeight);
	// Create the light object.
	m_Light = new LightClass;
	if(!m_Light){
		return false;
	}
	// Initialize the light object.
	m_Light->SetAmbientColor(0.05f, 0.05f, 0.05f, 1.0f);
	m_Light->SetDiffuseColor(1.0f, 1.0f, 1.0f, 1.0f);
	m_Light->SetDirection(1.0f,0.0f, 0.0f);
	//Initialize the texture to render to
	InitTextures(hwnd, screenWidth, screenHeight);
	//Initialize the sahders to be used.
	InitShaders(hwnd);
	// Create the full screen ortho window object.
	m_FullScreenWindow = new OrthoWindowClass;
	if(!m_FullScreenWindow){
		return false;
	}
	// Initialize the full screen ortho window object.
	result = m_FullScreenWindow->Initialize(m_Direct3D->GetDevice(), screenWidth, screenHeight);
	if(!result)
	{
		MessageBox(hwnd, L"Could not initialize the full screen ortho window object.", L"Error", MB_OK);
		return false;
	}
	//Generate a random height map for the terrain
	m_Terrain->GenerateHeightMap(m_Direct3D->GetDevice());
	bool isShader = mSimpleShader->Initialize(m_Direct3D->GetDevice(),hwnd);
	// Initialize Direct3D
    if (isShader && SUCCEEDED(InitTextures()))
    {
        // 2D
        // register the Direct3D resources that we'll use
        // we'll read to and write from g_texture_2d, so don't set any special map flags for it
        cudaGraphicsD3D11RegisterResource(&g_texture_2d.cudaResource, g_texture_2d.pTexture, cudaGraphicsRegisterFlagsNone);
        getLastCudaError("cudaGraphicsD3D11RegisterResource (g_texture_2d) failed");
        // cuda cannot write into the texture directly : the texture is seen as a cudaArray and can only be mapped as a texture
        // Create a buffer so that cuda can write into it
        // pixel fmt is DXGI_FORMAT_R32G32B32A32_FLOAT
        cudaMallocPitch(&g_texture_2d.cudaLinearMemory, &g_texture_2d.pitch, g_texture_2d.width * sizeof(float) * 4, g_texture_2d.height);
        getLastCudaError("cudaMallocPitch (g_texture_2d) failed");
        cudaMemset(g_texture_2d.cudaLinearMemory, 1, g_texture_2d.pitch * g_texture_2d.height);

		InitClouds();
    }


	return true;
}
void ApplicationClass::InitClouds(){
	// 3D
    cudaGraphicsD3D11RegisterResource(&g_texture_cloud.cudaVelocityResource, g_texture_cloud.pTexture, cudaGraphicsRegisterFlagsNone);
    getLastCudaError("cudaGraphicsD3D11RegisterResource (g_texture_cloud) failed");
    // create the buffer. pixel fmt is DXGI_FORMAT_R8G8B8A8_SNORM
    //cudaMallocPitch(&g_texture_3d.cudaLinearMemory, &g_texture_3d.pitch, g_texture_3d.width * 4, g_texture_3d.height * g_texture_3d.depth);
    cudaMalloc(&g_texture_cloud.cudaVelocityLinearMemory, g_texture_cloud.width * 4 * g_texture_cloud.height * g_texture_cloud.depth);
    g_texture_cloud.pitch = g_texture_cloud.width * 4;
    getLastCudaError("cudaMallocPitch (g_texture_cloud) failed");
    cudaMemset(g_texture_cloud.cudaVelocityLinearMemory, 1, g_texture_cloud.pitch * g_texture_cloud.height * g_texture_cloud.depth);
    getLastCudaError("cudaMemset (g_texture_cloud) failed");

	cudaMalloc(&g_texture_cloud.cudaAdvectLinearMemory, g_texture_cloud.width * 4 * g_texture_cloud.height * g_texture_cloud.depth);
    g_texture_cloud.pitch = g_texture_cloud.width * 4;
    getLastCudaError("cudaMallocPitch (g_texture_cloud) failed");
    cudaMemset(g_texture_cloud.cudaAdvectLinearMemory, 1, g_texture_cloud.pitch * g_texture_cloud.height * g_texture_cloud.depth);
    getLastCudaError("cudaMemset (g_texture_cloud) failed");

	cudaMalloc(&g_texture_cloud.cudaDivergenceLinearMemory, g_texture_cloud.width * 4 * g_texture_cloud.height * g_texture_cloud.depth);
    g_texture_cloud.pitch = g_texture_cloud.width * 4;
    getLastCudaError("cudaMallocPitch (g_texture_cloud) failed");
    cudaMemset(g_texture_cloud.cudaDivergenceLinearMemory, 1, g_texture_cloud.pitch * g_texture_cloud.height * g_texture_cloud.depth);
    getLastCudaError("cudaMemset (g_texture_cloud) failed");

	cudaMalloc(&g_texture_cloud.cudaPressureLinearMemory, g_texture_cloud.width * 4 * g_texture_cloud.height * g_texture_cloud.depth);
    g_texture_cloud.pitch = g_texture_cloud.width * 4;
    getLastCudaError("cudaMallocPitch (g_texture_cloud) failed");
    cudaMemset(g_texture_cloud.cudaPressureLinearMemory, 1, g_texture_cloud.pitch * g_texture_cloud.height * g_texture_cloud.depth);
    getLastCudaError("cudaMemset (g_texture_cloud) failed");
}
void ApplicationClass::Shutdown(){
	if(m_FullScreenWindow){
		m_FullScreenWindow->Shutdown();
		delete m_FullScreenWindow;
		m_FullScreenWindow = 0;
	}
	// Release the light object.
	if(m_Light){
		delete m_Light;
		m_Light = 0;
	}
	//Shutdown functions
	ShutdownText();
	ShutdownTextures();
	ShutdownShaders();
	ShutdownObjects();
	ShutdownCamera();
	// Release the Direct3D object.
	if(m_Direct3D){
		m_Direct3D->Shutdown();
		delete m_Direct3D;
		m_Direct3D = 0;
	}
	// Release the input object.
	if(m_Input){
		m_Input->Shutdown();
		delete m_Input;
		m_Input = 0;
	}
	return;
}
bool ApplicationClass::Frame(){
	bool result;
	if(m_Input){
		// Read the user input.
		result = m_Input->Frame();
		if(!result){
			return false;
		}
	
		// Check if the user pressed escape and wants to exit the application.
		if(m_Input->IsEscapePressed() == true){
			return false;
		}
	}
	// Update the system stats.
	m_Timer->Frame();
	m_Fps->Frame();
	m_Cpu->Frame();

	// Update the FPS value in the text object.
	result = m_Text->SetFps(m_Fps->GetFps(), m_Direct3D->GetDeviceContext());
	if(!result){
		return false;
	}
	
	// Update the CPU usage value in the text object.
	result = m_Text->SetCpu(m_Cpu->GetCpuPercentage(), m_Direct3D->GetDeviceContext());
	if(!result){
		return false;
	}
	if(m_Input){
		// Do the frame input processing.
		result = HandleInput(m_Timer->GetTime());
		if(!result){
			return false;
		}
	}
	// Render the graphics scene.
	result = Render();
	if(!result){
		return false;
	}
	return result;
}
bool ApplicationClass::HandleInput(float frameTime){
	bool keyDown, result;
	float posX, posY, posZ, rotX, rotY, rotZ;

	// Set the frame time for calculating the updated position.
	m_Position->SetFrameTime(frameTime);
	//if left or a was pressed turn left
	keyDown = (m_Input->IsLeftPressed() || m_Input->IsAPressed());
	m_Position->TurnLeft(keyDown);
	//if right or d was pressed turn right
	keyDown = (m_Input->IsRightPressed() || m_Input->IsDPressed());
	m_Position->TurnRight(keyDown);
	//if up or w was pressed turn up
	keyDown = (m_Input->IsUpPressed() || m_Input->IsWPressed());
	m_Position->MoveForward(keyDown);
	//if down or s was pressed turn down
	keyDown = (m_Input->IsDownPressed() || m_Input->IsSPressed());
	m_Position->MoveBackward(keyDown);
	//if q was pressed move up
	keyDown = m_Input->IsQPressed();
	m_Position->MoveUpward(keyDown);
	//if e was pressed move down
	keyDown = m_Input->IsEPressed();
	m_Position->MoveDownward(keyDown);
	//if page up or z was pressed look up
	keyDown = (m_Input->IsPgUpPressed() || m_Input->IsZPressed());
	m_Position->LookUpward(keyDown);
	//if page down or x was pressed look down
	keyDown = (m_Input->IsPgDownPressed() || m_Input->IsXPressed());
	m_Position->LookDownward(keyDown);
	
	// Get the view point position/rotation.
	m_Position->GetPosition(posX, posY, posZ);
	m_Position->GetRotation(rotX, rotY, rotZ);

	// Set the position of the camera.
	m_Camera->SetPosition(posX, posY, posZ);
	m_Camera->SetRotation(rotX, rotY, rotZ);

	// Update the position values in the text object.
	result = m_Text->SetCameraPosition(posX, posY, posZ, m_Direct3D->GetDeviceContext());
	if(!result){
		return false;
	}
	// Update the rotation values in the text object.
	result = m_Text->SetCameraRotation(rotX, rotY, rotZ, m_Direct3D->GetDeviceContext());
	if(!result){
		return false;
	}
	return true;
}
bool ApplicationClass::Render(){
	bool result;
	CudaRender();
	// First render the scene to a render texture.
	result = RenderSceneToTexture(m_RenderFullSizeTexture);
	if(!result){
		return false;
	}
	//render the texture to the scene
	result = Render2DTextureScene(m_RenderFullSizeTexture);

	if(!result){
		return false;
	}
	return true;
}
/*
*	Renders the objects on screen to a texture ready for post-processing.
*/
bool ApplicationClass::RenderSceneToTexture(RenderTextureClass* write){
	D3DXMATRIX worldMatrix, modelWorldMatrix, viewMatrix, projectionMatrix;
	bool result;
		
	// Set the render target to be the render to texture.
	write->SetRenderTarget(m_Direct3D->GetDeviceContext());
	// Clear the render to texture.
	write->ClearRenderTarget(m_Direct3D->GetDeviceContext(), 0.0f, 0.0f, 0.0f, 1.0f);
	// Generate the view matrix based on the camera's position.
	m_Camera->Render();
	// Get the world, view, and projection matrices from the camera and d3d objects.
	m_Direct3D->GetWorldMatrix(worldMatrix);
	m_Camera->GetViewMatrix(viewMatrix);
	m_Direct3D->GetProjectionMatrix(projectionMatrix);	

	// Render the terrain buffers.
	m_Terrain->Render(m_Direct3D->GetDeviceContext());
	// Render the terrain using the terrain shader.
	result = m_TerrainShader->Render(m_Direct3D->GetDeviceContext(), m_Terrain->GetIndexCount(), worldMatrix, viewMatrix, projectionMatrix, 
		m_Light->GetAmbientColor(), m_Light->GetDiffuseColor(), m_Light->GetDirection(), m_Terrain->GetTexture());
	if(!result){
		return false;
	}
	// Turn on the alpha blending before rendering the text.
	m_Direct3D->TurnOnAlphaBlending();

	// Turn off alpha blending after rendering the text.
	m_Direct3D->TurnOffAlphaBlending();
	// Reset the render target back to the original back buffer and not the render to texture anymore.
	m_Direct3D->SetBackBufferRenderTarget();

	// Reset the viewport back to the original.
	m_Direct3D->ResetViewport();

	return true;
}
/*
*	Takes a texture and performs shader actions on it.
*/
bool ApplicationClass::RenderTexture(ShaderClass *shader, RenderTextureClass *readTexture, RenderTextureClass *writeTexture, OrthoWindowClass *window){
	D3DXMATRIX orthoMatrix;
	float screenSizeX, screenSizeY;
	bool result;

	screenSizeY = (float)writeTexture->GetTextureHeight();
	screenSizeX = (float)writeTexture->GetTextureWidth();

	// Set the render target to be the render to texture.
	writeTexture->SetRenderTarget(m_Direct3D->GetDeviceContext());

	// Clear the render to texture.
	writeTexture->ClearRenderTarget(m_Direct3D->GetDeviceContext(), 0.0f, 0.0f, 0.0f, 1.0f);

	// Generate the view matrix based on the camera's position.
	m_Camera->Render();
	
	// Get the ortho matrix from the render to texture since texture has different dimensions being that it is smaller.
	writeTexture->GetOrthoMatrix(orthoMatrix);

	// Turn off the Z buffer to begin all 2D rendering.
	m_Direct3D->TurnZBufferOff();

	// Put the small ortho window vertex and index buffers on the graphics pipeline to prepare them for drawing.
	window->Render(m_Direct3D->GetDeviceContext());

	// Render the small ortho window using the texture shader and the render to texture of the scene as the texture resource.
	result = shader->Render(m_Direct3D->GetDeviceContext(), window->GetIndexCount(), orthoMatrix, 
		readTexture->GetShaderResourceView(),screenSizeY,screenSizeX);
	if(!result){
		return false;
	}

	// Turn the Z buffer back on now that all 2D rendering has completed.
	m_Direct3D->TurnZBufferOn();

	// Reset the render target back to the original back buffer and not the render to texture anymore.
	m_Direct3D->SetBackBufferRenderTarget();

	// Reset the viewport back to the original.
	m_Direct3D->ResetViewport();
	return true;
}
/*
*	Takes two textures and combines them to make a third texture;
*/
bool ApplicationClass::RenderMergeTexture(RenderTextureClass *readTexture, RenderTextureClass *readTexture2, RenderTextureClass *writeTexture, OrthoWindowClass *window){
	bool result;

	// Set the render target to be the render to texture.
	writeTexture->SetRenderTarget(m_Direct3D->GetDeviceContext());

	// Clear the render to texture.
	writeTexture->ClearRenderTarget(m_Direct3D->GetDeviceContext(), 0.0f, 0.0f, 0.0f, 1.0f);

	// Generate the view matrix based on the camera's position.
	m_Camera->Render();

	// Turn off the Z buffer to begin all 2D rendering.
	m_Direct3D->TurnZBufferOff();

	// Put the small ortho window vertex and index buffers on the graphics pipeline to prepare them for drawing.
	window->Render(m_Direct3D->GetDeviceContext());

	// Render the small ortho window using the texture shader and the render to texture of the scene as the texture resource.
	result = mMergerShader->Render(m_Direct3D->GetDeviceContext(), window->GetIndexCount(), readTexture->GetShaderResourceView(), 
		readTexture2->GetShaderResourceView());
	if(!result){
		return false;
	}

	// Turn the Z buffer back on now that all 2D rendering has completed.
	m_Direct3D->TurnZBufferOn();

	// Reset the render target back to the original back buffer and not the render to texture anymore.
	m_Direct3D->SetBackBufferRenderTarget();

	// Reset the viewport back to the original.
	m_Direct3D->ResetViewport();
	return true;
}
/*
* Renders a texture to the screen.
*/
bool ApplicationClass::Render2DTextureScene(RenderTextureClass* mRead){
	D3DXMATRIX worldMatrix, viewMatrix, orthoMatrix, projectionMatrix;
	bool result;

	// Clear the buffers to begin the scene.
	m_Direct3D->BeginScene(0.0f, 0.0f, 0.0f, 1.0f);

	// Generate the view matrix based on the camera's position.
	m_Camera->Render();

	// Get the world, view, and ortho matrices from the camera and d3d objects.
	m_Camera->GetViewMatrix(viewMatrix);
	m_Direct3D->GetWorldMatrix(worldMatrix);
	m_Direct3D->GetOrthoMatrix(orthoMatrix);
	// Turn off the Z buffer to begin all 2D rendering.
	m_Direct3D->TurnZBufferOff();

	// Put the full screen ortho window vertex and index buffers on the graphics pipeline to prepare them for drawing.
	m_FullScreenWindow->Render(m_Direct3D->GetDeviceContext());

	//Render the full screen ortho window using the texture shader and the full screen sized blurred render to texture resource.
	result = m_TextureToTextureShader->Render(m_Direct3D->GetDeviceContext(), m_FullScreenWindow->GetIndexCount(), orthoMatrix, mRead->GetShaderResourceView());
	///result = m_TextureToTextureShader->Render(m_Direct3D->GetDeviceContext(), m_FullScreenWindow->GetIndexCount(), orthoMatrix, g_texture_2d.pSRView);
	if(!result){
		return false;
	}
	
	// Turn on the alpha blending before rendering the text.
	m_Direct3D->TurnOnAlphaBlending();

	// Render the text user interface elements.
	result = m_Text->Render(m_Direct3D->GetDeviceContext(), m_FontShader, worldMatrix, orthoMatrix);
	if(!result){
		return false;
	}

	// Turn off alpha blending after rendering the text.
	m_Direct3D->TurnOffAlphaBlending();
	
	// Turn the Z buffer back on now that all 2D rendering has completed.
	m_Direct3D->TurnZBufferOn();

	// Present the rendered scene to the screen.
	m_Direct3D->EndScene();

	return true;
}
bool ApplicationClass::InitText(HWND hwnd, int screenWidth , int screenHeight){
	D3DXMATRIX baseViewMatrix;
	char videoCard[128];
	int videoMemory;
	bool result = true;
	// Create the timer object.
	m_Timer = new TimerClass;
	if(!m_Timer){
		return false;
	}
	// Initialize the timer object.
	result = m_Timer->Initialize();
	if(!result){
		MessageBox(hwnd, L"Could not initialize the timer object.", L"Error", MB_OK);
		return false;
	}
	// Create the fps object.
	m_Fps = new FpsClass;
	if(!m_Fps){
		return false;
	}

	// Initialize the fps object.
	m_Fps->Initialize();
	// Create the cpu object.
	m_Cpu = new CpuClass;
	if(!m_Cpu){
		return false;
	}
	// Initialize the cpu object.
	m_Cpu->Initialize();
	// Create the text object.
	m_Text = new TextClass;
	if(!m_Text){
		return false;
	}
	m_Camera->GetViewMatrix(baseViewMatrix);
	// Initialize the text object.
	result = m_Text->Initialize(m_Direct3D->GetDevice(), m_Direct3D->GetDeviceContext(), hwnd, screenWidth, screenHeight, baseViewMatrix);
	if(!result){
		MessageBox(hwnd, L"Could not initialize the text object.", L"Error", MB_OK);
		return false;
	}
	// Retrieve the video card information.
	m_Direct3D->GetVideoCardInfo(videoCard, videoMemory);
	// Set the video card information in the text object.
	result = m_Text->SetVideoCardInfo(videoCard, videoMemory, m_Direct3D->GetDeviceContext());
	if(!result){
		MessageBox(hwnd, L"Could not set video card info in the text object.", L"Error", MB_OK);
		return false;
	}
	return true;
}
bool ApplicationClass::InitObjects(HWND hwnd){
	bool result;

	// Create the terrain object.
	m_Terrain = new TerrainClass;
	if(!m_Terrain){
		return false;
	}
	// Initialize the terrain object.
	result = m_Terrain->InitializeTerrain(m_Direct3D->GetDevice(), 129, 129, L"data/ground.dds");   //initialize the flat terrain.
	if(!result){
		MessageBox(hwnd, L"Could not initialize the terrain object.", L"Error", MB_OK);
		return false;
	}
	
	return true;
}
bool ApplicationClass::InitTextures(HWND hwnd, int screenWidth, int screenHeight){
	bool result;
	int	downSampleWidth = screenWidth / 2;
	int downSampleHeight = screenHeight / 2;
	// Create the render to texture object.
	m_RenderFullSizeTexture = new RenderTextureClass;
	if(!m_RenderFullSizeTexture){
		return false;
	}
	// Initialize the render to texture object.
	result = m_RenderFullSizeTexture->Initialize(m_Direct3D->GetDevice(), screenWidth, screenHeight, SCREEN_DEPTH, SCREEN_NEAR);
	if(!result){
		MessageBox(hwnd, L"Could not initialize the render to texture object.", L"Error", MB_OK);
		return false;
	}
	// Create the up sample render to texture object.
	m_FullSizeTexure = new RenderTextureClass;
	if(!m_FullSizeTexure){
		return false;
	}
	// Initialize the up sample render to texture object.
	result = m_FullSizeTexure->Initialize(m_Direct3D->GetDevice(), screenWidth, screenHeight, SCREEN_DEPTH, SCREEN_NEAR);
	if(!result){
		MessageBox(hwnd, L"Could not initialize the full size render to texture object.", L"Error", MB_OK);
		return false;
	}
	//create a third full size texture for merging
	m_MergeFullSizeTexture = new RenderTextureClass;
	if (!m_MergeFullSizeTexture){
		return false;
	}
	result = m_MergeFullSizeTexture->Initialize(m_Direct3D->GetDevice(), screenWidth, screenHeight, SCREEN_DEPTH, SCREEN_NEAR);
	if (!result){
		MessageBox(hwnd, L"Could not initialize the merge full size to texture object.", L"Error", MB_OK);		
		return false;
	}
	// Create the down sample render to texture object.
	m_DownSampleHalfSizeTexure = new RenderTextureClass;
	if(!m_DownSampleHalfSizeTexure){
		return false;
	}
	// Initialize the down sample render to texture object.
	result = m_DownSampleHalfSizeTexure->Initialize(m_Direct3D->GetDevice(), downSampleWidth, downSampleHeight, SCREEN_DEPTH, SCREEN_NEAR);
	if(!result){
		MessageBox(hwnd, L"Could not initialize the down sample render to texture object.", L"Error", MB_OK);
		return false;
	}
	//create a second down sample texture for performing convolutions on
	m_HalfSizeTexture = new RenderTextureClass;
	if (!m_HalfSizeTexture){
		return false;
	}
	result = m_HalfSizeTexture->Initialize(m_Direct3D->GetDevice(), downSampleWidth, downSampleHeight, SCREEN_DEPTH, SCREEN_NEAR);
	if (!result){
		MessageBox(hwnd, L"Could not initialize the second half size to texture object.", L"Error", MB_OK);		
		return false;
	}
	return true;
}
bool ApplicationClass::InitCamera(){
	float cameraX, cameraY, cameraZ;
	// Create the camera object.
	m_Camera = new CameraClass;
	if(!m_Camera){
		return false;
	}
	// Initialize a base view matrix with the camera for 2D user interface rendering.
	m_Camera->SetPosition(0.0f, 0.0f, -1.0f);
	m_Camera->Render();
	// Set the initial position of the camera.
	cameraX = 0.0f;
	cameraY = 2.0f;
	cameraZ = -7.0f;
	m_Camera->SetPosition(cameraX, cameraY, cameraZ);
	// Create the position object.
	m_Position = new PositionClass;
	if(!m_Position){
		return false;
	}
	// Set the initial position of the viewer to the same as the initial camera position.
	m_Position->SetPosition(cameraX, cameraY, cameraZ);
	return true;
}
bool ApplicationClass::InitShaders(HWND hwnd){
	bool result;
	// Create the font shader object.
	m_FontShader = new FontShaderClass;
	if(!m_FontShader){
		return false;
	}
	// Initialize the font shader object.
	result = m_FontShader->Initialize(m_Direct3D->GetDevice(), hwnd);
	if(!result){
		MessageBox(hwnd, L"Could not initialize the font shader object.", L"Error", MB_OK);
		return false;
	}
	// Create the terrain shader object.
	m_TerrainShader = new TerrainShaderClass;
	if(!m_TerrainShader){
		return false;
	}
	// Initialize the terrain shader object.
	result = m_TerrainShader->Initialize(m_Direct3D->GetDevice(), hwnd);
	if(!result){
		MessageBox(hwnd, L"Could not initialize the terrain shader object.", L"Error", MB_OK);
		return false;
	}
	
	// Create the texture to texture shader object.
	m_TextureToTextureShader = new TextureToTextureShaderClass;
	if(!m_TextureToTextureShader){
		return false;
	}
	// Initialize the texture to texture shader object.
	result = m_TextureToTextureShader->Initialize(m_Direct3D->GetDevice(), hwnd);
	if(!result){
		MessageBox(hwnd, L"Could not initialize the texture shader object.", L"Error", MB_OK);
		return false;
	}
	//create the texture shader object
	m_TextureShader = new TextureShaderClass;
	if(!m_TextureShader){
		return false;
	}
	// Initialize the texture shader object.
	result = m_TextureShader->Initialize(m_Direct3D->GetDevice(), hwnd);
	if(!result){
		MessageBox(hwnd, L"Could not initialize the texture shader object.", L"Error", MB_OK);
		return false;
	}
	//create the merge shader object
	mMergerShader = new MergeTextureShaderClass;
	if (!mMergerShader){
		return false;
	}
	result= mMergerShader->Initialize(m_Direct3D->GetDevice(),hwnd);
	if (!result){
		MessageBox(hwnd, L"Could not initialize the convolution shader object.", L"Error", MB_OK);
		return false;
	}
	mSimpleShader = new SimpleShader;
	if (!mSimpleShader){
		return false;
	}
	return true;
}
void ApplicationClass::ShutdownText(){
	// Release the text object.
	if(m_Text){
		m_Text->Shutdown();
		delete m_Text;
		m_Text = 0;
	}
	// Release the cpu object.
	if(m_Cpu){
		m_Cpu->Shutdown();
		delete m_Cpu;
		m_Cpu = 0;
	}
	// Release the fps object.
	if(m_Fps){
		delete m_Fps;
		m_Fps = 0;
	}
	// Release the timer object.
	if(m_Timer){
		delete m_Timer;
		m_Timer = 0;
	}
}
void ApplicationClass::ShutdownObjects(){
	// Release the terrain object.
	if(m_Terrain){
		m_Terrain->Shutdown();
		delete m_Terrain;
		m_Terrain = 0;
	}
}
void ApplicationClass::ShutdownTextures(){
	// Release the up sample render to texture object.
	if(m_FullSizeTexure){
		m_FullSizeTexure->Shutdown();
		delete m_FullSizeTexure;
		m_FullSizeTexure = 0;
	}
	// Release the down sample render to texture object.
	if(m_DownSampleHalfSizeTexure){
		m_DownSampleHalfSizeTexure->Shutdown();
		delete m_DownSampleHalfSizeTexure;
		m_DownSampleHalfSizeTexure = 0;
	}
	// Release the render to texture object.
	if(m_RenderFullSizeTexture){
		m_RenderFullSizeTexture->Shutdown();
		delete m_RenderFullSizeTexture;
		m_RenderFullSizeTexture = 0;
	}
	//release half size texture
	if (m_HalfSizeTexture){
		m_HalfSizeTexture->Shutdown();
		delete m_HalfSizeTexture;
		m_HalfSizeTexture = 0;
	}
	//release merge full size texture
	if (m_MergeFullSizeTexture){
		m_MergeFullSizeTexture->Shutdown();
		delete m_MergeFullSizeTexture;
		m_MergeFullSizeTexture = 0;
	}
}
void ApplicationClass::ShutdownCamera(){
	//relase position object
	if(m_Position){
		delete m_Position;
		m_Position = 0;
	}
	// Release the camera object.
	if(m_Camera){
		delete m_Camera;
		m_Camera = 0;
	}
}
void ApplicationClass::ShutdownShaders(){
	// Release the texture shader object.
	if(m_TextureShader){
		m_TextureShader->Shutdown();
		delete m_TextureShader;
		m_TextureShader = 0;
	}
	//release texture to texture shader
	if(m_TextureToTextureShader){
		m_TextureToTextureShader->Shutdown();
		delete m_TextureToTextureShader;
		m_TextureToTextureShader = 0;
	}
	// Release the font shader object.
	if(m_FontShader){
		m_FontShader->Shutdown();
		delete m_FontShader;
		m_FontShader = 0;
	}

	// Release the terrain shader object.
	if(m_TerrainShader){
		m_TerrainShader->Shutdown();
		delete m_TerrainShader;
		m_TerrainShader = 0;
	}
	//Release merge shader
	if (mMergerShader){
		mMergerShader->Shutdown();
		delete mMergerShader;
		mMergerShader = 0;
	}
}


//-----------------------------------------------------------------------------
// Name: InitTextures()
// Desc: Initializes Direct3D Textures (allocation and initialization)
//-----------------------------------------------------------------------------
HRESULT ApplicationClass::InitTextures(){
	int offsetInShader = 0;
	ID3D11Device* g_pd3dDevice = m_Direct3D->GetDevice();
	ID3D11DeviceContext* g_pd3dDeviceContext = m_Direct3D->GetDeviceContext();
    //
    // create the D3D resources we'll be using
    //
    // 2D texture
    {
        g_texture_2d.width  = 256;
        g_texture_2d.height = 256;

        D3D11_TEXTURE2D_DESC desc;
        ZeroMemory(&desc, sizeof(D3D11_TEXTURE2D_DESC));
        desc.Width = g_texture_2d.width;
        desc.Height = g_texture_2d.height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		

        if (FAILED(g_pd3dDevice->CreateTexture2D(&desc, NULL, &g_texture_2d.pTexture))){
            return E_FAIL;
        }

        if (FAILED(g_pd3dDevice->CreateShaderResourceView(g_texture_2d.pTexture, NULL, &g_texture_2d.pSRView))){
            return E_FAIL;
        }

        offsetInShader = 0; // to be clean we should look for the offset from the shader code
        g_pd3dDeviceContext->PSSetShaderResources(offsetInShader, 1, &g_texture_2d.pSRView);
    }
	
	// CLoud texture
    {
        g_texture_cloud.width  = 64;
        g_texture_cloud.height = 64;
        g_texture_cloud.depth  = 64;

        D3D11_TEXTURE3D_DESC desc;
        ZeroMemory(&desc, sizeof(D3D11_TEXTURE3D_DESC));
        desc.Width = g_texture_cloud.width;
        desc.Height = g_texture_cloud.height;
        desc.Depth = g_texture_cloud.depth;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R8G8B8A8_SNORM;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

        if (FAILED(g_pd3dDevice->CreateTexture3D(&desc, NULL, &g_texture_cloud.pTexture))){
            return E_FAIL;
        }

        if (FAILED(g_pd3dDevice->CreateShaderResourceView(g_texture_cloud.pTexture, NULL, &g_texture_cloud.pSRView))){
            return E_FAIL;
        }

        offsetInShader = 1; // to be clean we should look for the offset from the shader code
        g_pd3dDeviceContext->PSSetShaderResources(offsetInShader, 1, &g_texture_cloud.pSRView);
    }

    return S_OK;
}
//-----------------------------------------------------------------------------
// Name: Render()
// Desc: Launches the CUDA kernels to fill in the texture data
//-----------------------------------------------------------------------------
void ApplicationClass::CudaRender(){
    //
    // map the resources we've registered so we can access them in Cuda
    // - it is most efficient to map and unmap all resources in a single call,
    //   and to have the map/unmap calls be the boundary between using the GPU
    //   for Direct3D and Cuda
    //
    static bool doit = true;

    if (doit){
        doit = true;
        cudaStream_t    stream = 0;
        const int nbResources = 2;
        cudaGraphicsResource *ppResources[nbResources] ={
            g_texture_2d.cudaResource,
			g_texture_cloud.cudaVelocityResource,
        };
        cudaGraphicsMapResources(nbResources, ppResources, stream);
        getLastCudaError("cudaGraphicsMapResources(3) failed");

        //
        // run kernels which will populate the contents of those textures
        //
        RunKernels();

        //
        // unmap the resources
        //
        cudaGraphicsUnmapResources(nbResources, ppResources, stream);
        getLastCudaError("cudaGraphicsUnmapResources(3) failed");
    }

    //
    // draw the scene using them
    //
   // CudaDrawScene();
}
////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void ApplicationClass::RunKernels()
{
    static float t = 0.0f;

    // populate the 2d texture
    {
        cudaArray *cuArray;
        cudaGraphicsSubResourceGetMappedArray(&cuArray, g_texture_2d.cudaResource, 0, 0);
        getLastCudaError("cudaGraphicsSubResourceGetMappedArray (cuda_texture_2d) failed");

        // kick off the kernel and send the staging buffer cudaLinearMemory as an argument to allow the kernel to write to it
        cuda_texture_2d(g_texture_2d.cudaLinearMemory, g_texture_2d.width, g_texture_2d.height, g_texture_2d.pitch, t);
        getLastCudaError("cuda_texture_2d failed");

        // then we want to copy cudaLinearMemory to the D3D texture, via its mapped form : cudaArray
        cudaMemcpy2DToArray(
            cuArray, // dst array
            0, 0,    // offset
            g_texture_2d.cudaLinearMemory, g_texture_2d.pitch,       // src
            g_texture_2d.width*4*sizeof(float), g_texture_2d.height, // extent
            cudaMemcpyDeviceToDevice); // kind
        getLastCudaError("cudaMemcpy2DToArray failed");
    }

	RunCloudKernals();
    t += 0.1f;
}

void ApplicationClass::RunCloudKernals(){
	// populate the volume texture
    {
        size_t pitchSlice = g_texture_cloud.pitch * g_texture_cloud.height;
        cudaArray *cuArray;
        cudaGraphicsSubResourceGetMappedArray(&cuArray, g_texture_cloud.cudaVelocityResource, 0, 0);
        getLastCudaError("cudaGraphicsSubResourceGetMappedArray (cuda_texture_3d) failed");

		float3 sizeWHD;
		sizeWHD.x = g_texture_cloud.width;
		sizeWHD.y = g_texture_cloud.height;
		sizeWHD.z = g_texture_cloud.depth;

        // kick off the kernel and send the staging buffer cudaLinearMemory as an argument to allow the kernel to write to it
        cuda_fluid_advect(g_texture_cloud.cudaAdvectLinearMemory,
			g_texture_cloud.cudaVelocityLinearMemory,
			sizeWHD,
			g_texture_cloud.pitch,
			pitchSlice);

        getLastCudaError("cuda_fluid_advect failed");
		
		// kick off the kernel and send the staging buffer cudaLinearMemory as an argument to allow the kernel to write to it
        cuda_fluid_divergence(g_texture_cloud.cudaDivergenceLinearMemory,
			g_texture_cloud.cudaAdvectLinearMemory,
			sizeWHD,
			g_texture_cloud.pitch,
			pitchSlice);

        getLastCudaError("cuda_fluid_divergence failed");

		// kick off the kernel and send the staging buffer cudaLinearMemory as an argument to allow the kernel to write to it
        cuda_fluid_jacobi(g_texture_cloud.cudaPressureLinearMemory,
			g_texture_cloud.cudaDivergenceLinearMemory,
			sizeWHD,
			g_texture_cloud.pitch,
			pitchSlice);

        getLastCudaError("cuda_fluid_jacobi failed");

		// kick off the kernel and send the staging buffer cudaLinearMemory as an argument to allow the kernel to write to it
        cuda_fluid_project(g_texture_cloud.cudaPressureLinearMemory,
			g_texture_cloud.cudaVelocityLinearMemory,
			sizeWHD,
			g_texture_cloud.pitch,
			pitchSlice);

        getLastCudaError("cuda_fluid_project failed");

        // then we want to copy cudaLinearMemory to the D3D texture, via its mapped form : cudaArray
        struct cudaMemcpy3DParms memcpyParams = {0};
        memcpyParams.dstArray = cuArray;
        memcpyParams.srcPtr.ptr = g_texture_cloud.cudaVelocityLinearMemory;
        memcpyParams.srcPtr.pitch = g_texture_cloud.pitch;
        memcpyParams.srcPtr.xsize = g_texture_cloud.width;
        memcpyParams.srcPtr.ysize = g_texture_cloud.height;
        memcpyParams.extent.width = g_texture_cloud.width;
        memcpyParams.extent.height = g_texture_cloud.height;
        memcpyParams.extent.depth = g_texture_cloud.depth;
        memcpyParams.kind = cudaMemcpyDeviceToDevice;
        cudaMemcpy3D(&memcpyParams);
        getLastCudaError("cudaMemcpy3D failed");
    }
}