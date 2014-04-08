////////////////////////////////////////////////////////////////////////////////
// Filename: applicationclass.cpp
////////////////////////////////////////////////////////////////////////////////
#include "application.h"
ApplicationClass::ApplicationClass():direct_3d_(0), input_(0),  camera_(0), player_position_(0),
	timer_(0),  FPS_(0), CPU_(0),  text_(0), light_object_(0), terrain_object_(0), cloud_object_(0),
	font_shader_(0), terrain_shader_(0),  texture_shader_(0), texture_to_texture_shader_(0), volume_shader_(0),
	face_shader_(0), velocity_cuda_(0), velocity_derivative_cuda_(0), pressure_divergence_cuda_(0),
	render_fullsize_texture_(0), down_sample_halfsize_texture_(0), fullsize_texture_(0), halfsize_texture_(0),
	full_screen_window_(0), is_done_once_(false)
{
}
ApplicationClass::ApplicationClass(const ApplicationClass& other):direct_3d_(0), input_(0),  camera_(0), player_position_(0),
	timer_(0),  FPS_(0), CPU_(0),  text_(0), light_object_(0), terrain_object_(0), cloud_object_(0),
	font_shader_(0), terrain_shader_(0),  texture_shader_(0), texture_to_texture_shader_(0), volume_shader_(0),
	face_shader_(0), velocity_cuda_(0), velocity_derivative_cuda_(0), pressure_divergence_cuda_(0),
	render_fullsize_texture_(0), down_sample_halfsize_texture_(0), fullsize_texture_(0), halfsize_texture_(0),
	full_screen_window_(0), is_done_once_(false)
{
}
ApplicationClass::~ApplicationClass(){
}
bool ApplicationClass::Initialize(HINSTANCE hinstance, HWND hwnd, int screen_width, int screen_height){
	bool result;
	// Set the size to sample down to.
	int down_sample_width = screen_width / 2;
	int down_sample_height = screen_height / 2;
	// Create the input object.  The input object will be used to handle reading the keyboard and mouse input from the user.
	input_ = new InputClass;
	if(!input_){
		return false;
	}
	// Initialize the input object.
	result = input_->Initialize(hinstance, hwnd, screen_width, screen_height);
	if(!result){
		MessageBox(hwnd, L"Could not initialize the input object.", L"Error", MB_OK);
		return false;
	}
	// Create the Direct3D object.
	//direct_3d_ = new D3DClass;
	direct_3d_ = new CUDAD3D;
	if(!direct_3d_){
		return false;
	}
	// Initialize the Direct3D object.
	result = direct_3d_->Initialize(screen_width, screen_height, VSYNC_ENABLED, hwnd, FULL_SCREEN, SCREEN_DEPTH, SCREEN_NEAR);
	if(!result){
		MessageBox(hwnd, L"Could not initialize DirectX 11.", L"Error", MB_OK);
		return false;
	}
	//Initialize the camera and position objects
	InitCamera();
	//Initialize the terrain, water, and cubes
	InitObjects(hwnd);
	//Initialize the text on screen. 
	InitText(hwnd, screen_width, screen_height);
	// Create the light object.
	light_object_ = new LightClass;
	if(!light_object_){
		return false;
	}
	// Initialize the light object.
	light_object_->SetAmbientColor(0.05f, 0.05f, 0.05f, 1.0f);
	light_object_->SetDiffuseColor(1.0f, 1.0f, 1.0f, 1.0f);
	light_object_->SetDirection(1.0f,-1.0f, 0.0f);
	//Initialize the texture to render to
	InitTextures(hwnd, screen_width, screen_height);
	//Initialize the sahders to be used.
	InitShaders(hwnd);
	// Create the full screen ortho window object.
	full_screen_window_ = new OrthoWindowClass;
	if(!full_screen_window_){
		return false;
	}
	// Initialize the full screen ortho window object.
	result = full_screen_window_->Initialize(direct_3d_->GetDevice(), screen_width, screen_height);
	if(!result){
		MessageBox(hwnd, L"Could not initialize the full screen ortho window object.", L"Error", MB_OK);
		return false;
	}
	result = InitCudaTextures();
	// Initialize Direct3D
	if(!result){
		MessageBox(hwnd, L"Could not initialize cuda textures.", L"Error", MB_OK);
		return false;
	}
	InitClouds();
	return true;
}
void ApplicationClass::InitClouds(){
	// 3D
	cudaGraphicsD3D11RegisterResource(&velocity_cuda_->cuda_resource_, velocity_cuda_->texture_, cudaGraphicsRegisterFlagsNone);
	getLastCudaError("cudaGraphicsD3D11RegisterResource (g_texture_cloud) failed");
	// create the buffer. pixel fmt is DXGI_FORMAT_R8G8B8A8_SNORM
	cudaMalloc(&velocity_cuda_->cuda_linear_memory_, velocity_cuda_->width_ * 4 * velocity_cuda_->height_ * velocity_cuda_->depth_);
	velocity_cuda_->pitch_ = velocity_cuda_->width_ * 4;
	getLastCudaError("cudaMallocPitch (g_texture_cloud) failed");
	cudaMemset(velocity_cuda_->cuda_linear_memory_, 1, velocity_cuda_->pitch_ * velocity_cuda_->height_ * velocity_cuda_->depth_);
	getLastCudaError("cudaMemset (g_texture_cloud) failed");
	
	cudaGraphicsD3D11RegisterResource(&velocity_derivative_cuda_->cuda_resource_, velocity_derivative_cuda_->texture_, cudaGraphicsRegisterFlagsNone);
	getLastCudaError("cudaGraphicsD3D11RegisterResource (g_texture_cloud) failed");
	// create the buffer. pixel fmt is DXGI_FORMAT_R8G8B8A8_SNORM
	cudaMalloc(&velocity_derivative_cuda_->cuda_linear_memory_, velocity_derivative_cuda_->width_ * 4 * velocity_derivative_cuda_->height_ * velocity_derivative_cuda_->depth_);
	velocity_derivative_cuda_->pitch_ = velocity_derivative_cuda_->width_ * 4;
	getLastCudaError("cudaMallocPitch (g_texture_cloud) failed");
	cudaMemset(velocity_derivative_cuda_->cuda_linear_memory_, 1, velocity_derivative_cuda_->pitch_ * velocity_derivative_cuda_->height_ * velocity_derivative_cuda_->depth_);
	getLastCudaError("cudaMemset (g_texture_cloud) failed");
	
	cudaGraphicsD3D11RegisterResource(&pressure_divergence_cuda_->cuda_resource_, pressure_divergence_cuda_->texture_, cudaGraphicsRegisterFlagsNone);
	getLastCudaError("cudaGraphicsD3D11RegisterResource (g_texture_cloud) failed");
	// create the buffer. pixel fmt is DXGI_FORMAT_R8G8B8A8_SNORM
	cudaMalloc(&pressure_divergence_cuda_->cuda_linear_memory_, pressure_divergence_cuda_->width_ * 4 * pressure_divergence_cuda_->height_ * pressure_divergence_cuda_->depth_);
	pressure_divergence_cuda_->pitch_ = pressure_divergence_cuda_->width_ * 4;
	getLastCudaError("cudaMallocPitch (g_texture_cloud) failed");
	cudaMemset(pressure_divergence_cuda_->cuda_linear_memory_, 1, pressure_divergence_cuda_->pitch_ * pressure_divergence_cuda_->height_ * pressure_divergence_cuda_->depth_);
	getLastCudaError("cudaMemset (g_texture_cloud) failed");
}
bool ApplicationClass::Frame(){
	bool result;
	if(input_){
		// Read the user input.
		result = input_->Frame();
		if(!result){
			return false;
		}
		// Check if the user pressed escape and wants to exit the application.
		if(input_->IsEscapePressed() == true){
			return false;
		}
	}
	// Update the system stats.
	timer_->Frame();
	FPS_->Frame();
	CPU_->Frame();
	// Update the FPS value in the text object.
	result = text_->SetFps(FPS_->GetFps(), direct_3d_->GetDeviceContext());
	if(!result){
		return false;
	}
	// Update the CPU usage value in the text object.
	result = text_->SetCpu(CPU_->GetCpuPercentage(), direct_3d_->GetDeviceContext());
	if(!result){
		return false;
	}
	if(input_){
		// Do the frame input processing.
		result = HandleInput(timer_->GetTime());
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
bool ApplicationClass::HandleInput(float frame_time){
	bool key_down, result;
	float pos_X, pos_Y, pos_Z, rot_X, rot_Y, rot_Z;
	// Set the frame time for calculating the updated position.
	player_position_->SetFrameTime(frame_time);
	//if left or a was pressed turn left
	key_down = (input_->IsLeftPressed() || input_->IsAPressed());
	player_position_->TurnLeft(key_down);
	//if right or d was pressed turn right
	key_down = (input_->IsRightPressed() || input_->IsDPressed());
	player_position_->TurnRight(key_down);
	//if up or w was pressed turn up
	key_down = (input_->IsUpPressed() || input_->IsWPressed());
	player_position_->MoveForward(key_down);
	//if down or s was pressed turn down
	key_down = (input_->IsDownPressed() || input_->IsSPressed());
	player_position_->MoveBackward(key_down);
	//if q was pressed move up
	key_down = input_->IsQPressed();
	player_position_->MoveUpward(key_down);
	//if e was pressed move down
	key_down = input_->IsEPressed();
	player_position_->MoveDownward(key_down);
	//if page up or z was pressed look up
	key_down = (input_->IsPgUpPressed() || input_->IsZPressed());
	player_position_->LookUpward(key_down);
	//if page down or x was pressed look down
	key_down = (input_->IsPgDownPressed() || input_->IsXPressed());
	player_position_->LookDownward(key_down);
	key_down = (input_->IsHPressed());
	if(key_down){
		direct_3d_->CreateBackFaceRaster();
	}
	key_down = (input_->IsRPressed());
	if(key_down){
		direct_3d_->CreateRaster();
	}
	// Get the view point position/rotation.
	player_position_->GetPosition(pos_X, pos_Y, pos_Z);
	player_position_->GetRotation(rot_X, rot_Y, rot_Z);
	// Set the position of the camera.
	camera_->SetPosition(pos_X, pos_Y, pos_Z);
	camera_->SetRotation(rot_X, rot_Y, rot_Z);
	// Update the position values in the text object.
	result = text_->SetCameraPosition(pos_X, pos_Y, pos_Z, direct_3d_->GetDeviceContext());
	if(!result){
		return false;
	}
	// Update the rotation values in the text object.
	result = text_->SetCameraRotation(rot_X, rot_Y, rot_Z, direct_3d_->GetDeviceContext());
	if(!result){
		return false;
	}
	return true;
}
bool ApplicationClass::Render(){
	bool result;
	CudaRender();
	RenderClouds();
	// First render the scene to a render texture.
	result = RenderSceneToTexture(render_fullsize_texture_);
	if(!result){
		return false;
	}
	
	//render the texture to the scene
	result = Render2DTextureScene(render_fullsize_texture_);
	if(!result){
		return false;
	}
	return true;
}
/*
*	Renders the objects on screen to a texture ready for post-processing.
*/
bool ApplicationClass::RenderSceneToTexture(RenderTextureClass* write_texture){
	D3DXMATRIX world_matrix, model_world_matrix, view_matrix, projection_matrix;
	bool result;
	// Set the render target to be the render to texture.
	write_texture->SetRenderTarget(direct_3d_->GetDeviceContext());
	// Clear the render to texture.
	write_texture->ClearRenderTarget(direct_3d_->GetDeviceContext(), 0.6f, 1.0f, 1.0f, 1.0f);
	// Generate the view matrix based on the camera's position.
	camera_->Render();
	// Get the world, view, and projection matrices from the camera and d3d objects.
	direct_3d_->GetWorldMatrix(world_matrix);
	camera_->GetViewMatrix(view_matrix);
	direct_3d_->GetProjectionMatrix(projection_matrix);
	// Render the terrain buffers.
	terrain_object_->Render(direct_3d_->GetDeviceContext());
	// Render the terrain using the terrain shader.
	result = terrain_shader_->Render(direct_3d_->GetDeviceContext(), terrain_object_->GetIndexCount(), world_matrix, view_matrix, projection_matrix, 
		light_object_->GetAmbientColor(), light_object_->GetDiffuseColor(), light_object_->GetDirection(), terrain_object_->GetTexture());
	if(!result){
		return false;
	}
	// Turn on the alpha blending before rendering the text.
	direct_3d_->TurnOnAlphaBlending();
	cloud_object_->Render(direct_3d_->GetDeviceContext());
	result = volume_shader_->Render(direct_3d_->GetDeviceContext(), terrain_object_->GetIndexCount(), world_matrix, view_matrix, projection_matrix, 
		cloud_object_->GetFrontShaderResource(), cloud_object_->GetBackShaderResource(), velocity_cuda_->sr_view_,cloud_object_->GetScale());
	if(!result){
		return false;
	}
	
	// Turn off alpha blending after rendering the text.
	direct_3d_->TurnOffAlphaBlending();
	// Reset the render target back to the original back buffer and not the render to texture anymore.
	direct_3d_->SetBackBufferRenderTarget();
	// Reset the viewport back to the original.
	direct_3d_->ResetViewport();
	return true;
}
bool ApplicationClass::RenderClouds(){
	D3DXMATRIX world_matrix, model_world_matrix, view_matrix, projection_matrix;
	bool result;
	// Set the render target to be the render to texture.
	cloud_object_->GetFrontTexture()->SetRenderTarget(direct_3d_->GetDeviceContext());
	// Clear the render to texture.
	cloud_object_->GetFrontTexture()->ClearRenderTarget(direct_3d_->GetDeviceContext(), 0.f, 0.f, 0.f, 1.0f);
	// Generate the view matrix based on the camera's position.
	camera_->Render();
	// Get the world, view, and projection matrices from the camera and d3d objects.
	direct_3d_->GetWorldMatrix(world_matrix);
	camera_->GetViewMatrix(view_matrix);
	direct_3d_->GetProjectionMatrix(projection_matrix);
	// Render the terrain buffers.
	cloud_object_->Render(direct_3d_->GetDeviceContext());
	// Render the terrain using the terrain shader.
	result = face_shader_->Render(direct_3d_->GetDeviceContext(), cloud_object_->GetIndexCount(), world_matrix, view_matrix, projection_matrix, cloud_object_->GetScale());
	if(!result){
		return false;
	}
	// Reset the render target back to the original back buffer and not the render to texture anymore.
	direct_3d_->SetBackBufferRenderTarget();
	// Reset the viewport back to the original.
	direct_3d_->ResetViewport();
	direct_3d_->CreateBackFaceRaster();
	cloud_object_->Render(direct_3d_->GetDeviceContext());
	// Set the render target to be the render to texture.
	cloud_object_->GetBackTexture()->SetRenderTarget(direct_3d_->GetDeviceContext());
	// Clear the render to texture.
	cloud_object_->GetBackTexture()->ClearRenderTarget(direct_3d_->GetDeviceContext(), 0.f, 0.f, 0.f, 1.0f);
	// Generate the view matrix based on the camera's position.
	camera_->Render();
	// Get the world, view, and projection matrices from the camera and d3d objects.
	direct_3d_->GetWorldMatrix(world_matrix);
	camera_->GetViewMatrix(view_matrix);
	direct_3d_->GetProjectionMatrix(projection_matrix);
	// Render the terrain buffers.
	cloud_object_->Render(direct_3d_->GetDeviceContext());
	// Render the terrain using the terrain shader.
	result = face_shader_->Render(direct_3d_->GetDeviceContext(), cloud_object_->GetIndexCount(), world_matrix, view_matrix, projection_matrix, cloud_object_->GetScale());
	if(!result){
		return false;
	}
	// Reset the render target back to the original back buffer and not the render to texture anymore.
	direct_3d_->SetBackBufferRenderTarget();
	// Reset the viewport back to the original.
	direct_3d_->ResetViewport();
	direct_3d_->CreateRaster();
	return true;
}
/*
*	Takes a texture and performs shader actions on it.
*/
bool ApplicationClass::RenderTexture(ShaderClass *shader, RenderTextureClass *read_texture, RenderTextureClass *write_texture, OrthoWindowClass *window){
	D3DXMATRIX ortho_matrix;
	bool result;

	float screen_size_Y = (float)write_texture->GetTextureHeight();
	float screen_size_X = (float)write_texture->GetTextureWidth();

	// Set the render target to be the render to texture.
	write_texture->SetRenderTarget(direct_3d_->GetDeviceContext());
	// Clear the render to texture.
	write_texture->ClearRenderTarget(direct_3d_->GetDeviceContext(), 0.0f, 0.0f, 0.0f, 1.0f);
	// Generate the view matrix based on the camera's position.
	camera_->Render();
	// Get the ortho matrix from the render to texture since texture has different dimensions being that it is smaller.
	write_texture->GetOrthoMatrix(ortho_matrix);
	// Turn off the Z buffer to begin all 2D rendering.
	direct_3d_->TurnZBufferOff();
	// Put the small ortho window vertex and index buffers on the graphics pipeline to prepare them for drawing.
	window->Render(direct_3d_->GetDeviceContext());
	// Render the small ortho window using the texture shader and the render to texture of the scene as the texture resource.
	result = shader->Render(direct_3d_->GetDeviceContext(), window->GetIndexCount(), ortho_matrix, 
		read_texture->GetShaderResourceView(),screen_size_Y,screen_size_X);
	if(!result){
		return false;
	}
	// Turn the Z buffer back on now that all 2D rendering has completed.
	direct_3d_->TurnZBufferOn();
	// Reset the render target back to the original back buffer and not the render to texture anymore.
	direct_3d_->SetBackBufferRenderTarget();
	// Reset the viewport back to the original.
	direct_3d_->ResetViewport();
	return true;
}
/*
* Renders a texture to the screen.
*/
bool ApplicationClass::Render2DTextureScene(RenderTextureClass* read_texture){
	D3DXMATRIX world_matrix, view_matrix, ortho_matrix, projection_matrix;
	bool result;
	// Clear the buffers to begin the scene.
	direct_3d_->BeginScene(0.0f, 0.0f, 0.0f, 1.0f);
	// Generate the view matrix based on the camera's position.
	camera_->Render();
	// Get the world, view, and ortho matrices from the camera and d3d objects.
	camera_->GetViewMatrix(view_matrix);
	direct_3d_->GetWorldMatrix(world_matrix);
	direct_3d_->GetOrthoMatrix(ortho_matrix);
	// Turn off the Z buffer to begin all 2D rendering.
	direct_3d_->TurnZBufferOff();
	// Put the full screen ortho window vertex and index buffers on the graphics pipeline to prepare them for drawing.
	full_screen_window_->Render(direct_3d_->GetDeviceContext());
	//Render the full screen ortho window using the texture shader and the full screen sized blurred render to texture resource.
	result = texture_to_texture_shader_->Render(direct_3d_->GetDeviceContext(), full_screen_window_->GetIndexCount(), ortho_matrix, read_texture->GetShaderResourceView());
	if(!result){
		return false;
	}
	// Turn on the alpha blending before rendering the text.
	direct_3d_->TurnOnAlphaBlending();
	// Render the text user interface elements.
	result = text_->Render(direct_3d_->GetDeviceContext(), font_shader_, world_matrix, ortho_matrix);
	if(!result){
		return false;
	}
	// Turn off alpha blending after rendering the text.
	direct_3d_->TurnOffAlphaBlending();
	// Turn the Z buffer back on now that all 2D rendering has completed.
	direct_3d_->TurnZBufferOn();
	// Present the rendered scene to the screen.
	direct_3d_->EndScene();
	return true;
}
bool ApplicationClass::InitText(HWND hwnd, int screen_width , int screen_height){
	D3DXMATRIX base_view_matrix;
	char video_card[128];
	int video_memory;
	bool result = true;
	// Create the timer object.
	timer_ = new TimerClass;
	if(!timer_){
		return false;
	}
	// Initialize the timer object.
	result = timer_->Initialize();
	if(!result){
		MessageBox(hwnd, L"Could not initialize the timer object.", L"Error", MB_OK);
		return false;
	}
	// Create the fps object.
	FPS_ = new FpsClass;
	if(!FPS_){
		return false;
	}
	// Initialize the fps object.
	FPS_->Initialize();
	// Create the cpu object.
	CPU_ = new CpuClass;
	if(!CPU_){
		return false;
	}
	// Initialize the cpu object.
	CPU_->Initialize();
	// Create the text object.
	text_ = new TextClass;
	if(!text_){
		return false;
	}
	camera_->GetViewMatrix(base_view_matrix);
	// Initialize the text object.
	result = text_->Initialize(direct_3d_->GetDevice(), direct_3d_->GetDeviceContext(), hwnd, screen_width, screen_height, base_view_matrix);
	if(!result){
		MessageBox(hwnd, L"Could not initialize the text object.", L"Error", MB_OK);
		return false;
	}
	// Retrieve the video card information.
	direct_3d_->GetVideoCardInfo(video_card, video_memory);
	// Set the video card information in the text object.
	result = text_->SetVideoCardInfo(video_card, video_memory, direct_3d_->GetDeviceContext());
	if(!result){
		MessageBox(hwnd, L"Could not set video card info in the text object.", L"Error", MB_OK);
		return false;
	}
	return true;
}
bool ApplicationClass::InitObjects(HWND hwnd){
	bool result;
	// Create the terrain object.
	terrain_object_ = new TerrainClass;
	if(!terrain_object_){
		return false;
	}
	// Initialize the terrain object.
	result = terrain_object_->Initialize(direct_3d_->GetDevice(),"Data/heightmap01.bmp" ,L"Data/ground.dds");
	if(!result){
		MessageBox(hwnd, L"Could not initialize the terrain object.", L"Error", MB_OK);
		return false;
	}
	cloud_object_ = new CloudClass;
	if(!cloud_object_){
		return false;
	}
	// Initialize the terrain object.
	result = cloud_object_->Initialize(direct_3d_->GetDevice(), 800,600,SCREEN_DEPTH, SCREEN_NEAR);
	if(!result){
		MessageBox(hwnd, L"Could not initialize the cloud object.", L"Error", MB_OK);
		return false;
	}
	return true;
}
bool ApplicationClass::InitTextures(HWND hwnd, int screen_width, int screen_height){
	bool result;
	int down_sample_width = screen_width / 2;
	int down_sample_height = screen_height / 2;
	// Create the render to texture object.
	render_fullsize_texture_ = new RenderTextureClass;
	if(!render_fullsize_texture_){
		return false;
	}
	// Initialize the render to texture object.
	result = render_fullsize_texture_->Initialize(direct_3d_->GetDevice(), screen_width, screen_height, SCREEN_DEPTH, SCREEN_NEAR);
	if(!result){
		MessageBox(hwnd, L"Could not initialize the render to texture object.", L"Error", MB_OK);
		return false;
	}
	// Create the up sample render to texture object.
	fullsize_texture_ = new RenderTextureClass;
	if(!fullsize_texture_){
		return false;
	}
	// Initialize the up sample render to texture object.
	result = fullsize_texture_->Initialize(direct_3d_->GetDevice(), screen_width, screen_height, SCREEN_DEPTH, SCREEN_NEAR);
	if(!result){
		MessageBox(hwnd, L"Could not initialize the full size render to texture object.", L"Error", MB_OK);
		return false;
	}
	// Create the down sample render to texture object.
	down_sample_halfsize_texture_ = new RenderTextureClass;
	if(!down_sample_halfsize_texture_){
		return false;
	}
	// Initialize the down sample render to texture object.
	result = down_sample_halfsize_texture_->Initialize(direct_3d_->GetDevice(), down_sample_width, down_sample_height, SCREEN_DEPTH, SCREEN_NEAR);
	if(!result){
		MessageBox(hwnd, L"Could not initialize the down sample render to texture object.", L"Error", MB_OK);
		return false;
	}
	//create a second down sample texture for performing convolutions on
	halfsize_texture_ = new RenderTextureClass;
	if (!halfsize_texture_){
		return false;
	}
	result = halfsize_texture_->Initialize(direct_3d_->GetDevice(), down_sample_width, down_sample_height, SCREEN_DEPTH, SCREEN_NEAR);
	if (!result){
		MessageBox(hwnd, L"Could not initialize the second half size to texture object.", L"Error", MB_OK);		
		return false;
	}
	return true;
}
bool ApplicationClass::InitCamera(){
	// Create the camera object.
	camera_ = new CameraClass;
	if(!camera_){
		return false;
	}
	// Initialize a base view matrix with the camera for 2D user interface rendering.
	camera_->SetPosition(0.0f, 0.0f, -1.0f);
	camera_->Render();
	// Set the initial position of the camera.
	camera_->SetPosition(0.0f, 2.0f, -7.0f);
	// Create the position object.
	player_position_ = new PositionClass;
	if(!player_position_){
		return false;
	}
	// Set the initial position of the viewer to the same as the initial camera position.
	player_position_->SetPosition(0.0f, 2.0f, -7.0f);
	return true;
}
bool ApplicationClass::InitShaders(HWND hwnd){
	// Create the font shader object.
	font_shader_ = new FontShaderClass;
	if(!font_shader_){
		return false;
	}
	// Initialize the font shader object.
	bool result = font_shader_->Initialize(direct_3d_->GetDevice(), hwnd);
	if(!result){
		MessageBox(hwnd, L"Could not initialize the font shader object.", L"Error", MB_OK);
		return false;
	}
	if(!InitTextureShaders(hwnd)){
		MessageBox(hwnd, L"Could not initialize the texture shaders.", L"Error", MB_OK);
		return false;
	}	
	if(!InitObjectShaders(hwnd)){
		MessageBox(hwnd, L"Could not initialize the object shaders.", L"Error", MB_OK);
		return false;
	}
	return true;
}
bool ApplicationClass::InitTextureShaders(HWND hwnd){
	bool result;
	// Create the texture to texture shader object.
	texture_to_texture_shader_ = new TextureToTextureShaderClass;
	if(!texture_to_texture_shader_){
		return false;
	}
	// Initialize the texture to texture shader object.
	result = texture_to_texture_shader_->Initialize(direct_3d_->GetDevice(), hwnd);
	if(!result){
		MessageBox(hwnd, L"Could not initialize the texture to texture shader object.", L"Error", MB_OK);
		return false;
	}
	//create the texture shader object
	texture_shader_ = new TextureShaderClass;
	if(!texture_shader_){
		return false;
	}
	// Initialize the texture shader object.
	result = texture_shader_->Initialize(direct_3d_->GetDevice(), hwnd);
	if(!result){
		MessageBox(hwnd, L"Could not initialize the texture shader object.", L"Error", MB_OK);
		return false;
	}
	return true;
}
bool ApplicationClass::InitObjectShaders(HWND hwnd){
	bool result;
	// Create the terrain shader object.
	terrain_shader_ = new TerrainShaderClass;
	if(!terrain_shader_){
		return false;
	}
	// Initialize the terrain shader object.
	result = terrain_shader_->Initialize(direct_3d_->GetDevice(), hwnd);
	if(!result){
		return false;
	}
	volume_shader_ = new VolumeShader;
	if (!volume_shader_){
		return false;
	}
	result = volume_shader_->Initialize(direct_3d_->GetDevice(),hwnd);
	if(!result){
		return false;
	}
	face_shader_ = new FaceShader;
	if (!face_shader_){
		return false;
	}
	result = face_shader_->Initialize(direct_3d_->GetDevice(),hwnd);
	if(!result){
		return false;
	}
	return true;
}
//-----------------------------------------------------------------------------
// Name: InitTextures()
// Desc: Initializes Direct3D Textures (allocation and initialization)
//-----------------------------------------------------------------------------
bool ApplicationClass::InitCudaTextures(){
	int offset_shader = 0;
	int3 size_WHD = {64,64,64};
	ID3D11Device* d3d_device = direct_3d_->GetDevice();
	ID3D11DeviceContext* d3d_device_context = direct_3d_->GetDeviceContext();
	D3D11_TEXTURE3D_DESC desc;

	velocity_cuda_ = new fluid_texture;
	if (!velocity_cuda_){
		return false;
	}
	velocity_derivative_cuda_ = new fluid_texture;
	if (!velocity_derivative_cuda_){
		return false;
	}
	pressure_divergence_cuda_ = new fluid_texture;
	if (!pressure_divergence_cuda_){
		return false;
	}

	velocity_cuda_->width_  = size_WHD.x;
	velocity_cuda_->height_ = size_WHD.y;
	velocity_cuda_->depth_  = size_WHD.z;
	
	ZeroMemory(&desc, sizeof(D3D11_TEXTURE3D_DESC));
	desc.Width = velocity_cuda_->width_;
	desc.Height = velocity_cuda_->height_;
	desc.Depth = velocity_cuda_->depth_;
	desc.MipLevels = 1;
	desc.Format = DXGI_FORMAT_R8G8B8A8_SNORM;
	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

	if (FAILED(d3d_device->CreateTexture3D(&desc, NULL, &velocity_cuda_->texture_))){
		return false;
	}
	if (FAILED(d3d_device->CreateShaderResourceView(velocity_cuda_->texture_, NULL, &velocity_cuda_->sr_view_))){
		return false;
	}
	d3d_device_context->PSSetShaderResources(offset_shader++, 1, &velocity_cuda_->sr_view_);

	velocity_derivative_cuda_->width_  = size_WHD.x;
	velocity_derivative_cuda_->height_ = size_WHD.y;
	velocity_derivative_cuda_->depth_  = size_WHD.z;

	desc.Width = velocity_derivative_cuda_->width_;
	desc.Height = velocity_derivative_cuda_->height_;
	desc.Depth = velocity_derivative_cuda_->depth_;

	if (FAILED(d3d_device->CreateTexture3D(&desc, NULL, &velocity_derivative_cuda_->texture_))){
		return false;
	}
	if (FAILED(d3d_device->CreateShaderResourceView(velocity_derivative_cuda_->texture_, NULL, &velocity_derivative_cuda_->sr_view_))){
		return false;
	}
	d3d_device_context->PSSetShaderResources(offset_shader++, 1, &velocity_derivative_cuda_->sr_view_);

	pressure_divergence_cuda_->width_  = size_WHD.x;
	pressure_divergence_cuda_->height_ = size_WHD.y;
	pressure_divergence_cuda_->depth_  = size_WHD.z;

	desc.Width = pressure_divergence_cuda_->width_;
	desc.Height = pressure_divergence_cuda_->height_;
	desc.Depth = pressure_divergence_cuda_->depth_;

	if (FAILED(d3d_device->CreateTexture3D(&desc, NULL, &pressure_divergence_cuda_->texture_))){
		return false;
	}
	if (FAILED(d3d_device->CreateShaderResourceView(pressure_divergence_cuda_->texture_, NULL, &pressure_divergence_cuda_->sr_view_))){
		return false;
	}
	d3d_device_context->PSSetShaderResources(offset_shader, 1, &pressure_divergence_cuda_->sr_view_);
	return true;
}
//-----------------------------------------------------------------------------
// Name: CudaRender()
// Desc: Launches the CUDA kernels to fill in the texture data
//-----------------------------------------------------------------------------
void ApplicationClass::CudaRender(){
	//map the resources we've registered so we can access them in Cuda
	//it is most efficient to map and unmap all resources in a single call,
	//and to have the map/unmap calls be the boundary between using the GPU
	//for Direct3D and Cuda
	cudaStream_t stream = 0;
	const int num_resources = 3;
	cudaGraphicsResource *resources[num_resources] ={
		velocity_cuda_->cuda_resource_,
		velocity_derivative_cuda_->cuda_resource_,
		pressure_divergence_cuda_->cuda_resource_
	};
	cudaGraphicsMapResources(num_resources, resources, stream);
	getLastCudaError("cudaGraphicsMapResources(3) failed");
	// run kernels which will populate the contents of those textures
	RunCloudKernals();
	// unmap the resources
	cudaGraphicsUnmapResources(num_resources, resources, stream);
	getLastCudaError("cudaGraphicsUnmapResources(3) failed");
}
void ApplicationClass::RunCloudKernals(){
	// populate the volume texture
	int pressure_index, divergence_index = 0;
	size_t pitch_slice = velocity_cuda_->pitch_ * velocity_cuda_->height_;
	cudaArray *cuda_velocity_array;
	float3 size_WHD ={velocity_cuda_->width_, velocity_cuda_->height_, velocity_cuda_->depth_};

	cudaGraphicsSubResourceGetMappedArray(&cuda_velocity_array, velocity_cuda_->cuda_resource_, 0, 0);
	getLastCudaError("cudaGraphicsSubResourceGetMappedArray (cuda_texture_3d) failed");
	if(is_done_once_ == false){
		cuda_fluid_initial(velocity_cuda_->cuda_linear_memory_, size_WHD, velocity_cuda_->pitch_, pitch_slice, 120.f);
		getLastCudaError("cuda_fluid_initial failed");
		cuda_fluid_initial(velocity_derivative_cuda_->cuda_linear_memory_, size_WHD, velocity_derivative_cuda_->pitch_, pitch_slice, 0.f);
		getLastCudaError("cuda_fluid_initial failed");
		cuda_fluid_initial(pressure_divergence_cuda_->cuda_linear_memory_, size_WHD, pressure_divergence_cuda_->pitch_, pitch_slice, 0.f);
		getLastCudaError("cuda_fluid_initial failed");
		is_done_once_ = true;
	}

	// kick off the kernel and send the staging buffer cuda_linear_memory_ as an argument to allow the kernel to write to it
	cuda_fluid_advect(velocity_derivative_cuda_->cuda_linear_memory_, velocity_cuda_->cuda_linear_memory_, size_WHD, velocity_cuda_->pitch_, pitch_slice);
	getLastCudaError("cuda_fluid_advect failed");
	divergence_index = 1;

	// kick off the kernel and send the staging buffer cuda_linear_memory_ as an argument to allow the kernel to write to it
	cuda_fluid_forces(velocity_derivative_cuda_->cuda_linear_memory_, velocity_cuda_->cuda_linear_memory_, size_WHD, velocity_cuda_->pitch_, pitch_slice);
	getLastCudaError("cuda_fluid_forces failed");

	// kick off the kernel and send the staging buffer cuda_linear_memory_ as an argument to allow the kernel to write to it
	cuda_fluid_divergence(pressure_divergence_cuda_->cuda_linear_memory_, velocity_derivative_cuda_->cuda_linear_memory_, size_WHD, velocity_cuda_->pitch_, pitch_slice, divergence_index);
	getLastCudaError("cuda_fluid_divergence failed");
	for(int i = 0; i < 16; i++){
		if(i%2 == 0){
			pressure_index = 0;
			divergence_index = 1;
		}else{
			pressure_index = 1;
			divergence_index = 0;
		}
		// kick off the kernel and send the staging buffer cuda_linear_memory_ as an argument to allow the kernel to write to it
		cuda_fluid_jacobi(pressure_divergence_cuda_->cuda_linear_memory_, size_WHD, velocity_cuda_->pitch_, pitch_slice, pressure_index, divergence_index);
		getLastCudaError("cuda_fluid_jacobi failed");
	}
	
	// kick off the kernel and send the staging buffer cuda_linear_memory_ as an argument to allow the kernel to write to it
	cuda_fluid_project(pressure_divergence_cuda_->cuda_linear_memory_, velocity_cuda_->cuda_linear_memory_, size_WHD, velocity_cuda_->pitch_, pitch_slice, pressure_index);
	getLastCudaError("cuda_fluid_project failed");
	
	// then we want to copy cuda_linear_memory_ to the D3D texture, via its mapped form : cudaArray
	struct cudaMemcpy3DParms memcpyParams = {0};
	memcpyParams.dstArray = cuda_velocity_array;
	memcpyParams.srcPtr.ptr = velocity_cuda_->cuda_linear_memory_;
	memcpyParams.srcPtr.pitch = velocity_cuda_->pitch_;
	memcpyParams.srcPtr.xsize = velocity_cuda_->width_;
	memcpyParams.srcPtr.ysize = velocity_cuda_->height_;
	memcpyParams.extent.width = velocity_cuda_->width_;
	memcpyParams.extent.height = velocity_cuda_->height_;
	memcpyParams.extent.depth = velocity_cuda_->depth_;
	memcpyParams.kind = cudaMemcpyDeviceToDevice;
	cudaMemcpy3D(&memcpyParams);
	getLastCudaError("cudaMemcpy3D failed");
}
void ApplicationClass::Shutdown(){
	if(full_screen_window_){
		full_screen_window_->Shutdown();
		delete full_screen_window_;
		full_screen_window_ = 0;
	}
	// Release the light object.
	if(light_object_){
		delete light_object_;
		light_object_ = 0;
	}
	//Shutdown functions
	ShutdownText();
	ShutdownTextures();
	ShutdownShaders();
	ShutdownObjects();
	ShutdownCamera();
	// Release the Direct3D object.
	if(direct_3d_){
		direct_3d_->Shutdown();
		delete direct_3d_;
		direct_3d_ = 0;
	}
	// Release the input object.
	if(input_){
		input_->Shutdown();
		delete input_;
		input_ = 0;
	}
	return;
}
void ApplicationClass::ShutdownText(){
	// Release the text object.
	if(text_){
		text_->Shutdown();
		delete text_;
		text_ = 0;
	}
	// Release the cpu object.
	if(CPU_){
		CPU_->Shutdown();
		delete CPU_;
		CPU_ = 0;
	}
	// Release the fps object.
	if(FPS_){
		delete FPS_;
		FPS_ = 0;
	}
	// Release the timer object.
	if(timer_){
		delete timer_;
		timer_ = 0;
	}
}
void ApplicationClass::ShutdownObjects(){
	// Release the terrain object.
	if(terrain_object_){
		terrain_object_->Shutdown();
		delete terrain_object_;
		terrain_object_ = 0;
	}
}
void ApplicationClass::ShutdownTextures(){
	// Release the up sample render to texture object.
	if(fullsize_texture_){
		fullsize_texture_->Shutdown();
		delete fullsize_texture_;
		fullsize_texture_ = 0;
	}
	// Release the down sample render to texture object.
	if(down_sample_halfsize_texture_){
		down_sample_halfsize_texture_->Shutdown();
		delete down_sample_halfsize_texture_;
		down_sample_halfsize_texture_ = 0;
	}
	// Release the render to texture object.
	if(render_fullsize_texture_){
		render_fullsize_texture_->Shutdown();
		delete render_fullsize_texture_;
		render_fullsize_texture_ = 0;
	}
	//release half size texture
	if (halfsize_texture_){
		halfsize_texture_->Shutdown();
		delete halfsize_texture_;
		halfsize_texture_ = 0;
	}
	if (velocity_cuda_){
		delete velocity_cuda_;
		velocity_cuda_ = NULL;
	}
	if (velocity_derivative_cuda_){
		delete velocity_derivative_cuda_;
		velocity_derivative_cuda_ = NULL;	}
	if (pressure_divergence_cuda_){
		delete pressure_divergence_cuda_;
		pressure_divergence_cuda_ = NULL;;
	}
}
void ApplicationClass::ShutdownCamera(){
	//relase position object
	if(player_position_){
		delete player_position_;
		player_position_ = 0;
	}
	// Release the camera object.
	if(camera_){
		delete camera_;
		camera_ = 0;
	}
}
void ApplicationClass::ShutdownShaders(){
	// Release the texture shader object.
	if(texture_shader_){
		texture_shader_->Shutdown();
		delete texture_shader_;
		texture_shader_ = 0;
	}
	//release texture to texture shader
	if(texture_to_texture_shader_){
		texture_to_texture_shader_->Shutdown();
		delete texture_to_texture_shader_;
		texture_to_texture_shader_ = 0;
	}
	// Release the font shader object.
	if(font_shader_){
		font_shader_->Shutdown();
		delete font_shader_;
		font_shader_ = 0;
	}
	// Release the terrain shader object.
	if(terrain_shader_){
		terrain_shader_->Shutdown();
		delete terrain_shader_;
		terrain_shader_ = 0;
	}
}