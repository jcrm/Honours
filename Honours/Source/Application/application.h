////////////////////////////////////////////////////////////////////////////////
// Filename: applicationclass.h
////////////////////////////////////////////////////////////////////////////////
#ifndef _APPLICATIONCLASS_H_
#define _APPLICATIONCLASS_H_
/////////////
// GLOBALS //
/////////////
//const bool FULL_SCREEN = true;
const bool FULL_SCREEN = false;
const bool VSYNC_ENABLED = true;
const float SCREEN_DEPTH = 1000.0f;
const float SCREEN_NEAR = 0.1f;
///////////////////////
// MY CLASS INCLUDES //
///////////////////////
#include <vector>
#include <windows.h>
#include <mmsystem.h>
#include <string>
// This header inclues all the necessary D3D11 and CUDA includes
#include <dynlink_d3d11.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
// includes, project
#include <rendercheck_d3d11.h>
#include <helper_cuda.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include "../Managers/input.h"
#include "../Managers/camera.h"
#include "../Managers/position.h"
#include "../Managers/light.h"
#include "../Managers/ortho_window.h"
#include "../DirectX/d3dclass.h"
#include "../DirectX/cuda_d3d.h"
#include "../Objects/terrain.h"
#include "../Objects/cloud_box.h"
#include "../Objects/particle_system.h"
#include "../Text/timerclass.h"
#include "../Text/fpsclass.h"
#include "../Text/cpuclass.h"
#include "../Text/textclass.h"
#include "../Shaders/particle_shader.h"
#include "../Shaders/fontshaderclass.h"
#include "../Shaders/terrain_shader.h"
#include "../Shaders/texture_shader.h"
#include "../Shaders/texture_to_texture_shader.h"
#include "../Shaders/shader.h"
#include "../Shaders/volume_shader.h"
#include "../Shaders/face_shader.h"
#include "../Shaders/merge_texture_shader.h"
#include "../Textures/render_texture.h"
#include "../CUDA/cuda_structs.h"
#include "../CUDA/cuda_kernals.h"
#include "../CUDA/cuda_header.h"

////////////////////////////////////////////////////////////////////////////////
// Class name: ApplicationClass
////////////////////////////////////////////////////////////////////////////////
class ApplicationClass
{
public:
	ApplicationClass();
	ApplicationClass(const ApplicationClass&);
	~ApplicationClass();
	bool Initialize(HINSTANCE, HWND, int, int);
	void Shutdown();
	bool Frame();
private:
	bool HandleInput(float);
	//Render Functions
	bool Render();
	bool RenderSceneToTexture(RenderTextureClass* );
	bool RenderMergeTexture(RenderTextureClass*, RenderTextureClass*, RenderTextureClass*);
	bool RenderParticlesToTexture(RenderTextureClass* );
	bool RenderTexture(ShaderClass*, RenderTextureClass*, RenderTextureClass*, OrthoWindowClass*);
	bool Render2DTextureScene(RenderTextureClass*);
	//Init functions
	bool InitObjects(HWND);
	bool InitTextures(HWND, int, int);
	bool InitText(HWND, int, int);
	bool InitShaders(HWND);
	bool InitObjectShaders(HWND);
	bool InitTextureShaders(HWND);
	bool InitCamera();
	//Shutdown functions
	void ShutdownObjects();
	void ShutdownText();
	void ShutdownTextures();
	void ShutdownCamera();
	void ShutdownShaders();
	void ShutdownCudaResources();

	bool InitCudaTextures();
	void CudaRender();
	void RunKernels();
	void InitClouds();
	void RunCloudKernals();
	bool RenderClouds();
	
private:
	InputClass* input_;
	CUDAD3D* direct_3d_;
	CameraClass* camera_;
	
	TimerClass* timer_;
	PositionClass* player_position_;
	FpsClass* FPS_;
	CpuClass* CPU_;
	TextClass* text_;
	LightClass* light_object_;
	OrthoWindowClass *full_screen_window_;
	//the points for the different objects
	TerrainClass* terrain_object_;
	CloudClass* cloud_object_;
	std::vector<ParticleSystemClass*> rain_systems_;
	//textures to render to
	RenderTextureClass *render_fullsize_texture_, *fullsize_texture_, *merge_texture_, *particle_texture_;
	//the different shaders used
	TextureShaderClass* texture_shader_;
	TextureToTextureShaderClass* texture_to_texture_shader_;
	FontShaderClass* font_shader_;
	TerrainShaderClass* terrain_shader_;
	VolumeShader* volume_shader_;
	FaceShader* face_shader_;
	ParticleShaderClass* particle_shader_;
	MergeTextureShaderClass* merge_shader_;
	//cuda textures
	fluid_texture *velocity_cuda_;
	fluid_texture *velocity_derivative_cuda_;
	fluid_texture *pressure_divergence_cuda_;
	fluid_texture *thermo_cuda_;
	fluid_texture *water_continuity_cuda_;
	fluid_texture *water_continuity_rain_cuda_;
	rain_texture *rain_cuda_;
	bool is_done_once_;
	float output[64*64*PIXEL_FMT_SIZE_RGBA* sizeof(float*)];
};
#endif