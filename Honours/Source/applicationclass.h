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
#include <windows.h>
#include <mmsystem.h>

// This header inclues all the necessary D3D11 and CUDA includes
#include <dynlink_d3d11.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>

// includes, project
#include <rendercheck_d3d11.h>
#include <helper_cuda.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

#include "Managers/inputclass.h"
#include "Managers/cameraclass.h"
#include "Managers/positionclass.h"
#include "Managers/lightclass.h"
#include "Managers/orthowindowclass.h"

#include "DirectX/d3dclass.h"
#include "DirectX/cudad3d.h"

#include "Objects/terrainclass.h"
#include "Objects/CloudBox.h"

#include "Text/timerclass.h"
#include "Text/fpsclass.h"
#include "Text/cpuclass.h"
#include "Text/textclass.h"

#include "Shaders/fontshaderclass.h"
#include "Shaders/terrainshaderclass.h"
#include "Shaders/textureshaderclass.h"
#include "Shaders/texturetotextureshaderclass .h"
#include "Shaders/ShaderClass.h"
#include "Shaders/VolumeShader.h"
#include "Shaders/FaceShader.h"
#include "Shaders/PositionShader.h"

#include "Textures/rendertextureclass.h"

#include "CUDA/cuda_structs.h"
#include "CUDA/cuda_kernals.h"

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
	bool RenderSceneToTexture(RenderTextureClass* mWrite);
	bool RenderTexture(ShaderClass *mShader, RenderTextureClass *mReadTexture, RenderTextureClass *mWriteTexture, OrthoWindowClass *mWindow);
	bool RenderFrontTexture(RenderTextureClass *mReadTexture, RenderTextureClass *mWriteTexture, OrthoWindowClass *mWindow);
	bool RenderMergeTexture(RenderTextureClass *readTexture, RenderTextureClass *readTexture2, RenderTextureClass *writeTexture, OrthoWindowClass *window);
	bool Render2DTextureScene(RenderTextureClass* mRead);
	//Init functions
	bool InitObjects(HWND hwnd);
	bool InitTextures(HWND hwnd, int screenWidth, int screenHeight);
	bool InitText(HWND hwnd, int screenWidth , int screenHeight);
	bool InitShaders(HWND hwnd);
	bool InitObjectShaders(HWND hwnd);
	bool InitTextureShaders(HWND hwnd);
	bool InitCamera();
	//Shutdown functions
	void ShutdownObjects();
	void ShutdownText();
	void ShutdownTextures();
	void ShutdownCamera();
	void ShutdownShaders();

	bool InitCudaTextures();
	void CudaRender();
	void RunKernels();
	void InitClouds();
	void RunCloudKernals();
private:
	InputClass* m_Input;
	CUDAD3D* m_Direct3D;
	CameraClass* m_Camera;
	
	TimerClass* m_Timer;
	PositionClass* m_Position;
	FpsClass* m_Fps;
	CpuClass* m_Cpu;
	TextClass* m_Text;
	LightClass* m_Light;
	OrthoWindowClass *m_FullScreenWindow;
	//the points for the different objects
	TerrainClass* m_Terrain;
	//textures to render to
	RenderTextureClass *m_RenderFullSizeTexture, *m_FullSizeTexure, *m_DownSampleHalfSizeTexure, *m_HalfSizeTexture;
	RenderTextureClass *m_FrontTexture, *m_BackTexure;
	//the different shaders used
	TextureShaderClass* m_TextureShader;
	TextureToTextureShaderClass* m_TextureToTextureShader;
	FontShaderClass* m_FontShader;
	TerrainShaderClass* m_TerrainShader;
	VolumeShader* mVolumeShader;
	FaceShader* mFaceShader;
	PositionShader* mPositionShader;
	CloudClass* mCloud;

	texture_2d g_texture_2d;
	fluid_texture_3d g_texture_cloud;
	bool RenderClouds();
};

#endif