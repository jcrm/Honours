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
#include "inputclass.h"
#include "d3dclass.h"
#include "cameraclass.h"
#include "terrainclass.h"
#include "timerclass.h"
#include "positionclass.h"
#include "fpsclass.h"
#include "cpuclass.h"
#include "fontshaderclass.h"
#include "textclass.h"
#include "terrainshaderclass.h"
#include "lightclass.h"
#include "textureshaderclass.h"
#include "rendertextureclass.h"
#include "orthowindowclass.h"
#include "texturetotextureshaderclass .h"
#include "ShaderClass.h"
#include "VolumeShader.h"
#include "FaceShader.h"
#include "cudad3d.h"

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
#include "cuda_structs.h"
#include "cuda_kernals.h"

#define MODEL_NUMBER 20
// testing/tracing function used pervasively in tests.  if the condition is unsatisfied
// then spew and fail the function immediately (doing no cleanup)
#define AssertOrQuit(x) \
    if (!(x)) \
    { \
        fprintf(stdout, "Assert unsatisfied in %s at %s:%d\n", __FUNCTION__, __FILE__, __LINE__); \
        return 1; \
    }

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
	bool RenderMergeTexture(RenderTextureClass *readTexture, RenderTextureClass *readTexture2, RenderTextureClass *writeTexture, OrthoWindowClass *window);
	bool Render2DTextureScene(RenderTextureClass* mRead);
	//Init functions
	bool InitObjects(HWND hwnd);
	bool InitTextures(HWND hwnd, int screenWidth, int screenHeight);
	bool InitText(HWND hwnd, int screenWidth , int screenHeight);
	bool InitShaders(HWND hwnd);
	bool InitCamera();
	//Shutdown functions
	void ShutdownObjects();
	void ShutdownText();
	void ShutdownTextures();
	void ShutdownCamera();
	void ShutdownShaders();

	HRESULT InitCudaTextures();
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

	texture_2d g_texture_2d;
	fluid_texture_3d g_texture_cloud;
};

#endif