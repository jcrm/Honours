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
#include "mergetextureshaderclass.h"
#include "ShaderClass.h"
#include "SimpleShader.h"
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

#define MODEL_NUMBER 20
// testing/tracing function used pervasively in tests.  if the condition is unsatisfied
// then spew and fail the function immediately (doing no cleanup)
#define AssertOrQuit(x) \
    if (!(x)) \
    { \
        fprintf(stdout, "Assert unsatisfied in %s at %s:%d\n", __FUNCTION__, __FILE__, __LINE__); \
        return 1; \
    }
// The CUDA kernel launchers that get called
extern "C"
{
    bool cuda_texture_2d(void *surface, size_t width, size_t height, size_t pitch, float t);
    bool cuda_texture_3d(void *surface, int width, int height, int depth, size_t pitch, size_t pitchslice, float t);
    bool cuda_texture_cube(void *surface, int width, int height, size_t pitch, int face, float t);
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
	//-----------------------------------------------------------------------------
	// Global variables
	//-----------------------------------------------------------------------------

	ID3D11InputLayout      *g_pInputLayout;
	struct ConstantBuffer
	{
		float   vQuadRect[4];
		int     UseCase;
	};

	ID3D11VertexShader  *g_pVertexShader;
	ID3D11PixelShader   *g_pPixelShader;
	ID3D11Buffer        *g_pConstantBuffer;
	ID3D11SamplerState  *g_pSamplerState;
	bool g_bDone;
	bool g_bPassed;

	int *pArgc;
	char **pArgv;

	unsigned int g_WindowWidth;
	unsigned int g_WindowHeight;

	int g_iFrameToCompare;

	// Data structure for 2D texture shared between DX10 and CUDA
	struct{
		ID3D11Texture2D         *pTexture;
		ID3D11ShaderResourceView *pSRView;
		cudaGraphicsResource    *cudaResource;
		void                    *cudaLinearMemory;
		size_t                  pitch;
		int                     width;
		int                     height;
	} g_texture_2d;

	// Data structure for volume textures shared between DX10 and CUDA
	struct{
		ID3D11Texture3D         *pTexture;
		ID3D11ShaderResourceView *pSRView;
		cudaGraphicsResource    *cudaResource;
		void                    *cudaLinearMemory;
		size_t                  pitch;
		int                     width;
		int                     height;
		int                     depth;
	}g_texture_3d;

	// Data structure for cube texture shared between DX10 and CUDA
	struct{
		ID3D11Texture2D         *pTexture;
		ID3D11ShaderResourceView *pSRView;
		cudaGraphicsResource    *cudaResource;
		void                    *cudaLinearMemory;
		size_t                  pitch;
		int                     size;
	} g_texture_cube;
	
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
	RenderTextureClass *m_RenderFullSizeTexture, *m_FullSizeTexure, *m_DownSampleHalfSizeTexure, *m_HalfSizeTexture, *m_MergeFullSizeTexture;
	//the different shaders used
	TextureShaderClass* m_TextureShader;
	TextureToTextureShaderClass* m_TextureToTextureShader;
	FontShaderClass* m_FontShader;
	TerrainShaderClass* m_TerrainShader;
	MergeTextureShaderClass* mMergerShader;
	SimpleShader* mSimpleShader;

	HRESULT InitTextures();
};

#endif