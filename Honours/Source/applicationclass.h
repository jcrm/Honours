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

#define MODEL_NUMBER 20

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
private:
	InputClass* m_Input;
	D3DClass* m_Direct3D;
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
};

#endif