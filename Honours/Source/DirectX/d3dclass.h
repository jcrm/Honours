////////////////////////////////////////////////////////////////////////////////
// Filename: d3dclass.h
////////////////////////////////////////////////////////////////////////////////
#ifndef _D3DCLASS_H_
#define _D3DCLASS_H_
/////////////
// LINKING //
/////////////
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dx11.lib")
#pragma comment(lib, "d3dx10.lib")
//////////////
// INCLUDES //
//////////////
#include <dxgi.h>
#include <d3dcommon.h>
#include <d3d11.h>
#include <d3dx10math.h>
////////////////////////////////////////////////////////////////////////////////
// Class name: D3DClass
////////////////////////////////////////////////////////////////////////////////
class D3DClass{
public:
	D3DClass();
	D3DClass(const D3DClass&);
	~D3DClass();
	virtual bool Initialize(int, int, bool, HWND, bool, float, float);
	void Shutdown();
	
	void BeginScene(float, float, float, float);
	void EndScene();
	inline ID3D11Device* GetDevice(){return device_;}
	inline ID3D11DeviceContext* GetDeviceContext(){return device_context_;}
	void GetProjectionMatrix(D3DXMATRIX&);
	void GetWorldMatrix(D3DXMATRIX&);
	void GetOrthoMatrix(D3DXMATRIX&);
	void GetVideoCardInfo(char*, int&);
	void TurnZBufferOn();
	void TurnZBufferOff();
	void TurnOnAlphaBlending();
	void TurnOffAlphaBlending();
	void SetBackBufferRenderTarget();
	void ResetViewport();
	bool CreateRaster();
	bool CreateBackFaceRaster();
	void EnableAlphaBlending();
	void DisableAlphaBlending();
protected:
	virtual bool InitDisplayMode(int, int,unsigned int&, unsigned int&);
	virtual bool InitSwapChain(HWND, int, int, unsigned int&, unsigned int&, bool);
	bool InitDepthBuffer(int, int);
	bool InitDepthStencil();
	bool InitDepthDisableStencil();
	bool InitBlendState();
protected:
	bool vsync_enabled_;
	int video_card_memory_;
	char video_card_description_[128];
	IDXGISwapChain* swap_chain_;
	ID3D11Device* device_;
	ID3D11DeviceContext* device_context_;
	ID3D11RenderTargetView* render_target_view_;
	ID3D11Texture2D* depth_stencil_buffer_;
	ID3D11DepthStencilState* depth_stencil_state_;
	ID3D11DepthStencilView* depth_stencil_view_;
	ID3D11RasterizerState* raster_state_;
	D3DXMATRIX projection_matrix_;
	D3DXMATRIX world_matrix_;
	D3DXMATRIX ortho_matrix_;
	ID3D11DepthStencilState* depth_disabled_stencil_state_;
	ID3D11BlendState* alpha_enable_blending_state_;
	ID3D11BlendState* alpha_disable_blending_state_;
	ID3D11BlendState* alpha_enable_additional_blending_state_;
	ID3D11BlendState* alpha_disable_additional_blending_state_;
	D3D11_VIEWPORT viewport_;
};
#endif