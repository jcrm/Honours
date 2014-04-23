////////////////////////////////////////////////////////////////////////////////
// Filename: rendertextureclass.h
////////////////////////////////////////////////////////////////////////////////
#ifndef _RENDERTEXTURECLASS_H_
#define _RENDERTEXTURECLASS_H_
//////////////
// INCLUDES //
//////////////
#include <d3d11.h>
#include <d3dx10math.h>
////////////////////////////////////////////////////////////////////////////////
// Class name: RenderTextureClass
////////////////////////////////////////////////////////////////////////////////
class RenderTextureClass
{
public:
	RenderTextureClass();
	RenderTextureClass(const RenderTextureClass&);
	~RenderTextureClass();
	bool Initialize(ID3D11Device*, int, int, float, float);
	void Shutdown();
	void SetRenderTarget(ID3D11DeviceContext*);
	void ClearRenderTarget(ID3D11DeviceContext*, float, float, float, float);
	ID3D11ShaderResourceView* GetShaderResourceView();
	void GetProjectionMatrix(D3DXMATRIX&);
	void GetOrthoMatrix(D3DXMATRIX&);
	int GetTextureWidth();
	int GetTextureHeight();
private:
	int texture_width_, texture_height_;
	ID3D11Texture2D* render_target_texture_;
	ID3D11RenderTargetView* render_target_view_;
	ID3D11ShaderResourceView* shader_resource_view_;
	ID3D11Texture2D* depth_stencil_buffer_;
	ID3D11DepthStencilView* depth_stencil_view_;
	D3D11_VIEWPORT viewport_;
	D3DXMATRIX projection_matrix_;
	D3DXMATRIX ortho_matrix_;
};
#endif