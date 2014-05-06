#pragma once
#include "shader.h"
class VolumeShaderTWO : public ShaderClass
{
public:
	VolumeShaderTWO(void);
	~VolumeShaderTWO(void);
	bool Initialize(ID3D11Device*, HWND);
	bool Render(ID3D11DeviceContext*, int, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, ID3D11ShaderResourceView* , D3DXVECTOR4);
	ID3D11Buffer* GetBuffer(){return matrix_buffer_;}
private:
	struct CameraBufferType{
		D3DXVECTOR4 CameraPositionTS;
		D3DXMATRIX Inverse;
	};
private:
	bool InitializeShader(ID3D11Device*, HWND, WCHAR*, WCHAR*);
	bool SetShaderParameters(ID3D11DeviceContext* , D3DXMATRIX , D3DXMATRIX , D3DXMATRIX, ID3D11ShaderResourceView*, D3DXVECTOR4);
	void RenderShader(ID3D11DeviceContext*, int);
private:
	ID3D11Buffer* volume_buffer_;
};