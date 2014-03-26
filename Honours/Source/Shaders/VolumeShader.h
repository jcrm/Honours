#pragma once
#include "shaderclass.h"
class VolumeShader : public ShaderClass
{
public:
	VolumeShader(void);
	~VolumeShader(void);
	bool Initialize(ID3D11Device*, HWND);
	bool Render(ID3D11DeviceContext*, int, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, ID3D11ShaderResourceView*);
	ID3D11Buffer* GetBuffer(){return m_matrixBuffer;}
protected:
	bool InitializeShader(ID3D11Device*, HWND, WCHAR*, WCHAR*);

	bool SetShaderParameters(ID3D11DeviceContext*, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, ID3D11ShaderResourceView*);
	bool SetShaderParameters(ID3D11DeviceContext* deviceContext, D3DXMATRIX projectionMatrix, ID3D11ShaderResourceView* texture);
	bool SetShaderParameters(ID3D11DeviceContext* deviceContext, D3DXMATRIX projectionMatrix, ID3D11ShaderResourceView* texture, float screenHeight, float screenWidth);
	void RenderShader(ID3D11DeviceContext*, int);
};

