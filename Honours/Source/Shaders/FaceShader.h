#pragma once
#include "shaderclass.h"
class FaceShader : public ShaderClass
{
public:
	FaceShader(void);
	~FaceShader(void);
	bool Initialize(ID3D11Device*, HWND);
	bool Render(ID3D11DeviceContext*, int, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, ID3D11ShaderResourceView*);
	ID3D11Buffer* GetBuffer(){return m_matrixBuffer;}
protected:
	bool InitializeShader(ID3D11Device*, HWND, WCHAR*, WCHAR*);

	bool SetShaderParameters(ID3D11DeviceContext*, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, ID3D11ShaderResourceView*);
	void RenderShader(ID3D11DeviceContext*, int);

	ID3D11PixelShader* m_pixelShaderPosition;
};

