#pragma once
#include "shaderclass.h"
class PositionShader : public ShaderClass
{
public:
	PositionShader(void);
	~PositionShader(void);
	bool Initialize(ID3D11Device*, HWND);
	bool Render(ID3D11DeviceContext*, int, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX);
	ID3D11Buffer* GetBuffer(){return m_matrixBuffer;}
protected:
	bool InitializeShader(ID3D11Device*, HWND, WCHAR*, WCHAR*);
	bool SetShaderParameters(ID3D11DeviceContext*, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX);
	void RenderShader(ID3D11DeviceContext*, int);
};

