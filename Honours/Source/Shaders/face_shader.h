#pragma once
#include "shader.h"
class FaceShader : public ShaderClass
{
private:
	struct ScaleBufferType{
		D3DXVECTOR4 scale;
	};
public:
	FaceShader(void);
	~FaceShader(void);
	bool Initialize(ID3D11Device*, HWND);
	bool Render(ID3D11DeviceContext*, int, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, float);
	ID3D11Buffer* GetBuffer(){return m_matrixBuffer;}
protected:
	bool InitializeShader(ID3D11Device*, HWND, WCHAR*, WCHAR*);
	bool SetShaderParameters(ID3D11DeviceContext*, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, float);
	void RenderShader(ID3D11DeviceContext*, int);
	ID3D11Buffer* m_scaleBuffer;
};

