#pragma once
#include "shader.h"
class VolumeShader : public ShaderClass
{
public:
	VolumeShader(void);
	~VolumeShader(void);
	bool Initialize(ID3D11Device*, HWND);
	bool Render(ID3D11DeviceContext*, int, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, ID3D11ShaderResourceView*, ID3D11ShaderResourceView*, ID3D11ShaderResourceView*, float);
	ID3D11Buffer* GetBuffer(){return matrix_buffer_;}
private:
	struct VolumeBufferType{
		D3DXVECTOR4 scale;
		D3DXVECTOR3 StepSize;
		float Iterations;
	};
private:
	bool InitializeShader(ID3D11Device*, HWND, WCHAR*, WCHAR*);
	bool SetShaderParameters(ID3D11DeviceContext*, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, ID3D11ShaderResourceView*, ID3D11ShaderResourceView*, ID3D11ShaderResourceView*, float);
	void RenderShader(ID3D11DeviceContext*, int);
private:
	ID3D11Buffer* volume_buffer_;
};