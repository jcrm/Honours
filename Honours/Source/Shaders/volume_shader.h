#ifndef _VOLUME_SHADER_H_
#define _VOLUME_SHADER_H_

#include "shader.h"
class VolumeShader : public ShaderClass
{
public:
	VolumeShader(void);
	~VolumeShader(void);
	bool Initialize(ID3D11Device*, HWND);
	bool Render(ID3D11DeviceContext*, int, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, ID3D11ShaderResourceView*, ID3D11ShaderResourceView*, ID3D11ShaderResourceView*, D3DXVECTOR3);
	ID3D11Buffer* GetBuffer(){return matrix_buffer_;}
private:
	struct VolumeBufferType{
		D3DXVECTOR4 scale_;
		D3DXVECTOR3 step_size_;
		float iterations_;
	};
private:
	bool InitializeShader(ID3D11Device*, HWND, WCHAR*, WCHAR*);
	bool SetShaderParameters(ID3D11DeviceContext*, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, ID3D11ShaderResourceView*, ID3D11ShaderResourceView*, ID3D11ShaderResourceView*, D3DXVECTOR3);
	void RenderShader(ID3D11DeviceContext*, int);
private:
	ID3D11Buffer* volume_buffer_;
};
#endif