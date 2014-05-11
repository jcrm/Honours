#ifndef _VOLUME_SHADER_
#define _VOLUME_SHADER_
#include "shader.h"

#define GRID_X 64
#define GRID_Y 64
#define GRID_Z 64

class VolumeShader : public ShaderClass
{
public:
	VolumeShader(void);
	~VolumeShader(void);
	bool Initialize(ID3D11Device*, HWND);
	bool Render(ID3D11DeviceContext*, int, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, ID3D11ShaderResourceView*, D3DXVECTOR4);
	ID3D11Buffer* GetBuffer(){return matrix_buffer_;}
private:
	struct ScaleBuffer{
		D3DXVECTOR3 step_size_;
		float iterations_;
	};
	struct CameraBufferType{
		D3DXVECTOR4 camera_position_;
	};
private:
	bool InitializeShader(ID3D11Device*, HWND, WCHAR*, WCHAR*);
	bool SetShaderParameters(ID3D11DeviceContext*, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, ID3D11ShaderResourceView*, D3DXVECTOR4);
	void RenderShader(ID3D11DeviceContext*, int);
private:
	ID3D11Buffer* scale_buffer_;
	ID3D11Buffer* camera_buffer_;
};
#endif