#ifndef _FACE_SHADER_
#define _FACE_SHADER_

#include "shader.h"
class FaceShader : public ShaderClass
{
private:
	struct ScaleBufferType{
		D3DXVECTOR4 scale_;
	};
public:
	FaceShader(void);
	~FaceShader(void);
	bool Initialize(ID3D11Device*, HWND);
	bool Render(ID3D11DeviceContext*, int, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, D3DXVECTOR3);
	ID3D11Buffer* GetBuffer(){return matrix_buffer_;}
protected:
	bool InitializeShader(ID3D11Device*, HWND, WCHAR*, WCHAR*);
	bool SetShaderParameters(ID3D11DeviceContext*, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, D3DXVECTOR3);
	void RenderShader(ID3D11DeviceContext*, int);
	ID3D11Buffer* scale_buffer_;
};
#endif