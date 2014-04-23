////////////////////////////////////////////////////////////////////////////////
// Filename: texturetotextureshaderclass.h
////////////////////////////////////////////////////////////////////////////////
#ifndef _TEXTURETOTEXTURESHADERCLASS_H_
#define _TEXTURETOTEXTURESHADERCLASS_H_
//////////////
// INCLUDES //
//////////////
#include <d3d11.h>
#include <d3dx10math.h>
#include <d3dx11async.h>
#include <fstream>
#include "shader.h"
using namespace std;
////////////////////////////////////////////////////////////////////////////////
// Class name: TextureToTextureShaderClass
////////////////////////////////////////////////////////////////////////////////
class TextureToTextureShaderClass : public ShaderClass
{
public:
	TextureToTextureShaderClass();
	TextureToTextureShaderClass(const TextureToTextureShaderClass&);
	~TextureToTextureShaderClass();
	bool Initialize(ID3D11Device*, HWND);
	bool Render(ID3D11DeviceContext*, int, D3DXMATRIX, ID3D11ShaderResourceView*, float, float);
	bool Render(ID3D11DeviceContext*, int, D3DXMATRIX, ID3D11ShaderResourceView*);
protected:
	struct MatrixBufferType2
	{
		D3DXMATRIX projection_;
	};
	bool InitializeShader(ID3D11Device*, HWND, WCHAR*, WCHAR*);
	bool SetShaderParameters(ID3D11DeviceContext*, D3DXMATRIX, ID3D11ShaderResourceView*);
	bool SetShaderParameters(ID3D11DeviceContext*, D3DXMATRIX, ID3D11ShaderResourceView*, float, float);
	void RenderShader(ID3D11DeviceContext*, int);
};
#endif