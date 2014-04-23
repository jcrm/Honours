////////////////////////////////////////////////////////////////////////////////
// Filename: mergetextureshaderclass.h
////////////////////////////////////////////////////////////////////////////////
#ifndef _MERGETEXTURESHADERCLASS_H_
#define _MERGETEXTURESHADERCLASS_H_


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
// Class name: MergeTextureShaderClass
////////////////////////////////////////////////////////////////////////////////
class MergeTextureShaderClass : public ShaderClass
{
public:
	struct MergeBufferType
	{
		float strength_;
		D3DXVECTOR3 padding_;
	};
	MergeTextureShaderClass();
	MergeTextureShaderClass(const MergeTextureShaderClass&);
	~MergeTextureShaderClass();

	bool Initialize(ID3D11Device*, HWND);
	void Shutdown();
	//takes two texture and merges them to make a third
	bool Render(ID3D11DeviceContext*, int, ID3D11ShaderResourceView*, ID3D11ShaderResourceView*);
protected:
	bool InitializeShader(ID3D11Device*, HWND, WCHAR*, WCHAR*);
	void ShutdownShader();
	bool SetShaderParameters(ID3D11DeviceContext*, ID3D11ShaderResourceView*, ID3D11ShaderResourceView*);
	void RenderShader(ID3D11DeviceContext*, int);
	ID3D11Buffer* merge_buffer_;
};

#endif