#ifndef _ShaderClass_h_
#define _ShaderClass_h_
//////////////
// INCLUDES //
//////////////
#include <d3d11.h>
#include <d3dx10math.h>
#include <d3dx11async.h>
#include <fstream>
using namespace std;
////////////////////////////////////////////////////////////////////////////////
// Class name: ShaderClass virtual class that gets inherited by most shader classes
////////////////////////////////////////////////////////////////////////////////
class ShaderClass
{
protected:
	struct MatrixBufferType{
		D3DXMATRIX world_;
		D3DXMATRIX view_;
		D3DXMATRIX projection_;
	};
public:
	ShaderClass(void);
	ShaderClass(const ShaderClass&);
	virtual ~ShaderClass(void);
	virtual bool Initialize(ID3D11Device*, HWND);
	virtual void Shutdown();
	virtual bool Render(ID3D11DeviceContext*, int, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, ID3D11ShaderResourceView*, float, float);
	virtual	bool Render(ID3D11DeviceContext*, int, D3DXMATRIX, ID3D11ShaderResourceView*, float, float);
protected:
	virtual bool InitializeShader(ID3D11Device*, HWND, WCHAR*, WCHAR*);
	virtual void ShutdownShader();
	virtual void OutputShaderErrorMessage(ID3D10Blob*, HWND, WCHAR*);
	virtual bool SetShaderParameters(ID3D11DeviceContext*, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, ID3D11ShaderResourceView*);
	virtual void RenderShader(ID3D11DeviceContext*, int);
protected:
	ID3D11VertexShader* vertex_shader_;
	ID3D11PixelShader* pixel_shader_;
	ID3D11InputLayout* layout_;
	ID3D11Buffer* matrix_buffer_;
	ID3D11SamplerState* sample_state_;
};
#endif // _ShaderClass_h_