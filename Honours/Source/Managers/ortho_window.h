////////////////////////////////////////////////////////////////////////////////
// Filename: orthowindowclass.h
////////////////////////////////////////////////////////////////////////////////
#ifndef _ORTHOWINDOWCLASS_H_
#define _ORTHOWINDOWCLASS_H_
//////////////
// INCLUDES //
//////////////
#include <d3d11.h>
#include <d3dx10math.h>
////////////////////////////////////////////////////////////////////////////////
// Class name: OrthoWindowClass
////////////////////////////////////////////////////////////////////////////////
class OrthoWindowClass{
private:
	struct VertexType{
		D3DXVECTOR3 position_;
	    D3DXVECTOR2 texture;
	};
public:
	OrthoWindowClass();
	OrthoWindowClass(const OrthoWindowClass&);
	~OrthoWindowClass();
	bool Initialize(ID3D11Device*, int, int);
	void Shutdown();
	void Render(ID3D11DeviceContext*);
	inline int GetIndexCount(){return index_count_;}
private:
	bool InitializeBuffers(ID3D11Device*, int, int);
	void ShutdownBuffers();
	void RenderBuffers(ID3D11DeviceContext*);
private:
	ID3D11Buffer *vertex_buffer_, *index_buffer_;
	int vertex_count_, index_count_;
};
#endif