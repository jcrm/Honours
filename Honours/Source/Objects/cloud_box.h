////////////////////////////////////////////////////////////////////////////////
// Filename: terrainclass.h
////////////////////////////////////////////////////////////////////////////////
#ifndef _CLOUDCLASS_H_
#define _CLOUDCLASS_H_
//////////////
// INCLUDES //
//////////////
#include <d3d11.h>
#include <d3dx10math.h>
#include <stdio.h>
#include "../Textures/render_texture.h"
////////////////////////////////////////////////////////////////////////////////
// Class name: TerrainClass
////////////////////////////////////////////////////////////////////////////////
class CloudClass{
private:
	struct VertexType{
		D3DXVECTOR3 position_;
		D3DXVECTOR3 texture_;
	};
	struct VectorType { 
		float x, y, z;
	};
public:
	CloudClass();
	CloudClass(const CloudClass&);
	~CloudClass();
	bool Initialize(ID3D11Device*, int, int, float, float);
	void Shutdown();
	void Render(ID3D11DeviceContext*);
	//getters
	inline int  GetIndexCount() const {return index_count_;}
	inline RenderTextureClass* GetFrontTexture(){return front_texture_;}
	inline RenderTextureClass* GetBackTexture(){return back_texture_;}
	
	inline ID3D11ShaderResourceView* GetFrontShaderResource(){return front_texture_->GetShaderResourceView();}
	inline ID3D11ShaderResourceView* GetBackShaderResource(){return back_texture_->GetShaderResourceView();}
	inline float GetScale(){return 1/size_;}
	inline D3DXMATRIX GetTranslation() {return transform_;}
private:
	//buffer functions
	bool InitializeBuffers(ID3D11Device*);
	void ShutdownBuffers();
	void RenderBuffers(ID3D11DeviceContext*);
	//texture functions 
	void ReleaseTexture();
private:
	int vertex_count_, index_count_;
	ID3D11Buffer *vertex_buffer_, *index_buffer_;
	RenderTextureClass *front_texture_, *back_texture_;
	D3DXMATRIX transform_;
	float size_;
};
#endif