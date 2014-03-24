////////////////////////////////////////////////////////////////////////////////
// Filename: terrainclass.h
////////////////////////////////////////////////////////////////////////////////
#ifndef _TERRAINCLASS_H_
#define _CLOUDCLASS_H_

//////////////
// INCLUDES //
//////////////
#include <d3d11.h>
#include <d3dx10math.h>
#include <stdio.h>
#include "rendertextureclass.h"

#define TEXTURE_REPEAT 8
////////////////////////////////////////////////////////////////////////////////
// Class name: TerrainClass
////////////////////////////////////////////////////////////////////////////////
class CloudClass
{
private:
	struct VertexType{
		D3DXVECTOR3 position;
		D3DXVECTOR3 texture;
	};
	struct VectorType { 
		float x, y, z;
	};

public:
	CloudClass();
	CloudClass(const CloudClass&);
	~CloudClass();

	bool Initialize(ID3D11Device* device);
	void Shutdown();
	void Render(ID3D11DeviceContext*);
	//getters
	inline int  GetIndexCount() const {return m_indexCount;}
	inline RenderTextureClass* GetFrontTexture(){return m_FrontTexture;}
	inline RenderTextureClass* GetBackTexture(){return m_BackTexture;}
private:
	//buffer functions
	bool InitializeBuffers(ID3D11Device*);
	void ShutdownBuffers();
	void RenderBuffers(ID3D11DeviceContext*);
	//texture functions 
	void ReleaseTexture();
private:
	int m_vertexCount, m_indexCount;
	ID3D11Buffer *m_vertexBuffer, *m_indexBuffer;
	RenderTextureClass* m_FrontTexture, *m_BackTexture;
};

#endif