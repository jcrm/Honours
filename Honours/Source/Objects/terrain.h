////////////////////////////////////////////////////////////////////////////////
// Filename: terrainclass.h
////////////////////////////////////////////////////////////////////////////////
#ifndef _TERRAINCLASS_H_
#define _TERRAINCLASS_H_
//////////////
// INCLUDES //
//////////////
#include <d3d11.h>
#include <d3dx10math.h>
#include <stdio.h>
///////////////////////
// MY CLASS INCLUDES //
///////////////////////
#include "../Textures/texture.h"
/////////////
// GLOBALS //
/////////////
const int TEXTURE_REPEAT = 8;
////////////////////////////////////////////////////////////////////////////////
// Class name: TerrainClass
////////////////////////////////////////////////////////////////////////////////
class TerrainClass
{
private:
	struct VertexType{
		D3DXVECTOR3 position_;
		D3DXVECTOR2 texture_;
	    D3DXVECTOR3 normal_;
	};
	struct HeightMapType{ 
		float x, y, z;
		float tu, tv;
		float nx, ny, nz;
	};
	struct VectorType{ 
		float x, y, z;
	};
public:
	TerrainClass();
	TerrainClass(const TerrainClass&);
	~TerrainClass();
	bool Initialize(ID3D11Device*, char*, WCHAR*);
	void Shutdown();
	void Render(ID3D11DeviceContext*);
	int GetIndexCount();
	ID3D11ShaderResourceView* GetTexture();
private:
	bool LoadHeightMap(char*);
	void NormalizeHeightMap();
	bool CalculateNormals();
	void ShutdownHeightMap();
	void CalculateTextureCoordinates();
	bool LoadTexture(ID3D11Device*, WCHAR*);
	void ReleaseTexture();
	bool InitializeBuffers(ID3D11Device*);
	void ShutdownBuffers();
	void RenderBuffers(ID3D11DeviceContext*);
	
private:
	int terrain_width_, terrain_height_;
	int vertex_count_, index_count_;
	ID3D11Buffer *vertex_buffer_, *index_buffer_;
	HeightMapType* height_map_;
	TextureClass* texture_;
};
#endif