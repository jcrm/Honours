////////////////////////////////////////////////////////////////////////////////
// Filename: fontclass.h
////////////////////////////////////////////////////////////////////////////////
#ifndef _FONTCLASS_H_
#define _FONTCLASS_H_
//////////////
// INCLUDES //
//////////////
#include <d3d11.h>
#include <d3dx10math.h>
#include <fstream>
using namespace std;
///////////////////////
// MY CLASS INCLUDES //
///////////////////////
#include "../Textures/texture.h"
////////////////////////////////////////////////////////////////////////////////
// Class name: FontClass
////////////////////////////////////////////////////////////////////////////////
class FontClass
{
private:
	struct FontType
	{
		float left_, right_;
		int size_;
	};
	struct VertexType
	{
		D3DXVECTOR3 position_;
	    D3DXVECTOR2 texture_;
	};
public:
	FontClass();
	FontClass(const FontClass&);
	~FontClass();
	bool Initialize(ID3D11Device*, char*, WCHAR*);
	void Shutdown();
	ID3D11ShaderResourceView* GetTexture();
	void BuildVertexArray(void*, char*, float, float);
private:
	bool LoadFontData(char*);
	void ReleaseFontData();
	bool LoadTexture(ID3D11Device*, WCHAR*);
	void ReleaseTexture();
private:
	FontType* font_;
	TextureClass* texture_;
};
#endif