////////////////////////////////////////////////////////////////////////////////
// Filename: lightclass.h
////////////////////////////////////////////////////////////////////////////////
#ifndef _LIGHTCLASS_H_
#define _LIGHTCLASS_H_
//////////////
// INCLUDES //
//////////////
#include <d3dx10math.h>
////////////////////////////////////////////////////////////////////////////////
// Class name: LightClass
////////////////////////////////////////////////////////////////////////////////
class LightClass{
public:
	LightClass();
	LightClass(const LightClass&);
	~LightClass();
	void SetAmbientColor(float, float, float, float);
	void SetDiffuseColor(float, float, float, float);
	void SetDirection(float, float, float);
	D3DXVECTOR4 GetAmbientColor();
	D3DXVECTOR4 GetDiffuseColor();
	D3DXVECTOR3 GetDirection();
private:
	D3DXVECTOR4 ambient_color_;
	D3DXVECTOR4 diffuse_color_;
	D3DXVECTOR3 direction_;
};
#endif