////////////////////////////////////////////////////////////////////////////////
// Filename: cameraclass.h
////////////////////////////////////////////////////////////////////////////////
#ifndef _CAMERACLASS_H_
#define _CAMERACLASS_H_
//////////////
// INCLUDES //
//////////////
#include <d3dx10math.h>
////////////////////////////////////////////////////////////////////////////////
// Class name: CameraClass
////////////////////////////////////////////////////////////////////////////////
class CameraClass
{
public:
	CameraClass();
	CameraClass(const CameraClass&);
	~CameraClass();
	void SetPosition(float, float, float);
	void SetRotation(float, float, float);
	inline D3DXVECTOR3 GetPosition(){return D3DXVECTOR3(position_x_, position_y_, position_z_);}
	inline D3DXVECTOR3 GetRotation(){return D3DXVECTOR3(rotation_x_, rotation_y_, rotation_z_);}
	void Render();
	void GetViewMatrix(D3DXMATRIX&);
private:
	float position_x_, position_y_, position_z_;
	float rotation_x_, rotation_y_, rotation_z_;
	D3DXMATRIX view_matrix_;
};
#endif