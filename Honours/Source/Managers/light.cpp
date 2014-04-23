////////////////////////////////////////////////////////////////////////////////
// Filename: lightclass.cpp
////////////////////////////////////////////////////////////////////////////////
#include "light.h"
LightClass::LightClass(){
}
LightClass::LightClass(const LightClass& other){
}
LightClass::~LightClass(){
}
void LightClass::SetAmbientColor(float red, float green, float blue, float alpha){
	ambient_color_ = D3DXVECTOR4(red, green, blue, alpha);
	return;
}
void LightClass::SetDiffuseColor(float red, float green, float blue, float alpha){
	diffuse_color_ = D3DXVECTOR4(red, green, blue, alpha);
	return;
}
void LightClass::SetDirection(float x, float y, float z){
	direction_ = D3DXVECTOR3(x, y, z);
	return;
}
D3DXVECTOR4 LightClass::GetAmbientColor(){
	return ambient_color_;
}
D3DXVECTOR4 LightClass::GetDiffuseColor(){
	return diffuse_color_;
}
D3DXVECTOR3 LightClass::GetDirection(){
	return direction_;
}