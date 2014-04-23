////////////////////////////////////////////////////////////////////////////////
// Filename: cameraclass.cpp
////////////////////////////////////////////////////////////////////////////////
#include "camera.h"
CameraClass::CameraClass(){
	position_x_ = 0.0f;
	position_y_ = 0.0f;
	position_z_ = 0.0f;
	rotation_x_ = 0.0f;
	rotation_y_ = 0.0f;
	rotation_z_ = 0.0f;
}
CameraClass::CameraClass(const CameraClass& other){
}
CameraClass::~CameraClass(){
}
void CameraClass::SetPosition(float x, float y, float z){
	position_x_ = x;
	position_y_ = y;
	position_z_ = z;
	return;
}
void CameraClass::SetRotation(float x, float y, float z){
	rotation_x_ = x;
	rotation_y_ = y;
	rotation_z_ = z;
	return;
}
void CameraClass::Render(){
	D3DXVECTOR3 up, position, lookAt;
	float yaw, pitch, roll;
	D3DXMATRIX rotationMatrix;
	// Setup the vector that points upwards.
	up.x = 0.0f;
	up.y = 1.0f;
	up.z = 0.0f;
	// Setup the position of the camera in the world.
	position.x = position_x_;
	position.y = position_y_;
	position.z = position_z_;
	// Setup where the camera is looking by default.
	lookAt.x = 0.0f;
	lookAt.y = 0.0f;
	lookAt.z = 1.0f;
	// Set the yaw (Y axis), pitch (X axis), and roll (Z axis) rotations in radians.
	pitch = rotation_x_ * 0.0174532925f;
	yaw   = rotation_y_ * 0.0174532925f;
	roll  = rotation_z_ * 0.0174532925f;
	// Create the rotation matrix from the yaw, pitch, and roll values.
	D3DXMatrixRotationYawPitchRoll(&rotationMatrix, yaw, pitch, roll);
	// Transform the lookAt and up vector by the rotation matrix so the view is correctly rotated at the origin.
	D3DXVec3TransformCoord(&lookAt, &lookAt, &rotationMatrix);
	D3DXVec3TransformCoord(&up, &up, &rotationMatrix);
	// Translate the rotated camera position to the location of the viewer.
	lookAt = position + lookAt;
	// Finally create the view matrix from the three updated vectors.
	D3DXMatrixLookAtLH(&view_matrix_, &position, &lookAt, &up);
	return;
}
void CameraClass::GetViewMatrix(D3DXMATRIX& viewMatrix){
	viewMatrix = view_matrix_;
	return;
}