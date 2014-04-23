////////////////////////////////////////////////////////////////////////////////
// Filename: positionclass.h
////////////////////////////////////////////////////////////////////////////////
#ifndef _POSITIONCLASS_H_
#define _POSITIONCLASS_H_
//////////////
// INCLUDES //
//////////////
#include <math.h>
////////////////////////////////////////////////////////////////////////////////
// Class name: PositionClass
////////////////////////////////////////////////////////////////////////////////
class PositionClass
{
public:
	PositionClass();
	PositionClass(const PositionClass&);
	~PositionClass();
	void SetPosition(float, float, float);
	void SetRotation(float, float, float);
	void GetPosition(float&, float&, float&);
	void GetRotation(float&, float&, float&);
	void SetFrameTime(float);
	void MoveForward(bool);
	void MoveBackward(bool);
	void MoveUpward(bool);
	void MoveDownward(bool);
	void TurnLeft(bool);
	void TurnRight(bool);
	void LookUpward(bool);
	void LookDownward(bool);
private:
	float position_x_, position_y_, position_z_;
	float rotation_x_, rotation_y_, rotation_z_;
	float frame_time_;
	float forward_speed_, backward_speed_;
	float upward_speed_, downward_speed_;
	float left_turn_speed_, right_turn_speed_;
	float look_up_speed_, look_down_speed_;
};
#endif