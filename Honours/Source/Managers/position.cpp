////////////////////////////////////////////////////////////////////////////////
// Filename: positionclass.cpp
////////////////////////////////////////////////////////////////////////////////
#include "position.h"
PositionClass::PositionClass()
{
	position_x_ = 0.0f;
	position_y_ = 0.0f;
	position_z_ = 0.0f;
	
	rotation_x_ = 0.0f;
	rotation_y_ = 0.0f;
	rotation_z_ = 0.0f;
	frame_time_ = 0.0f;
	forward_speed_   = 0.0f;
	backward_speed_  = 0.0f;
	upward_speed_    = 0.0f;
	downward_speed_  = 0.0f;
	left_turn_speed_  = 0.0f;
	right_turn_speed_ = 0.0f;
	look_up_speed_    = 0.0f;
	look_down_speed_  = 0.0f;
}
PositionClass::PositionClass(const PositionClass& other)
{
}
PositionClass::~PositionClass()
{
}
void PositionClass::SetPosition(float x, float y, float z)
{
	position_x_ = x;
	position_y_ = y;
	position_z_ = z;
	return;
}
void PositionClass::SetRotation(float x, float y, float z)
{
	rotation_x_ = x;
	rotation_y_ = y;
	rotation_z_ = z;
	return;
}
void PositionClass::GetPosition(float& x, float& y, float& z)
{
	x = position_x_;
	y = position_y_;
	z = position_z_;
	return;
}
void PositionClass::GetRotation(float& x, float& y, float& z)
{
	x = rotation_x_;
	y = rotation_y_;
	z = rotation_z_;
	return;
}
void PositionClass::SetFrameTime(float time)
{
	frame_time_ = time;
	return;
}
void PositionClass::MoveForward(bool key_down)
{
	float radians;
	// Update the forward speed movement based on the frame time and whether the user is holding the key down or not.
	if(key_down)
	{
		forward_speed_ += frame_time_ * 0.001f;
		if(forward_speed_ > (frame_time_ * 0.03f))
		{
			forward_speed_ = frame_time_ * 0.03f;
		}
	}
	else
	{
		forward_speed_ -= frame_time_ * 0.0007f;
		if(forward_speed_ < 0.0f)
		{
			forward_speed_ = 0.0f;
		}
	}
	// Convert degrees to radians.
	radians = rotation_y_ * 0.0174532925f;
	// Update the position.
	position_x_ += sinf(radians) * forward_speed_;
	position_z_ += cosf(radians) * forward_speed_;
	return;
}
void PositionClass::MoveBackward(bool key_down)
{
	float radians;
	// Update the backward speed movement based on the frame time and whether the user is holding the key down or not.
	if(key_down)
	{
		backward_speed_ += frame_time_ * 0.001f;
		if(backward_speed_ > (frame_time_ * 0.03f))
		{
			backward_speed_ = frame_time_ * 0.03f;
		}
	}
	else
	{
		backward_speed_ -= frame_time_ * 0.0007f;
		
		if(backward_speed_ < 0.0f)
		{
			backward_speed_ = 0.0f;
		}
	}
	// Convert degrees to radians.
	radians = rotation_y_ * 0.0174532925f;
	// Update the position.
	position_x_ -= sinf(radians) * backward_speed_;
	position_z_ -= cosf(radians) * backward_speed_;
	return;
}
void PositionClass::MoveUpward(bool key_down)
{
	// Update the upward speed movement based on the frame time and whether the user is holding the key down or not.
	if(key_down)
	{
		upward_speed_ += frame_time_ * 0.003f;
		if(upward_speed_ > (frame_time_ * 0.03f))
		{
			upward_speed_ = frame_time_ * 0.03f;
		}
	}
	else
	{
		upward_speed_ -= frame_time_ * 0.002f;
		if(upward_speed_ < 0.0f)
		{
			upward_speed_ = 0.0f;
		}
	}
	// Update the height position.
	position_y_ += upward_speed_;
	return;
}
void PositionClass::MoveDownward(bool key_down)
{
	// Update the downward speed movement based on the frame time and whether the user is holding the key down or not.
	if(key_down)
	{
		downward_speed_ += frame_time_ * 0.003f;
		if(downward_speed_ > (frame_time_ * 0.03f))
		{
			downward_speed_ = frame_time_ * 0.03f;
		}
	}
	else
	{
		downward_speed_ -= frame_time_ * 0.002f;
		if(downward_speed_ < 0.0f)
		{
			downward_speed_ = 0.0f;
		}
	}
	// Update the height position.
	position_y_ -= downward_speed_;
	return;
}
void PositionClass::TurnLeft(bool key_down)
{
	// Update the left turn speed movement based on the frame time and whether the user is holding the key down or not.
	if(key_down)
	{
		left_turn_speed_ += frame_time_ * 0.01f;
		if(left_turn_speed_ > (frame_time_ * 0.15f))
		{
			left_turn_speed_ = frame_time_ * 0.15f;
		}
	}
	else
	{
		left_turn_speed_ -= frame_time_* 0.005f;
		if(left_turn_speed_ < 0.0f)
		{
			left_turn_speed_ = 0.0f;
		}
	}
	// Update the rotation.
	rotation_y_ -= left_turn_speed_;
	// Keep the rotation in the 0 to 360 range.
	if(rotation_y_ < 0.0f)
	{
		rotation_y_ += 360.0f;
	}
	return;
}
void PositionClass::TurnRight(bool key_down)
{
	// Update the right turn speed movement based on the frame time and whether the user is holding the key down or not.
	if(key_down)
	{
		right_turn_speed_ += frame_time_ * 0.01f;
		if(right_turn_speed_ > (frame_time_ * 0.15f))
		{
			right_turn_speed_ = frame_time_ * 0.15f;
		}
	}
	else
	{
		right_turn_speed_ -= frame_time_* 0.005f;
		if(right_turn_speed_ < 0.0f)
		{
			right_turn_speed_ = 0.0f;
		}
	}
	// Update the rotation.
	rotation_y_ += right_turn_speed_;
	// Keep the rotation in the 0 to 360 range.
	if(rotation_y_ > 360.0f)
	{
		rotation_y_ -= 360.0f;
	}
	return;
}
void PositionClass::LookUpward(bool key_down)
{
	// Update the upward rotation speed movement based on the frame time and whether the user is holding the key down or not.
	if(key_down)
	{
		look_up_speed_ += frame_time_ * 0.01f;
		if(look_up_speed_ > (frame_time_ * 0.15f))
		{
			look_up_speed_ = frame_time_ * 0.15f;
		}
	}
	else
	{
		look_up_speed_ -= frame_time_* 0.005f;
		if(look_up_speed_ < 0.0f)
		{
			look_up_speed_ = 0.0f;
		}
	}
	// Update the rotation.
	rotation_x_ -= look_up_speed_;
	// Keep the rotation maximum 90 degrees.
	if(rotation_x_ > 90.0f)
	{
		rotation_x_ = 90.0f;
	}
	return;
}
void PositionClass::LookDownward(bool key_down)
{
	// Update the downward rotation speed movement based on the frame time and whether the user is holding the key down or not.
	if(key_down)
	{
		look_down_speed_ += frame_time_ * 0.01f;
		if(look_down_speed_ > (frame_time_ * 0.15f))
		{
			look_down_speed_ = frame_time_ * 0.15f;
		}
	}
	else
	{
		look_down_speed_ -= frame_time_* 0.005f;
		if(look_down_speed_ < 0.0f)
		{
			look_down_speed_ = 0.0f;
		}
	}
	// Update the rotation.
	rotation_x_ += look_down_speed_;
	// Keep the rotation maximum 90 degrees.
	if(rotation_x_ < -90.0f)
	{
		rotation_x_ = -90.0f;
	}
	return;
}