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
	m_frameTime = 0.0f;
	m_forwardSpeed   = 0.0f;
	m_backwardSpeed  = 0.0f;
	m_upwardSpeed    = 0.0f;
	m_downwardSpeed  = 0.0f;
	m_leftTurnSpeed  = 0.0f;
	m_rightTurnSpeed = 0.0f;
	m_lookUpSpeed    = 0.0f;
	m_lookDownSpeed  = 0.0f;
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
	m_frameTime = time;
	return;
}
void PositionClass::MoveForward(bool key_down)
{
	float radians;
	// Update the forward speed movement based on the frame time and whether the user is holding the key down or not.
	if(key_down)
	{
		m_forwardSpeed += m_frameTime * 0.001f;
		if(m_forwardSpeed > (m_frameTime * 0.03f))
		{
			m_forwardSpeed = m_frameTime * 0.03f;
		}
	}
	else
	{
		m_forwardSpeed -= m_frameTime * 0.0007f;
		if(m_forwardSpeed < 0.0f)
		{
			m_forwardSpeed = 0.0f;
		}
	}
	// Convert degrees to radians.
	radians = rotation_y_ * 0.0174532925f;
	// Update the position.
	position_x_ += sinf(radians) * m_forwardSpeed;
	position_z_ += cosf(radians) * m_forwardSpeed;
	return;
}
void PositionClass::MoveBackward(bool key_down)
{
	float radians;
	// Update the backward speed movement based on the frame time and whether the user is holding the key down or not.
	if(key_down)
	{
		m_backwardSpeed += m_frameTime * 0.001f;
		if(m_backwardSpeed > (m_frameTime * 0.03f))
		{
			m_backwardSpeed = m_frameTime * 0.03f;
		}
	}
	else
	{
		m_backwardSpeed -= m_frameTime * 0.0007f;
		
		if(m_backwardSpeed < 0.0f)
		{
			m_backwardSpeed = 0.0f;
		}
	}
	// Convert degrees to radians.
	radians = rotation_y_ * 0.0174532925f;
	// Update the position.
	position_x_ -= sinf(radians) * m_backwardSpeed;
	position_z_ -= cosf(radians) * m_backwardSpeed;
	return;
}
void PositionClass::MoveUpward(bool key_down)
{
	// Update the upward speed movement based on the frame time and whether the user is holding the key down or not.
	if(key_down)
	{
		m_upwardSpeed += m_frameTime * 0.003f;
		if(m_upwardSpeed > (m_frameTime * 0.03f))
		{
			m_upwardSpeed = m_frameTime * 0.03f;
		}
	}
	else
	{
		m_upwardSpeed -= m_frameTime * 0.002f;
		if(m_upwardSpeed < 0.0f)
		{
			m_upwardSpeed = 0.0f;
		}
	}
	// Update the height position.
	position_y_ += m_upwardSpeed;
	return;
}
void PositionClass::MoveDownward(bool key_down)
{
	// Update the downward speed movement based on the frame time and whether the user is holding the key down or not.
	if(key_down)
	{
		m_downwardSpeed += m_frameTime * 0.003f;
		if(m_downwardSpeed > (m_frameTime * 0.03f))
		{
			m_downwardSpeed = m_frameTime * 0.03f;
		}
	}
	else
	{
		m_downwardSpeed -= m_frameTime * 0.002f;
		if(m_downwardSpeed < 0.0f)
		{
			m_downwardSpeed = 0.0f;
		}
	}
	// Update the height position.
	position_y_ -= m_downwardSpeed;
	return;
}
void PositionClass::TurnLeft(bool key_down)
{
	// Update the left turn speed movement based on the frame time and whether the user is holding the key down or not.
	if(key_down)
	{
		m_leftTurnSpeed += m_frameTime * 0.01f;
		if(m_leftTurnSpeed > (m_frameTime * 0.15f))
		{
			m_leftTurnSpeed = m_frameTime * 0.15f;
		}
	}
	else
	{
		m_leftTurnSpeed -= m_frameTime* 0.005f;
		if(m_leftTurnSpeed < 0.0f)
		{
			m_leftTurnSpeed = 0.0f;
		}
	}
	// Update the rotation.
	rotation_y_ -= m_leftTurnSpeed;
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
		m_rightTurnSpeed += m_frameTime * 0.01f;
		if(m_rightTurnSpeed > (m_frameTime * 0.15f))
		{
			m_rightTurnSpeed = m_frameTime * 0.15f;
		}
	}
	else
	{
		m_rightTurnSpeed -= m_frameTime* 0.005f;
		if(m_rightTurnSpeed < 0.0f)
		{
			m_rightTurnSpeed = 0.0f;
		}
	}
	// Update the rotation.
	rotation_y_ += m_rightTurnSpeed;
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
		m_lookUpSpeed += m_frameTime * 0.01f;
		if(m_lookUpSpeed > (m_frameTime * 0.15f))
		{
			m_lookUpSpeed = m_frameTime * 0.15f;
		}
	}
	else
	{
		m_lookUpSpeed -= m_frameTime* 0.005f;
		if(m_lookUpSpeed < 0.0f)
		{
			m_lookUpSpeed = 0.0f;
		}
	}
	// Update the rotation.
	rotation_x_ -= m_lookUpSpeed;
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
		m_lookDownSpeed += m_frameTime * 0.01f;
		if(m_lookDownSpeed > (m_frameTime * 0.15f))
		{
			m_lookDownSpeed = m_frameTime * 0.15f;
		}
	}
	else
	{
		m_lookDownSpeed -= m_frameTime* 0.005f;
		if(m_lookDownSpeed < 0.0f)
		{
			m_lookDownSpeed = 0.0f;
		}
	}
	// Update the rotation.
	rotation_x_ += m_lookDownSpeed;
	// Keep the rotation maximum 90 degrees.
	if(rotation_x_ < -90.0f)
	{
		rotation_x_ = -90.0f;
	}
	return;
}