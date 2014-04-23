///////////////////////////////////////////////////////////////////////////////
// Filename: fpsclass.cpp
///////////////////////////////////////////////////////////////////////////////
#include "fpsclass.h"
FpsClass::FpsClass()
{
}
FpsClass::FpsClass(const FpsClass& other)
{
}
FpsClass::~FpsClass()
{
}
void FpsClass::Initialize()
{
	// Initialize the counters and the start time.
	fps_ = 0;
	count_ = 0;
	start_time_ = timeGetTime();
	
	return;
}
void FpsClass::Frame()
{
	count_++;
	// If one second has passed then update the frame per second speed.
	if(timeGetTime() >= (start_time_ + 1000))
	{
		fps_ = count_;
		count_ = 0;
		
		start_time_ = timeGetTime();
	}
}
int FpsClass::GetFps()
{
	return fps_;
}