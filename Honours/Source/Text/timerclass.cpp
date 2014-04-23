///////////////////////////////////////////////////////////////////////////////
// Filename: timerclass.cpp
///////////////////////////////////////////////////////////////////////////////
#include "timerclass.h"
TimerClass::TimerClass(){
}
TimerClass::TimerClass(const TimerClass& other){
}
TimerClass::~TimerClass(){
}
bool TimerClass::Initialize()
{
	// Check to see if this system supports high performance timers.
	QueryPerformanceFrequency((LARGE_INTEGER*)&frequency_);
	if(frequency_ == 0){
		return false;
	}
	// Find out how many times the frequency counter ticks every millisecond.
	ticks_per_mins_ = (float)(frequency_ / 1000);
	QueryPerformanceCounter((LARGE_INTEGER*)&start_time_);
	return true;
}
void TimerClass::Frame(){
	INT64 current_time;
	float time_difference;
	// Query the current time.
	QueryPerformanceCounter((LARGE_INTEGER*)&current_time);
	// Calculate the difference in time since the last time we queried for the current time.
	time_difference = (float)(current_time - start_time_);
	// Calculate the frame time by the time difference over the timer speed resolution.
	frame_time_ = time_difference / ticks_per_mins_;
	// Restart the timer.
	start_time_ = current_time;
	return;
}
float TimerClass::GetTime(){
	return frame_time_;
}