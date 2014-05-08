///////////////////////////////////////////////////////////////////////////////
// Filename: cpuclass.cpp
///////////////////////////////////////////////////////////////////////////////
#include "cpuclass.h"
CpuClass::CpuClass(){
}
CpuClass::CpuClass(const CpuClass& other){
}
CpuClass::~CpuClass(){
}
void CpuClass::Initialize(){
	PDH_STATUS status;
	// Initialize the flag indicating whether this object can read the system cpu usage or not.
	can_read_cpu_ = true;
	// Create a query object to poll cpu usage.
	status = PdhOpenQuery(NULL, 0, &query_handle_);
	if(status != ERROR_SUCCESS){
		can_read_cpu_ = false;
	}
	// Set query object to poll all cpus in the system.
	status = PdhAddCounter(query_handle_, TEXT("\\Processor(_Total)\\% processor time"), 0, &counter_handle_);
	if(status != ERROR_SUCCESS){
		can_read_cpu_ = false;
	}
	// Initialize the start time and cpu usage.
	last_sample_time_ = GetTickCount(); 
	cpu_usage_ = 0;
	return;
}
void CpuClass::Shutdown(){
	if(can_read_cpu_){
		PdhCloseQuery(query_handle_);
	}
	return;
}
void CpuClass::Frame(){
	PDH_FMT_COUNTERVALUE value; 
	if(can_read_cpu_){
		// If it has been 1 second then update the current cpu usage and reset the 1 second timer again.
		if((last_sample_time_ + 1000) < GetTickCount()){
			last_sample_time_ = GetTickCount(); 
			PdhCollectQueryData(query_handle_);
		
			PdhGetFormattedCounterValue(counter_handle_, PDH_FMT_LONG, NULL, &value);
			cpu_usage_ = value.longValue;
		}
	}
	return;
}
int CpuClass::GetCpuPercentage(){
	int usage;
	// If the class can read the cpu from the operating system then return the current usage.  If not then return zero.
	if(can_read_cpu_){
		usage = (int)cpu_usage_;
	}else{
		usage = 0;
	}
	return usage;
}