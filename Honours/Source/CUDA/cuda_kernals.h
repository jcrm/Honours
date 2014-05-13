#ifndef _CUDA_KERNALS_
#define _CUDA_KERNALS_

#include "../CUDA/cuda_header.h"
// The CUDA kernel launchers that get called
extern "C"
{
	void cuda_fluid_initial(void*, Size, float);
	void cuda_fluid_initial_float(void*, Size, float);
	void cuda_fluid_initial_float_2d(void*, Size, float);
	void cuda_fluid_advect_velocity(void*, void*, Size);
	void cuda_fluid_advect_thermo(void*, void*, Size);
	void cuda_fluid_vorticity(void*, void*, Size);
	void cuda_fluid_bouyancy(void*, void*, void*, Size, Size);
	void cuda_fluid_water_thermo(void*, void*, void*, Size);
	void cuda_fluid_divergence(void*, void*, Size);
	void cuda_fluid_jacobi(void*, Size);
	void cuda_fluid_project(void*, void*,void*, Size);
	void cuda_fluid_rain(void*, void*, Size, Size);
	void cuda_fluid_boundaries(void*, Size);
	void cuda_fluid_boundaries_thermo(void*, Size, float, float);
}

#endif