#include "../CUDA/cuda_header.h"
// The CUDA kernel launchers that get called
extern "C"
{
	void cuda_fluid_initial(void*, Size, float);
	void cuda_fluid_initial_float(void* , Size, float);
	void cuda_fluid_initial_float_2d(void*, Size, float);
	void cuda_fluid_advect_velocity(void*, void*, Size);
	void cuda_fluid_advect_thermo(void*, Size);
	void cuda_fluid_vorticity(void*, void*, Size);
	void cuda_fluid_bouyancy(void*, void*, void*, Size, Size);
	void cuda_fluid_water_thermo(void*, void*, void*, Size);
	void cuda_fluid_divergence(void*, void*, Size, int);
	void cuda_fluid_jacobi(void*, Size, int, int);
	void cuda_fluid_project(void*, void*, Size, int);
	void cuda_fluid_rain(void*, void*, Size, Size);
}