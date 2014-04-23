// The CUDA kernel launchers that get called
extern "C"
{
	void cuda_fluid_initial(void*, float3, size_t, size_t, float);
	void cuda_fluid_advect_one_texture(void*, float3, size_t, size_t, float4);
	void cuda_fluid_advect_two_texture(void*, void*, float3, size_t, size_t);
	void cuda_fluid_forces(void*, void*, float3, size_t, size_t);
	void cuda_fluid_water(void*, float3, size_t, size_t, int, int);
	void cuda_fluid_divergence(void*, void*, float3, size_t, size_t, int);
	void cuda_fluid_jacobi(void*, float3, size_t, size_t, int, int);
	void cuda_fluid_project(void*, void*, float3, size_t, size_t, int);
}