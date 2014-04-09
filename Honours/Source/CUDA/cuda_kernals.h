// The CUDA kernel launchers that get called
extern "C"
{
	void cuda_fluid_initial(void *velocityinput, float3 size_WHD, size_t pitch, size_t pitch_slice, float value);
	void cuda_fluid_advect(void *output, void *velocityinput, float3 size_WHD, size_t pitch, size_t pitch_slice);
	void cuda_fluid_forces(void *output, void *velocityinput, float3 size_WHD, size_t pitch, size_t pitch_slice);
	void cuda_fluid_water(void *pressuredivergence, float3 size_WHD, size_t pitch, size_t pitch_slice, int pressure_index, int divergence_index);
	void cuda_fluid_divergence(void *divergence, void *velocityInput, float3 size_WHD, size_t pitch, size_t pitch_slice, int divergence_index);
	void cuda_fluid_jacobi(void *pressuredivergence, float3 size_WHD, size_t pitch, size_t pitch_slice, int pressure_index, int divergence_index);
	void cuda_fluid_project(void *pressure, void *velocityInput, float3 size_WHD, size_t pitch, size_t pitch_slice, int pressure_index);
}