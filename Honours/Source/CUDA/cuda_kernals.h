// The CUDA kernel launchers that get called
extern "C"
{
	void cuda_fluid_initial(void *velocity_input, float3 size_WHD, size_t pitch, size_t pitch_slice, float value);
	void cuda_fluid_advect_one_texture(void *input, float3 size_WHD, size_t pitch, size_t pitch_slice, float4 advect_index);
	void cuda_fluid_advect_two_texture(void *output, void *input, float3 size_WHD, size_t pitch, size_t pitch_slice);
	void cuda_fluid_forces(void *output, void *input, float3 size_WHD, size_t pitch, size_t pitch_slice);
	void cuda_fluid_water(void *input, float3 size_WHD, size_t pitch, size_t pitch_slice, int pressure_index, int divergence_index);
	void cuda_fluid_divergence(void *divergence, void *input, float3 size_WHD, size_t pitch, size_t pitch_slice, int divergence_index);
	void cuda_fluid_jacobi(void *input, float3 size_WHD, size_t pitch, size_t pitch_slice, int pressure_index, int divergence_index);
	void cuda_fluid_project(void *pressure, void *velocityInput, float3 size_WHD, size_t pitch, size_t pitch_slice, int pressure_index);
}