// The CUDA kernel launchers that get called
extern "C"
{
	bool cuda_texture_2d(void *surface, size_t width, size_t height, size_t pitch, float t);

	void cuda_fluid_initial(void *velocityinput, float3 sizeWHD, size_t pitch, size_t pitchSlice, float value);
	void cuda_fluid_advect(void *output, void *velocityinput, float3 sizeWHD, size_t pitch, size_t pitchSlice);
	void cuda_fluid_divergence(void *divergence, void *velocityInput, float3 sizeWHD, size_t pitch, size_t pitchSlice, int divergenceIndex);
	void cuda_fluid_jacobi(void *pressuredivergence, float3 sizeWHD, size_t pitch, size_t pitchSlice, int pressureIndex, int divergenceIndex);
	void cuda_fluid_project(void *pressure, void *velocityInput, float3 sizeWHD, size_t pitch, size_t pitchSlice, int pressureIndex);
}