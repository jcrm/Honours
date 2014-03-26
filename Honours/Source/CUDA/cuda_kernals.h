
// The CUDA kernel launchers that get called
extern "C"
{
    bool cuda_texture_2d(void *surface, size_t width, size_t height, size_t pitch, float t);
    bool cuda_texture_3d(void *surface, int width, int height, int depth, size_t pitch, size_t pitchslice, float t);
    bool cuda_texture_cube(void *surface, int width, int height, size_t pitch, int face, float t);
	
	void cuda_fluid_advect(void *output, void *velocityinput, float3 sizeWHD, size_t pitch, size_t pitchSlice);
	void cuda_fluid_divergence(void *divergence, void *velocityInput, float3 sizeWHD, size_t pitch, size_t pitchSlice);
	void cuda_fluid_jacobi(void *pressure, void *divergence, float3 sizeWHD, size_t pitch, size_t pitchSlice);
	void cuda_fluid_project(void *pressure, void *velocityInput, float3 sizeWHD, size_t pitch, size_t pitchSlice);
}