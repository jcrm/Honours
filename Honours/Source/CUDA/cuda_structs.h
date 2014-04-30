// This header inclues all the necessary D3D11 and CUDA includes
#include <dynlink_d3d11.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
// includes, project
#include <rendercheck_d3d11.h>
#include <helper_cuda.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
// Data structure for volume textures shared between DX10 and CUDA
struct fluid_texture{
	ID3D11Texture3D *texture_;
	ID3D11ShaderResourceView *sr_view_;
	cudaGraphicsResource *cuda_resource_;
	void *cuda_linear_memory_;
	size_t pitch_;
	int width_;
	int height_;
	int depth_;
};
// Data structure for 2D texture shared between DX10 and CUDA
struct rain_texture{
	ID3D11Texture2D *texture_;
	ID3D11ShaderResourceView *sr_view_;
	cudaGraphicsResource *cuda_resource_;
	void *cuda_linear_memory_;
	size_t pitch_;
	int width_;
	int height_;
};