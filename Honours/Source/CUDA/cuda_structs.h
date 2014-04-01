// This header inclues all the necessary D3D11 and CUDA includes
#include <dynlink_d3d11.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>

// includes, project
#include <rendercheck_d3d11.h>
#include <helper_cuda.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// Data structure for 2D texture shared between DX10 and CUDA
struct texture_2d{
	ID3D11Texture2D			*pTexture;
	ID3D11ShaderResourceView *pSRView;
	cudaGraphicsResource	*cudaResource;
	void					*cudaLinearMemory;
	size_t					pitch;
	int						width;
	int						height;
};
// Data structure for volume textures shared between DX10 and CUDA
struct fluid_texture_3d{
	ID3D11Texture3D			*pTexture;
	ID3D11ShaderResourceView *pSRView;
	cudaGraphicsResource	*cudaVelocityResource;
	void					*cudaAdvectLinearMemory;
	void					*cudaDivergenceLinearMemory;
	void					*cudaPressureLinearMemory;
	void					*cudaVelocityLinearMemory;
	size_t					pitch;
	int						width;
	int						height;
	int						depth;
};
// Data structure for volume textures shared between DX10 and CUDA
struct fluid_texture{
	ID3D11Texture3D			*pTexture;
	ID3D11ShaderResourceView *pSRView;
	cudaGraphicsResource	*cudaResource;
	void					*cudaLinearMemory;
	size_t					pitch;
	int						width;
	int						height;
	int						depth;
};