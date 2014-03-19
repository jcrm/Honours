#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

__global__ void cuda_kernel_jacobi(unsigned char *pressure, unsigned char *divergence, size_t pitch, size_t pitchSlice){  
	int xIter = blockIdx.x*blockDim.x + threadIdx.x;
	int yIter = blockIdx.y*blockDim.y + threadIdx.y;
	int zIter = 0;
	unsigned char* cellDivergence = divergence + (zIter*pitchSlice) + (yIter*pitch) + (4*xIter);
	// Get the divergence at the current cell.  
	float dCentre = cellDivergence[0];  
	// Get pressure values from neighboring cells.  
	unsigned char *pLeft = pressure + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter-1));
	unsigned char *pRight = pressure + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter+1));
	unsigned char *pBottom = pressure + ((zIter+1)*pitchSlice) + (yIter*pitch) + (4*xIter);
	unsigned char *pTop = pressure + ((zIter-1)*pitchSlice) + (yIter*pitch) + (4*xIter);
	unsigned char *pDown = pressure + (zIter*pitchSlice) + ((yIter-1)*pitch) + (4*xIter);  
	unsigned char *pUp = pressure + (zIter*pitchSlice) + ((yIter+1)*pitch) + (4*xIter); 

	// Compute the new pressure value for the center cell.
	unsigned char* cellPressure = pressure + (zIter*pitchSlice) + (yIter*pitch) + (4*xIter);
	cellPressure[0] = (pLeft[0] + pRight[0] + pBottom[0] + pTop[0] + pUp[0] + pDown[0] - dCentre) / 6.0;  
}
extern "C"
void cuda_fluid_jacobi(unsigned char* pressure, unsigned char* divergence, float3 sizeWHD, size_t pitch, size_t pitchSlice){
	cudaError_t error = cudaSuccess;

    dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((sizeWHD.x+Db.x-1)/Db.x, (sizeWHD.y+Db.y-1)/Db.y);

    cuda_kernel_jacobi<<<Dg,Db>>>((unsigned char *)pressure, (unsigned char *)divergence, pitch, pitchSlice);

    error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("cuda_kernel_texture_3d() failed to launch error = %d\n", error);
    }
}