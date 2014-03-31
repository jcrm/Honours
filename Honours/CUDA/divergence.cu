#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

__global__ void cuda_kernel_divergence(unsigned char* divergence, unsigned char* velocityInput,float3 sizeWHD, size_t pitch, size_t pitchSlice){
	int xIter = blockIdx.x*blockDim.x + threadIdx.x;
	int yIter = blockIdx.y*blockDim.y + threadIdx.y;
	int zIter = 0;

	 // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't
    // correspond to valid pixels
    if (xIter >= sizeWHD.x || yIter >= sizeWHD.y) return;

	for(zIter = 0; zIter < sizeWHD.z; ++zIter){ 
		// Get velocity values from neighboring cells.  
		unsigned char *fieldLeft = velocityInput + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter-1));
		unsigned char *fieldRight = velocityInput + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter+1));
		unsigned char *fieldBottom = velocityInput + ((zIter+1)*pitchSlice) + (yIter*pitch) + (4*xIter);  
		unsigned char *fieldTop = velocityInput + ((zIter-1)*pitchSlice) + (yIter*pitch) + (4*xIter);   
		unsigned char *fieldDown = velocityInput + (zIter*pitchSlice) + ((yIter-1)*pitch) + (4*xIter); 
		unsigned char *fieldUp = velocityInput + (zIter*pitchSlice) + ((yIter+1)*pitch) + (4*xIter);
		unsigned char* cellDivergence = divergence + (zIter*pitchSlice) + (yIter*pitch) + (4*xIter);
		// Compute the velocity's divergence using central differences.  
		cellDivergence[0] =  0.5 * ((fieldRight[0] - fieldLeft[0])+  
									(fieldTop[1] - fieldBottom[1])+  
									(fieldUp[2] - fieldDown[2])); 
	}
}
extern "C"
void cuda_fluid_divergence(void *divergence, void *velocityInput, float3 sizeWHD, size_t pitch, size_t pitchSlice){
	cudaError_t error = cudaSuccess;

    dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((sizeWHD.x+Db.x-1)/Db.x, (sizeWHD.y+Db.y-1)/Db.y);

    cuda_kernel_divergence<<<Dg,Db>>>((unsigned char *)divergence, (unsigned char *)velocityInput,sizeWHD, pitch, pitchSlice);

    error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("cuda_kernel_divergence() failed to launch error = %d\n", error);
    }
}
