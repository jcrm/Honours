#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

__global__ void cuda_kernel_advect(unsigned char *output, unsigned char *velocityInput, float3 sizeWHD, size_t pitch, size_t pitchSlice){ 
	int xIter = blockIdx.x*blockDim.x + threadIdx.x;
	int yIter = blockIdx.y*blockDim.y + threadIdx.y;
	int zIter = 0;
	float timeStep = 0.01f;

	for(zIter = 0; zIter < sizeWHD.z; ++zIter){ 
		//location is z slide + y position + variable size time x position
		int location =(zIter*pitchSlice) + (yIter*pitch) + (4*xIter);
		unsigned char* cellVelocity = velocityInput + location;
		float3 cVelo;
		cVelo.x = cellVelocity[0];
		cVelo.y = cellVelocity[1];
		cVelo.z = cellVelocity[2];
		float3 pos;
		pos.x = xIter - (timeStep * cVelo.x);
		pos.x = yIter - (timeStep * cVelo.y);
		pos.x = zIter - (timeStep * cVelo.z);
		
		pos.x = pos.x / sizeWHD.x;
		pos.y = pos.y / sizeWHD.y;
		pos.z = (pos.z +0.5)/ sizeWHD.z;

		unsigned char* outputPixel = output + location;
		location =(pos.z*pitchSlice) + (pos.y*pitch) + (4*pos.x);
		cellVelocity = velocityInput + location;
		cVelo.x = cellVelocity[0];
		cVelo.y = cellVelocity[1];
		cVelo.z = cellVelocity[2];
		outputPixel[0] = cVelo.x;
		outputPixel[1] = cVelo.y;
		outputPixel[2] = cVelo.z;
		outputPixel[3] = 255; 
	}
}

extern "C"
void cuda_fluid_advect(void *output, void *velocityinput, float3 sizeWHD, size_t pitch, size_t pitchSlice){
	cudaError_t error = cudaSuccess;

    dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((sizeWHD.x+Db.x-1)/Db.x, (sizeWHD.y+Db.y-1)/Db.y);

    cuda_kernel_advect<<<Dg,Db>>>((unsigned char *)output, (unsigned char *)velocityinput, sizeWHD, pitch, pitchSlice);

    error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("cuda_kernel_advect() failed to launch error = %d\n", error);
    }
}