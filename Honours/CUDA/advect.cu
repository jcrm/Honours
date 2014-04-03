#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

__global__ void cuda_kernel_advect(unsigned char *output, unsigned char *velocityInput, float3 size_WHD, size_t pitch, size_t pitch_slice){ 
	int xIter = blockIdx.x*blockDim.x + threadIdx.x;
	int yIter = blockIdx.y*blockDim.y + threadIdx.y;
	int zIter = 0;
	float timeStep = 1.f;
	unsigned int location = 0;
	unsigned char* cellVelocity = NULL;
	float cellVX, cellVY, cellVZ = 0.f;
	float posX, posY, posZ = 0.f;
	unsigned char* outputPixel = NULL;
	for(zIter = 0; zIter < size_WHD.z; ++zIter){ 
		//location is z slide + y position + variable size time x position
		location =(zIter*pitch_slice) + (yIter*pitch) + (4*xIter);
		cellVelocity = velocityInput + location;

		cellVX = signed int(cellVelocity[0]);
		cellVY = signed int(cellVelocity[1]);
		cellVZ = signed int(cellVelocity[2]);
		float3 cell = {cellVX, cellVY, cellVZ};
		posX = float((xIter +0.5f)- (timeStep * cellVX))/ size_WHD.x;
		posY = float((yIter +0.5f) - (timeStep * cellVY))/ size_WHD.y;
		posZ = float((zIter +0.5f) - (timeStep * cellVZ)+0.5f)/ size_WHD.z;
		float3 pos = {posX, posY, posZ};
		outputPixel = output + location;
		location =(posZ*pitch_slice) + (posY*pitch) + (4*posX);
		cellVelocity = velocityInput + location;
		cellVX = signed int(cellVelocity[0]);
		cellVY = signed int(cellVelocity[1]);
		cellVZ = signed int(cellVelocity[2]);
		float3 cellG = {cellVX, cellVY, cellVZ};
		outputPixel[0] = cellVX;
		outputPixel[1] = cellVY;
		outputPixel[2] = cellVZ;
	}
}

extern "C"
void cuda_fluid_advect(void *output, void *velocityinput, float3 size_WHD, size_t pitch, size_t pitch_slice){
	cudaError_t error = cudaSuccess;

    dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((size_WHD.x+Db.x-1)/Db.x, (size_WHD.y+Db.y-1)/Db.y);

    cuda_kernel_advect<<<Dg,Db>>>((unsigned char *)output, (unsigned char *)velocityinput, size_WHD, pitch, pitch_slice);

    error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("cuda_kernel_advect() failed to launch error = %d\n", error);
    }
}