#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

__global__ void cuda_kernel_project(unsigned char *pressure, unsigned char* velocityInput, size_t pitch, size_t pitchSlice){
	int xIter = blockIdx.x*blockDim.x + threadIdx.x;
	int yIter = blockIdx.y*blockDim.y + threadIdx.y;
	int zIter = 0;
	// Compute the gradient of pressure at the current cell by  
	// taking central differences of neighboring pressure values.  
	unsigned char *pLeft = pressure + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter-1));
	unsigned char *pRight = pressure + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter+1));
	unsigned char *pBottom = pressure + ((zIter+1)*pitchSlice) + (yIter*pitch) + (4*xIter);
	unsigned char *pTop = pressure + ((zIter-1)*pitchSlice) + (yIter*pitch) + (4*xIter);
	unsigned char *pDown = pressure + (zIter*pitchSlice) + ((yIter-1)*pitch) + (4*xIter);  
	unsigned char *pUp = pressure + (zIter*pitchSlice) + ((yIter+1)*pitch) + (4*xIter);
	float3 gradP;
	gradP.x = 0.5 *(pRight[0] - pLeft[0]);
	gradP.y = 0.5 *(pTop[0] - pBottom[0]);
	gradP.z = 0.5 *(pUp[0] - pDown[0]);
	unsigned char* cellVelocity = velocityInput + (zIter*pitchSlice) + (yIter*pitch) + (4*xIter);
	// Project the velocity onto its divergence-free component by  
	// subtracting the gradient of pressure.  
	float3 vOld;  
	vOld.x = cellVelocity[0];
	vOld.y = cellVelocity[1];
	vOld.z = cellVelocity[2];
	cellVelocity[0] = vOld.x - gradP.x;
	cellVelocity[1]= vOld.y - gradP.y;
	cellVelocity[2] = vOld.z - gradP.z; 
}

extern "C"
void cuda_fluid_project(unsigned char* pressure, unsigned char* velocityInput, float3 sizeWHD, size_t pitch, size_t pitchSlice){
	cudaError_t error = cudaSuccess;

    dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((sizeWHD.x+Db.x-1)/Db.x, (sizeWHD.y+Db.y-1)/Db.y);

    cuda_kernel_project<<<Dg,Db>>>((unsigned char *)pressure, (unsigned char *)velocityInput, pitch, pitchSlice);

    error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("cuda_kernel_texture_3d() failed to launch error = %d\n", error);
    }
}