#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <math.h> 
#define dx 1.f
#define time_step 1.f

__global__ void cuda_kernel_forces(unsigned char *output, unsigned char *velocityInput, float3 size_WHD, size_t pitch, size_t pitch_slice){ 
	int xIter = blockIdx.x*blockDim.x + threadIdx.x;
	int yIter = blockIdx.y*blockDim.y + threadIdx.y;
	int zIter = 0;

	for(zIter = 0; zIter < size_WHD.z; ++zIter){ 
		if(xIter +1 < size_WHD.x && xIter - 1 > 0){
			if(yIter + 1 < size_WHD.y && yIter - 1 > 0){
				if(zIter + 1 < size_WHD.z && zIter - 1 > 0){
					unsigned char*outputVelocity = output + (zIter*pitch_slice) + (yIter*pitch) + (4*xIter);
					//vorticity confinement
					float scalar = 1.f;
					unsigned char *pLeft = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
					unsigned char *pRight = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
					unsigned char *pDown = velocityInput + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
					unsigned char *pUp = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
					unsigned char *pTop = velocityInput + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
					unsigned char *pBottom = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);

					float3 curl_value = {
						((pDown[2] - pUp[2]) - (pBottom[1] - pTop[1])) / dx, 
						((pBottom[0] - pTop[0]) - (pRight[2] - pLeft[2])) / dx, 
						((pRight[1] - pLeft[1]) - (pDown[0] - pUp[0])) / dx};

					;
					float invLen = 1.0f / sqrt((curl_value.x * curl_value.x) + (curl_value.y * curl_value.y) + (curl_value.z * curl_value.z));
					float3 norm_value = {
						curl_value.x * invLen,
						curl_value.y * invLen,
						curl_value.z * invLen};

					outputVelocity[0] += ((norm_value.y * curl_value.z) - (norm_value.z * curl_value.y)) * dx * scalar * time_step;
					outputVelocity[1] += ((norm_value.z * curl_value.x) - (norm_value.x * curl_value.z)) * dx * scalar * time_step;
					outputVelocity[2] += ((norm_value.x * curl_value.y) - (norm_value.y * curl_value.x)) * dx * scalar * time_step;
				}
			}
		}
	}
}

extern "C"
void cuda_fluid_forces(void *output, void *velocityinput, float3 size_WHD, size_t pitch, size_t pitch_slice){
	cudaError_t error = cudaSuccess;

    dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((size_WHD.x+Db.x-1)/Db.x, (size_WHD.y+Db.y-1)/Db.y);

    cuda_kernel_forces<<<Dg,Db>>>((unsigned char *)output, (unsigned char *)velocityinput, size_WHD, pitch, pitch_slice);

    error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("cuda_kernel_advect() failed to launch error = %d\n", error);
    }
}