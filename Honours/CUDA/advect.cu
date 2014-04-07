#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#define timeStep 1.f

__global__ void cuda_kernel_advect(unsigned char *output, unsigned char *velocityInput, float3 size_WHD, size_t pitch, size_t pitch_slice){ 
	int xIter = blockIdx.x*blockDim.x + threadIdx.x;
	int yIter = blockIdx.y*blockDim.y + threadIdx.y;
	int zIter = 0;

	for(zIter = 0; zIter < size_WHD.z; ++zIter){ 
		if(xIter +1 < size_WHD.x){
			if(xIter - 1 > 0){
				if(yIter + 1 < size_WHD.y){
					if(yIter - 1 > 0){
						if(zIter + 1 < size_WHD.z){
							if(zIter - 1 > 0){
								//location is z slide + y position + variable size time x position
								unsigned char*cellVelocity = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*xIter);
								unsigned char* outputPixel = output + (zIter*pitch_slice) + (yIter*pitch) + (4*xIter);

								float posX = float((xIter +0.5f)- (timeStep * signed int(cellVelocity[0])))/ size_WHD.x;
								float posY = float((yIter +0.5f) - (timeStep * signed int(cellVelocity[1])))/ size_WHD.y;
								float posZ = float((zIter +0.5f) - (timeStep * signed int(cellVelocity[2]))+0.5f)/ size_WHD.z;

								unsigned int location = (posZ*pitch_slice) + (posY*pitch) + (4*posX);
								cellVelocity = velocityInput + location;

								outputPixel[0] = signed int(cellVelocity[0]);
								outputPixel[1] = signed int(cellVelocity[1]);
								outputPixel[2] = signed int(cellVelocity[2]);
							}
						}
					}
				}
			}
		}
		
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