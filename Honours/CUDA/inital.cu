#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

__global__ void cuda_kernel_initial(unsigned char *velocityInput, float3 size_WHD, size_t pitch, size_t pitch_slice, float value){ 
	int xIter = blockIdx.x*blockDim.x + threadIdx.x;
	int yIter = blockIdx.y*blockDim.y + threadIdx.y;
	int zIter = 0;

	for(zIter = 0; zIter < size_WHD.z; ++zIter){ 
		//location is z slide + y position + variable size time x position
		unsigned char* cellVelocity = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*xIter);
		cellVelocity[0] = signed int(0);
		cellVelocity[1] = signed int(0);
		cellVelocity[2] = signed int(0);
		cellVelocity[3] = signed int(0);
		if(xIter +1 < size_WHD.x){
			if(xIter - 1 > 0){
				if(yIter + 1 < size_WHD.y){
					if(yIter - 1 > 0){
						if(zIter + 1 < size_WHD.z){
							if(zIter - 1 > 0){
								cellVelocity[0] = signed int(0);
								cellVelocity[1] = signed int(0);
								cellVelocity[2] = signed int(0);
								cellVelocity[3] = signed int(0);
							}
						}
					}
				}
			}
		}
	}
}
extern "C"
void cuda_fluid_initial(void *velocityinput, float3 size_WHD, size_t pitch, size_t pitch_slice, float value){
	cudaError_t error = cudaSuccess;

    dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((size_WHD.x+Db.x-1)/Db.x, (size_WHD.y+Db.y-1)/Db.y);

    cuda_kernel_initial<<<Dg,Db>>>((unsigned char *)velocityinput, size_WHD, pitch, pitch_slice, value);

    error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("cuda_kernel_initial() failed to launch error = %d\n", error);
    }
}