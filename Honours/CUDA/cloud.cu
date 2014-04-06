#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

__global__ void cuda_kernel_project(unsigned char *pressure, unsigned char* velocityInput,float3 size_WHD, size_t pitch, size_t pitch_slice, int pressure_index){
	int xIter = blockIdx.x*blockDim.x + threadIdx.x;
	int yIter = blockIdx.y*blockDim.y + threadIdx.y;
	int zIter = 0;

	for(zIter = 0; zIter < size_WHD.z; ++zIter){ 
		// Get pressure values from neighboring cells. 
		unsigned char *pLeft = NULL;
		unsigned char *pRight = NULL;
		unsigned char *pDown = NULL;
		unsigned char *pUp = NULL;
		unsigned char *pBottom = NULL;
		unsigned char *pTop = NULL;

		if(xIter +1 < size_WHD.x){
			if(xIter - 1 > 0){
				if(yIter + 1 < size_WHD.y){
					if(yIter - 1 > 0){
						if(zIter + 1 < size_WHD.z){
							if(zIter - 1 > 0){
								pLeft = pressure + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
								pRight = pressure + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
								pDown = pressure + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
								pUp = pressure + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
								pTop = pressure + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
								pBottom = pressure + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
								float3 gradP;
		gradP.x = 0.5 *(pRight[pressure_index] - pLeft[pressure_index]);
		gradP.y = 0.5 *(pTop[pressure_index] - pBottom[pressure_index]);
		gradP.z = 0.5 *(pUp[pressure_index] - pDown[pressure_index]);
		unsigned char* cellVelocity = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*xIter);
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
						}
					}
				}
			}
		}
		
	}
}

extern "C"
void cuda_fluid_project(void *pressure, void *velocityInput, float3 size_WHD, size_t pitch, size_t pitch_slice, int pressure_index){
	cudaError_t error = cudaSuccess;

    dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((size_WHD.x+Db.x-1)/Db.x, (size_WHD.y+Db.y-1)/Db.y);

    cuda_kernel_project<<<Dg,Db>>>((unsigned char *)pressure, (unsigned char *)velocityInput,size_WHD, pitch, pitch_slice, pressure_index);

    error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("cuda_fluid_project() failed to launch error = %d\n", error);
    }
}