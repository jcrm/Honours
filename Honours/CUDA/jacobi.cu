#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

__global__ void cuda_kernel_jacobi(unsigned char *pressuredivergence, float3 size_WHD, size_t pitch, size_t pitch_slice, int pressure_index, int divergence_index){  
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
								unsigned char* cellDivergence = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*xIter);
								// Get the divergence at the current cell.  
								float dCentre = cellDivergence[divergence_index];

								unsigned char *pLeft = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
								unsigned char *pRight = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
								unsigned char *pDown = pressuredivergence + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
								unsigned char *pUp = pressuredivergence + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
								unsigned char *pTop = pressuredivergence + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
								unsigned char *pBottom = pressuredivergence + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);

								// Compute the new pressure value for the center cell.
								unsigned char* cellPressure = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*xIter);
								cellPressure[pressure_index] = (pLeft[pressure_index] + pRight[pressure_index] + pBottom[pressure_index] + pTop[pressure_index] + pUp[pressure_index] + pDown[pressure_index] - dCentre)/6.f;
							}
						}
					}
				}
			}
		}
	}
}
extern "C"
void cuda_fluid_jacobi(void *pressuredivergence, float3 size_WHD, size_t pitch, size_t pitch_slice, int pressure_index, int divergence_index){
	cudaError_t error = cudaSuccess;

    dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((size_WHD.x+Db.x-1)/Db.x, (size_WHD.y+Db.y-1)/Db.y);

    cuda_kernel_jacobi<<<Dg,Db>>>((unsigned char *)pressuredivergence,size_WHD, pitch, pitch_slice, pressure_index, divergence_index);

    error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("cuda_kernel_jacobi() failed to launch error = %d\n", error);
    }
}