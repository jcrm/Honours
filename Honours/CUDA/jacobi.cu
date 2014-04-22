#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

__global__ void cuda_kernel_jacobi(unsigned char *pressuredivergence, float3 size_WHD, size_t pitch, size_t pitch_slice, int pressure_index, int divergence_index){  
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;
	for(int i = 0; i < 16; i++){
		if(i%2 == 0){
			pressure_index = 0;
			divergence_index = 1;
		}else{
			pressure_index = 1;
			divergence_index = 0;
		}
		for(z_iter = 0; z_iter < size_WHD.z; ++z_iter){
			if(x_iter +1 < size_WHD.x && x_iter - 1 > 0){
				if(y_iter + 1 < size_WHD.y && y_iter - 1 > 0){
					if(z_iter + 1 < size_WHD.z && z_iter - 1 > 0){
					
						unsigned char* cellDivergence = pressuredivergence + (z_iter*pitch_slice) + (y_iter*pitch) + (4*x_iter);
						// Get the divergence at the current cell.  
						float dCentre = cellDivergence[divergence_index];

						unsigned char *pLeft = pressuredivergence + (z_iter*pitch_slice) + (y_iter*pitch) + (4*(x_iter-1));
						unsigned char *pRight = pressuredivergence + (z_iter*pitch_slice) + (y_iter*pitch) + (4*(x_iter+1));
						unsigned char *pDown = pressuredivergence + (z_iter*pitch_slice) + ((y_iter-1)*pitch) + (4*x_iter); 
						unsigned char *pUp = pressuredivergence + (z_iter*pitch_slice) + ((y_iter+1)*pitch) + (4*x_iter); 
						unsigned char *pTop = pressuredivergence + ((z_iter-1)*pitch_slice) + (y_iter*pitch) + (4*x_iter);
						unsigned char *pBottom = pressuredivergence + ((z_iter+1)*pitch_slice) + (y_iter*pitch) + (4*x_iter);

						// Compute the new pressure value for the center cell.
						unsigned char* cellPressure = pressuredivergence + (z_iter*pitch_slice) + (y_iter*pitch) + (4*x_iter);
						cellPressure[pressure_index] = (pLeft[pressure_index] + pRight[pressure_index] + pBottom[pressure_index] + pTop[pressure_index] + pUp[pressure_index] + pDown[pressure_index] - dCentre)/6.f;
					}
				}
			}
		}
	}
}
extern "C"
void cuda_fluid_jacobi(void *input, float3 size_WHD, size_t pitch, size_t pitch_slice, int pressure_index, int divergence_index){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size_WHD.x+Db.x-1)/Db.x, (size_WHD.y+Db.y-1)/Db.y);

	cuda_kernel_jacobi<<<Dg,Db>>>((unsigned char *)input,size_WHD, pitch, pitch_slice, pressure_index, divergence_index);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_jacobi() failed to launch error = %d\n", error);
	}
}