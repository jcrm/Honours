#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include "math.h"
#define PIXEL_FMT_SIZE 4
//output velocity derrivitive teture //input velcoity texutre
__global__ void cuda_kernel_project(unsigned char *pressure, unsigned char* velocity,float3 size_WHD, size_t pitch, size_t pitch_slice, int pressure_index){
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size_WHD.z; ++z_iter){ 
		if(x_iter +1 < size_WHD.x && x_iter - 1 >= 0){
			if(y_iter + 1 < size_WHD.y && y_iter - 1 >= 0){
				if(z_iter + 1 < size_WHD.z && z_iter - 1 >= 0){
					// Get pressure values from neighboring cells. 
					unsigned char *pLeft = pressure + (z_iter*pitch_slice) + (y_iter*pitch) + (PIXEL_FMT_SIZE * (x_iter-1));
					unsigned char *pRight = pressure + (z_iter*pitch_slice) + (y_iter*pitch) + (PIXEL_FMT_SIZE * (x_iter+1));
					unsigned char *pDown = pressure + (z_iter*pitch_slice) + ((y_iter-1)*pitch) + (PIXEL_FMT_SIZE * x_iter); 
					unsigned char *pUp = pressure + (z_iter*pitch_slice) + ((y_iter+1)*pitch) + (PIXEL_FMT_SIZE * x_iter); 
					unsigned char *pTop = pressure + ((z_iter-1)*pitch_slice) + (y_iter*pitch) + (PIXEL_FMT_SIZE * x_iter);
					unsigned char *pBottom = pressure + ((z_iter+1)*pitch_slice) + (y_iter*pitch) + (PIXEL_FMT_SIZE * x_iter);
					unsigned char *cell_velocity = velocity + (z_iter*pitch_slice) + (y_iter*pitch) + (PIXEL_FMT_SIZE * x_iter);
					cell_velocity[0] = cell_velocity[0] - (0.5f *(pRight[pressure_index] - pLeft[pressure_index]));
					cell_velocity[1] = cell_velocity[1] - (0.5f *(pTop[pressure_index] - pBottom[pressure_index]));
					cell_velocity[2] = cell_velocity[2] - (0.5f *(pUp[pressure_index] - pDown[pressure_index])); 
					float density = (cell_velocity[0] * cell_velocity[0]) + (cell_velocity[1] * cell_velocity[1])+ (cell_velocity[2] * cell_velocity[2]);
					cell_velocity[3] = sqrt(density);
					//cell_velocity[3] = density;
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