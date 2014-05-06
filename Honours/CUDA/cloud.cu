#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include "math.h"
#include "../Source/CUDA/cuda_header.h"

#define PIXEL_FMT_SIZE 4
#define x_identifier_ 0
#define y_identifier_ 1
#define z_identifier_ 2
//output velocity derrivitive teture //input velcoity texutre
__global__ void cuda_kernel_project(unsigned char *pressure, unsigned char* velocity, Size size, int pressure_index){
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size.depth_; ++z_iter){ 
		unsigned char *cell_velocity = velocity + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE * x_iter);
		if(x_iter +1 < size.width_ && x_iter - 1 >= 0){
			if(y_iter + 1 < size.height_ && y_iter - 1 >= 0){
				if(z_iter + 1 < size.depth_ && z_iter - 1 >= 0){
					// Get pressure values from neighboring cells. 
					unsigned char *pLeft = pressure + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE * (x_iter-1));
					unsigned char *pRight = pressure + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE * (x_iter+1));
					unsigned char *pDown = pressure + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE * x_iter); 
					unsigned char *pUp = pressure + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE * x_iter); 
					unsigned char *pTop = pressure + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE * x_iter);
					unsigned char *pBottom = pressure + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE * x_iter);
					
					cell_velocity[x_identifier_] = cell_velocity[x_identifier_] - (0.5f *(pRight[pressure_index] - pLeft[pressure_index]));
					cell_velocity[y_identifier_] = cell_velocity[y_identifier_] - (0.5f *(pTop[pressure_index] - pBottom[pressure_index]));
					cell_velocity[2] = cell_velocity[2] - (0.5f *(pUp[pressure_index] - pDown[pressure_index])); 
					float density = (cell_velocity[x_identifier_] * cell_velocity[x_identifier_]) + 
						(cell_velocity[y_identifier_] * cell_velocity[y_identifier_]) + 
						(cell_velocity[z_identifier_] * cell_velocity[z_identifier_]);
					cell_velocity[3] = sqrt(density);
				}
			}
		}
	}
}

extern "C"
void cuda_fluid_project(void *pressure, void *velocityInput, Size size, int pressure_index){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_project<<<Dg,Db>>>((unsigned char *)pressure, (unsigned char *)velocityInput, size, pressure_index);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_fluid_project() failed to launch error = %d\n", error);
	}
}