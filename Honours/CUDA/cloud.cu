#ifndef _CLOUD_CUDA_
#define _CLOUD_CUDA_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "math.h"

#include "../Source/CUDA/cuda_header.h"

//output velocity derrivitive teture //input velcoity texutre
__global__ void cuda_kernel_project(float*pressure, float* velocity, Size size, int pressure_index){
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size.depth_; ++z_iter){ 
		float*cell_velocity = velocity + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
		if(x_iter +1 < size.width_ && x_iter - 1 >= 0){
			if(y_iter + 1 < size.height_ && y_iter - 1 >= 0){
				if(z_iter + 1 < size.depth_ && z_iter - 1 >= 0){
					// Get pressure values from neighboring cells. 
					float*pLeft = pressure + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
					float*pRight = pressure + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
					float*pDown = pressure + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
					float*pUp = pressure + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
					float*pTop = pressure + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
					float*pBottom = pressure + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);

					float temp_x = (-pRight[pressure_index]) - pLeft[pressure_index];
					float temp_y = (-pTop[pressure_index]) - pBottom[pressure_index];
					float temp_z = (-pUp[pressure_index]) - pDown[pressure_index];

					float new_x = cell_velocity[x_identifier_];
					float new_y = cell_velocity[y_identifier_];
					float new_z = cell_velocity[z_identifier_];

					new_x = new_x - (time_step * temp_x);
					new_y = new_y - (time_step * temp_y);
					new_z = new_z - (time_step * temp_z);

					cell_velocity[x_identifier_] = new_x;
					cell_velocity[y_identifier_] = new_y;
					cell_velocity[z_identifier_] = new_z; 

					float density = (cell_velocity[x_identifier_] * cell_velocity[x_identifier_]) + 
						(cell_velocity[y_identifier_] * cell_velocity[y_identifier_]) + 
						(cell_velocity[z_identifier_] * cell_velocity[z_identifier_]);
					density = sqrt(density);
					//density =0.5f;
					cell_velocity[3] = density;
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

	cuda_kernel_project<<<Dg,Db>>>((float *)pressure, (float *)velocityInput, size, pressure_index);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_fluid_project() failed to launch error = %d\n", error);
	}
}
#endif