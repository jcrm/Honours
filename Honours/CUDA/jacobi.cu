#ifndef _JACOBI_CUDA_
#define _JACOBI_CUDA_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "../Source/CUDA/cuda_header.h"

__global__ void cuda_kernel_jacobi(float *pressuredivergence, Size size){
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;
	int read_identifier = pressure_identifier_;
	int write_identifier = pressure_identifier_two_;
	for(int i = 0; i < 32; i++){
		if(i%2 == 0){
			read_identifier = pressure_identifier_;
			write_identifier = pressure_identifier_two_;
		}else{
			read_identifier = pressure_identifier_two_;
			write_identifier = pressure_identifier_;
		}
		for(z_iter = 0; z_iter < size.depth_; z_iter++){
			float* cellPressure = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
			if(x_iter == 0){
				float*pRight = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
				cellPressure[write_identifier] = pRight[read_identifier];
			}else if(x_iter + 1 == size.width_){
				float*pLeft = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
				cellPressure[write_identifier] = pLeft[read_identifier];
			}
			if(y_iter == 0){
				float*pUp = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				cellPressure[write_identifier] = pUp[read_identifier];
			}else if(y_iter + 1 == size.height_){
				float*pDown = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				cellPressure[write_identifier] = pDown[read_identifier];
			}
			if(z_iter == 0){
				float*pBottom = pressuredivergence + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				cellPressure[write_identifier] = pBottom[read_identifier];
			}else if(z_iter + 1 == size.depth_){
				float*pTop = pressuredivergence + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				cellPressure[write_identifier] = pTop[read_identifier];
			}
			float sum = 0.f;
			if(x_iter +1 < size.width_ && x_iter - 1 >= 0){
				if(y_iter + 1 < size.height_ && y_iter - 1 >= 0){
					if(z_iter + 1 < size.depth_ && z_iter - 1 >= 0){
						float*p = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
						float*pLeft = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
						float*pRight = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
						float*pDown = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
						float*pUp = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
						float*pTop = pressuredivergence + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
						float*pBottom = pressuredivergence + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
						sum = pLeft[read_identifier] + pRight[read_identifier] + pUp[read_identifier] + pDown[read_identifier] + pTop[read_identifier] + pBottom[read_identifier];
						sum -= (4 *p[read_identifier]);
					}
				}
			}
			// Get the divergence at the current cell.  
			float dCentre = cellPressure[divergence_identifier_];
			float value = (sum + (dx * dx) ) * dCentre;
			value /= 8.f;
			// Compute the new pressure value for the center cell.
			cellPressure[write_identifier] = value;
		}
	}
}
extern "C"
void cuda_fluid_jacobi(void *input, Size size){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_jacobi<<<Dg,Db>>>((float *)input, size);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_jacobi() failed to launch error = %d\n", error);
	}
}
#endif