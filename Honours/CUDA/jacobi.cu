#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include "../Source/CUDA/cuda_header.h"

__global__ void cuda_kernel_jacobi(unsigned char *pressuredivergence, Size size, int pressure_index, int divergence_index){
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
		for(z_iter = 0; z_iter < size.depth_; ++z_iter){ 
			if(x_iter +1 < size.width_ && x_iter - 1 >= 0){
				if(y_iter + 1 < size.height_ && y_iter - 1 >= 0){
					if(z_iter + 1 < size.depth_ && z_iter - 1 >= 0){
					
						unsigned char* cellDivergence = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE * x_iter);
						// Get the divergence at the current cell.  
						float dCentre = cellDivergence[divergence_index];

						unsigned char *pLeft = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE * (x_iter-1));
						unsigned char *pRight = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE * (x_iter+1));
						unsigned char *pDown = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE * x_iter); 
						unsigned char *pUp = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE * x_iter); 
						unsigned char *pTop = pressuredivergence + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE * x_iter);
						unsigned char *pBottom = pressuredivergence + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE * x_iter);

						// Compute the new pressure value for the center cell.
						unsigned char* cellPressure = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE * x_iter);
						cellPressure[pressure_index] = (pLeft[pressure_index] + pRight[pressure_index] + pBottom[pressure_index] + pTop[pressure_index] + pUp[pressure_index] + pDown[pressure_index] - dCentre)/6.f;
					}
				}
			}
		}
	}
}
extern "C"
void cuda_fluid_jacobi(void *input, Size size, int pressure_index, int divergence_index){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_jacobi<<<Dg,Db>>>((unsigned char *)input, size, pressure_index, divergence_index);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_jacobi() failed to launch error = %d\n", error);
	}
}