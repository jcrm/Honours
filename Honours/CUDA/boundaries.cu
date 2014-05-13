#ifndef _BOUNDARIES_CUDA_
#define _BOUNDARIES_CUDA_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "math.h"

#include "../Source/CUDA/cuda_header.h"

//output velocity derrivitive teture //input velcoity texutre
__global__ void cuda_kernel_boundaries(float*input, Size size){
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size.depth_; z_iter++){ 
		float*cell = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);

		if(x_iter == 0){
			float*pRight = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
			cell[x_identifier_] = -pRight[x_identifier_];
			cell[y_identifier_] = -pRight[y_identifier_];
			cell[z_identifier_] = -pRight[z_identifier_];
		}else if(x_iter + 1 == size.width_){
			float*pLeft = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
			cell[x_identifier_] = -pLeft[x_identifier_];
			cell[y_identifier_] = -pLeft[y_identifier_];
			cell[z_identifier_] = -pLeft[z_identifier_];
		}
		if(y_iter == 0){
			float*pUp = input + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
			cell[x_identifier_] = -pUp[x_identifier_];
			cell[y_identifier_] = -pUp[y_identifier_];
			cell[z_identifier_] = -pUp[z_identifier_];
		}else if(y_iter + 1 == size.height_){
			float*pDown = input + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
			cell[x_identifier_] = -pDown[x_identifier_];
			cell[y_identifier_] = -pDown[y_identifier_];
			cell[z_identifier_] = -pDown[z_identifier_];
		}
		if(z_iter == 0){
			float*pBottom = input + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
			cell[x_identifier_] = -pBottom[x_identifier_];
			cell[y_identifier_] = -pBottom[y_identifier_];
			cell[z_identifier_] = -pBottom[z_identifier_];
		}else if(z_iter + 1 == size.depth_){
			float*pTop = input + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
			cell[x_identifier_] = -pTop[x_identifier_];
			cell[y_identifier_] = -pTop[y_identifier_];
			cell[z_identifier_] = -pTop[z_identifier_];
		}
	}
}

extern "C"
void cuda_fluid_boundaries(void *input, Size size){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_boundaries<<<Dg,Db>>>((float *)input, size);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_fluid_project() failed to launch error = %d\n", error);
	}
}
#endif