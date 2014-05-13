#ifndef _JACOBI_CUDA_
#define _JACOBI_CUDA_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "../Source/CUDA/cuda_header.h"

__global__ void cuda_kernel_jacobi(float *pressuredivergence, Size size, int pressure_index, int divergence_index){
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;
	for(int i = 0; i < 1; i++){
		if(i%2 == 0){
			pressure_index = 0;
			divergence_index = 1;
		}else{
			pressure_index = 1;
			divergence_index = 0;
		}
		for(z_iter = 0; z_iter < size.depth_; ++z_iter){
			float sum = 0.f;
			float* cellPressure = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);

			if(x_iter == 0 && y_iter == 0 && z_iter == 0){
				float*pRight = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
				float*pUp = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pBottom = pressuredivergence + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pRight[pressure_index] + pUp[pressure_index] + pBottom[pressure_index];
			}else if(x_iter == 0 && y_iter == 0 && z_iter + 1 == size.depth_){
				float*pRight = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
				float*pUp = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pTop = pressuredivergence + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pRight[pressure_index] + pUp[pressure_index] + pTop[pressure_index];
			}else if(x_iter == 0 && y_iter + 1 == size.height_ && z_iter == 0){
				float*pRight = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
				float*pDown = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pBottom = pressuredivergence + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pRight[pressure_index] + pDown[pressure_index] + pBottom[pressure_index];
			}else if(x_iter == 0 && y_iter + 1 == size.height_ && z_iter + 1 == size.depth_){
				float*pRight = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
				float*pDown = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pTop = pressuredivergence + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pRight[pressure_index] + pDown[pressure_index] + pTop[pressure_index];
			}else if(x_iter + 1 == size.width_ && y_iter == 0 && z_iter == 0){
				float*pLeft = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
				float*pUp = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pBottom = pressuredivergence + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pLeft[pressure_index] + pUp[pressure_index] + pBottom[pressure_index];
			}else if(x_iter + 1 == size.width_ && y_iter == 0 && z_iter + 1 == size.depth_){
				float*pLeft = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
				float*pUp = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pTop = pressuredivergence + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pLeft[pressure_index] + pUp[pressure_index] + pTop[pressure_index];
			}else if(x_iter + 1 == size.width_ && y_iter + 1 == size.height_ && z_iter == 0){
				float*pLeft = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
				float*pDown = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pBottom = pressuredivergence + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pLeft[pressure_index] + pDown[pressure_index] + pBottom[pressure_index];
			}else if(x_iter + 1 == size.width_ && y_iter + 1 == size.height_ && z_iter + 1 == size.depth_){
				float*pLeft = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
				float*pDown = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pTop = pressuredivergence + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pLeft[pressure_index] + pDown[pressure_index] + pTop[pressure_index];
			}else if(x_iter == 0 && (y_iter > 0 && y_iter + 1 <size.height_) && (z_iter > 0 && z_iter + 1 < size.depth_)){
				float*pRight = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
				float*pDown = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pUp = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pTop = pressuredivergence + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				float*pBottom = pressuredivergence + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pRight[pressure_index] + pUp[pressure_index] + pDown[pressure_index] + pTop[pressure_index] + pBottom[pressure_index];
			}else if(x_iter == size.width_ && (y_iter > 0 && y_iter + 1 < size.height_) && (z_iter > 0 && z_iter + 1 < size.depth_)){
				float*pLeft = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
				float*pDown = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pUp = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pTop = pressuredivergence + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				float*pBottom = pressuredivergence + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pLeft[pressure_index] + pUp[pressure_index] + pDown[pressure_index] + pTop[pressure_index] + pBottom[pressure_index];
			}else if((x_iter > 0 && x_iter + 1 < size.width_) && y_iter == 0 && (z_iter > 0 && z_iter + 1 < size.depth_)){
				float*pLeft = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
				float*pRight = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
				float*pUp = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pTop = pressuredivergence + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				float*pBottom = pressuredivergence + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pLeft[pressure_index] + pRight[pressure_index] + pUp[pressure_index] + pTop[pressure_index] + pBottom[pressure_index];
			}else if((x_iter > 0 && x_iter + 1 < size.width_) && y_iter == size.height_ && (z_iter > 0 && z_iter + 1 < size.depth_)){
				float*pLeft = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
				float*pRight = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
				float*pDown = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pTop = pressuredivergence + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				float*pBottom = pressuredivergence + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pLeft[pressure_index] + pRight[pressure_index] + pDown[pressure_index] + pTop[pressure_index] + pBottom[pressure_index];
			}else if((x_iter > 0 && x_iter + 1 < size.width_) && (y_iter > 0 && y_iter +1 < size.height_) && z_iter == 0){
				float*pLeft = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
				float*pRight = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
				float*pDown = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pUp = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pBottom = pressuredivergence + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pLeft[pressure_index] + pRight[pressure_index] + pUp[pressure_index] + pDown[pressure_index] + pBottom[pressure_index];
			}else if((x_iter > 0 && x_iter + 1 < size.width_) && (y_iter > 0 && y_iter + 1 < size.height_) && z_iter == size.depth_){
				float*pLeft = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
				float*pRight = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
				float*pDown = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pUp = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pTop = pressuredivergence + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pLeft[pressure_index] + pRight[pressure_index] + pUp[pressure_index] + pDown[pressure_index] + pTop[pressure_index];
			}else if(x_iter == 0 && y_iter == 0 && (z_iter > 0 && z_iter + 1 < size.depth_)){
				float*pRight = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
				float*pUp = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pTop = pressuredivergence + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				float*pBottom = pressuredivergence + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pRight[pressure_index] + pUp[pressure_index] + pTop[pressure_index] + pBottom[pressure_index];
			}else if(x_iter == 0 && y_iter == size.height_ && (z_iter > 0 && z_iter + 1 < size.depth_)){
				float*pRight = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
				float*pDown = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pTop = pressuredivergence + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				float*pBottom = pressuredivergence + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pRight[pressure_index] +pDown[pressure_index] + pTop[pressure_index] + pBottom[pressure_index];
			}else if(x_iter == 0 && (y_iter > 0 && y_iter +1 < size.height_) && z_iter == 0){
				float*pRight = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
				float*pDown = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pUp = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pBottom = pressuredivergence + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pRight[pressure_index] + pUp[pressure_index] + pDown[pressure_index] + pBottom[pressure_index];
			}else if(x_iter == 0 && (y_iter > 0 && y_iter +1 < size.height_) && z_iter == size.depth_){
				float*pRight = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
				float*pDown = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pUp = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pTop = pressuredivergence + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pRight[pressure_index] + pUp[pressure_index] + pDown[pressure_index] + pTop[pressure_index];
			}else if(x_iter + 1 == size.width_ && y_iter == 0 && (z_iter > 0 && z_iter + 1 < size.depth_)){
				float*pLeft = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
				float*pUp = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pTop = pressuredivergence + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				float*pBottom = pressuredivergence + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pLeft[pressure_index] + pUp[pressure_index] + pTop[pressure_index] + pBottom[pressure_index];
			}else if(x_iter == size.width_ && y_iter == size.height_ && (z_iter > 0 && z_iter + 1 < size.depth_)){
				float*pLeft = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
				float*pUp = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pTop = pressuredivergence + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				float*pBottom = pressuredivergence + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pLeft[pressure_index] + pUp[pressure_index] + pTop[pressure_index] + pBottom[pressure_index];
			}else if(x_iter == size.width_ && (y_iter > 0 && y_iter +1 < size.height_) && z_iter == 0){
				float*pLeft = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
				float*pDown = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pUp = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pBottom = pressuredivergence + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pLeft[pressure_index] + pUp[pressure_index] + pDown[pressure_index] + pBottom[pressure_index];
			}else if(x_iter == size.width_ && (y_iter > 0 && y_iter +1 < size.height_) && z_iter == size.depth_){
				float*pLeft = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
				float*pDown = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pUp = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pTop = pressuredivergence + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pLeft[pressure_index] + pUp[pressure_index] + pDown[pressure_index] + pTop[pressure_index];
			}else if((x_iter > 0 && x_iter + 1 < size.width_) && y_iter == 0 && z_iter == 0){
				float*pLeft = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
				float*pRight = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
				float*pUp = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pBottom = pressuredivergence + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pLeft[pressure_index] + pRight[pressure_index] + pUp[pressure_index] + pBottom[pressure_index];
			}else if((x_iter > 0 && x_iter + 1 < size.width_)&& y_iter == size.height_ && z_iter == 0){
				float*pLeft = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
				float*pRight = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
				float*pDown = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pBottom = pressuredivergence + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pLeft[pressure_index] + pRight[pressure_index] + pDown[pressure_index] + pBottom[pressure_index];
			}else if((x_iter > 0 && x_iter + 1 < size.width_) && y_iter == 0 && z_iter == 0){
				float*pLeft = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
				float*pRight = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
				float*pUp = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pBottom = pressuredivergence + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pLeft[pressure_index] + pRight[pressure_index] + pUp[pressure_index] + pBottom[pressure_index];
			}else if((x_iter > 0 && x_iter + 1 < size.width_) && y_iter == 0 && z_iter == size.depth_){
				float*pLeft = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
				float*pRight = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
				float*pUp = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
				float*pTop = pressuredivergence + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
				sum = pLeft[pressure_index] + pRight[pressure_index] + pUp[pressure_index] + pTop[pressure_index];
			}else if(x_iter +1 < size.width_ && x_iter - 1 >= 0){
				if(y_iter + 1 < size.height_ && y_iter - 1 >= 0){
					if(z_iter + 1 < size.depth_ && z_iter - 1 >= 0){
						float*pLeft = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
						float*pRight = pressuredivergence + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
						float*pDown = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
						float*pUp = pressuredivergence + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
						float*pTop = pressuredivergence + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
						float*pBottom = pressuredivergence + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
						sum = pLeft[pressure_index] + pRight[pressure_index] + pUp[pressure_index] + pDown[pressure_index] + pTop[pressure_index] + pBottom[pressure_index];
					}
				}
			}
			// Get the divergence at the current cell.  
			float dCentre = cellPressure[divergence_index];
			float value = (sum - (4*cellPressure[pressure_index]) - dCentre);
			// Compute the new pressure value for the center cell.
			cellPressure[pressure_index] = value/6.f;
		}
	}
}
extern "C"
void cuda_fluid_jacobi(void *input, Size size, int pressure_index, int divergence_index){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_jacobi<<<Dg,Db>>>((float *)input, size, pressure_index, divergence_index);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_jacobi() failed to launch error = %d\n", error);
	}
}
#endif