#ifndef ADVECT_VELOCITY_CUDA_
#define ADVECT_VELOCITY_CUDA_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "../Source/CUDA/cuda_header.h"

//output velocity derrivitive teture //input velcoity texutre
__global__ void cuda_kernel_advect_velocity(float *output, float*input, Size size, float3 x_left, float3 x_right,  float3 z_front, float3 z_back){ 
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size.depth_; z_iter++){ 
		float *cellVelocity = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
		
		float3 pos = {x_iter, y_iter, z_iter};
		float3 cell_velocity = {cellVelocity[x_identifier_], cellVelocity[y_identifier_], cellVelocity[z_identifier_]};
		pos.x = pos.x - (time_step * cell_velocity.x);
		pos.y = pos.y - (time_step * cell_velocity.y);
		pos.z = pos.z - (time_step * cell_velocity.z);
			
		int3 location = {pos.x,pos.y, pos.z};
		location.x = location.x < 0 ? 0 : location.x;
		location.y = location.y < 0 ? 0 : location.y;
		location.z = location.z < 0 ? 0 : location.z;
		location.x = location.x >= size.width_ ? location.x = size.width_ - 1 : location.x;
		location.y = location.y >= size.height_ ? location.y = size.height_ - 1 : location.y;
		location.z = location.z >= size.depth_ ? location.z = size.depth_ - 1 : location.z;

		int3 location_two = {location.x+1,location.y+1, location.z+1};
		location_two.x = location_two.x >= size.width_ ? location_two.x = size.width_ - 1 : location_two.x;
		location_two.y = location_two.y >= size.height_ ? location_two.y = size.height_ - 1 : location_two.y;
		location_two.z = location_two.z >= size.depth_ ? location_two.z = size.depth_ - 1 : location_two.z;

		float *field_left_up = input + (location.z*size.pitch_slice_) + (location.y*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * location.x);
		float *field_left_down = input + (location.z*size.pitch_slice_) + (location_two.y*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * location.x);
		float *field_right_up = input + (location.z*size.pitch_slice_) + (location.y*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * location_two.x);
		float *field_right_down = input + (location.z*size.pitch_slice_) + (location_two.y*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * location_two.x);

		float *field_left_up_back = input + (location_two.z*size.pitch_slice_) + (location.y*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * location.x);
		float *field_left_down_back = input + (location_two.z*size.pitch_slice_) + (location_two.y*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * location.x);
		float *field_right_up_back = input + (location_two.z*size.pitch_slice_) + (location.y*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * location_two.x);
		float *field_right_down_back = input + (location_two.z*size.pitch_slice_) + (location_two.y*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * location_two.x);

		float temp_X_1 = field_left_up[x_identifier_] + field_left_down[x_identifier_] + field_right_up[x_identifier_] + field_right_down[x_identifier_];
		float temp_Y_1 = field_left_up[y_identifier_] + field_left_down[y_identifier_] + field_right_up[y_identifier_] + field_right_down[y_identifier_];
		float temp_Z_1 = field_left_up[z_identifier_] + field_left_down[z_identifier_] + field_right_up[z_identifier_] + field_right_down[z_identifier_];

		float temp_X_2 = field_left_up_back[x_identifier_] + field_left_down_back[x_identifier_] + field_right_up_back[x_identifier_] + field_right_down_back[x_identifier_];
		float temp_Y_2 = field_left_up_back[y_identifier_] + field_left_down_back[y_identifier_] + field_right_up_back[y_identifier_] + field_right_down_back[x_identifier_];
		float temp_Z_2 = field_left_up_back[z_identifier_] + field_left_down_back[z_identifier_] + field_right_up_back[z_identifier_] + field_right_down_back[x_identifier_];

		temp_X_1 /=4.f;
		temp_Y_1 /=4.f;
		temp_Z_1 /=4.f;

		temp_X_2 /=4.f;
		temp_Y_2 /=4.f;
		temp_Z_2 /=4.f;
			
		float*output_velocity = output + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
		output_velocity[x_identifier_] = (temp_X_1 + temp_X_2)/2.f;
		output_velocity[y_identifier_] = (temp_Y_1 + temp_Y_2)/2.f;
		output_velocity[z_identifier_] = (temp_Z_1 + temp_Z_2)/2.f;

		if(y_iter == 0){
			cellVelocity[x_identifier_] = 0.f;
			cellVelocity[y_identifier_] = 0.f;
			cellVelocity[z_identifier_] = 0.f;
		}else if (y_iter + 1 == size.height_){
			float*down = output + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
			output_velocity[x_identifier_] = -down[x_identifier_];
			output_velocity[y_identifier_] = -down[y_identifier_];
			output_velocity[z_identifier_] = -down[z_identifier_];
		}else if (x_iter == 0){
			cellVelocity[x_identifier_] = x_left.x;
			cellVelocity[y_identifier_] = x_left.y;
			cellVelocity[z_identifier_] = x_left.z;
		}else if (x_iter + 1 == size.width_){
			cellVelocity[x_identifier_] = x_right.x;
			cellVelocity[y_identifier_] = x_right.y;
			cellVelocity[z_identifier_] = x_right.z;
		}else if (z_iter == 0){
			cellVelocity[x_identifier_] = z_front.x;
			cellVelocity[y_identifier_] = z_front.y;
			cellVelocity[z_identifier_] = z_front.z;
		}else if (z_iter + 1 == size.depth_){
			cellVelocity[x_identifier_] = z_back.x;
			cellVelocity[y_identifier_] = z_back.y;
			cellVelocity[z_identifier_] = z_back.z;
		}
	}
	
}

extern "C"
void cuda_fluid_advect_velocity(void *output, void *input, Size size, float3 x_left, float3 x_right,  float3 z_front, float3 z_back){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_advect_velocity<<<Dg,Db>>>((float *)output, (float *)input, size, x_left, x_right, z_front, z_back);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_advect() failed to launch error = %d\n", error);
	}
}
#endif