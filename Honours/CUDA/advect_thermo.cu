#ifndef _ADVECT_THERMO_CUDA_
#define _ADVECT_THERMO_CUDA_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "../Source/CUDA/cuda_header.h"

__global__ void cuda_kernel_advect_thermo(float *input, float *velocity, Size size, float ambient_temp, float delta_time){ 
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size.depth_; z_iter++){ 
		float *cellVelo = velocity + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
		float3 pos = {x_iter, y_iter, z_iter};
		float3 cell_velo = {cellVelo[x_identifier_], cellVelo[y_identifier_], cellVelo[z_identifier_]};
		pos.x = pos.x - (time_step * delta_time * cell_velo.x);
		pos.y = pos.y - (time_step * delta_time * cell_velo.y);
		pos.z = pos.z - (time_step * delta_time * cell_velo.z);
			
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

		float *field_left_up = input + (location.z*size.pitch_slice_) + (location.y*size.pitch_) + (PIXEL_FMT_SIZE_RG * location.x);
		float *field_left_down = input + (location.z*size.pitch_slice_) + (location_two.y*size.pitch_) + (PIXEL_FMT_SIZE_RG * location.x);
		float *field_right_up = input + (location.z*size.pitch_slice_) + (location.y*size.pitch_) + (PIXEL_FMT_SIZE_RG * location_two.x);
		float *field_right_down = input + (location.z*size.pitch_slice_) + (location_two.y*size.pitch_) + (PIXEL_FMT_SIZE_RG * location_two.x);

		float *field_left_up_back = input + (location_two.z*size.pitch_slice_) + (location.y*size.pitch_) + (PIXEL_FMT_SIZE_RG * location.x);
		float *field_left_down_back = input + (location_two.z*size.pitch_slice_) + (location_two.y*size.pitch_) + (PIXEL_FMT_SIZE_RG * location.x);
		float *field_right_up_back = input + (location_two.z*size.pitch_slice_) + (location.y*size.pitch_) + (PIXEL_FMT_SIZE_RG * location_two.x);
		float *field_right_down_back = input + (location_two.z*size.pitch_slice_) + (location_two.y*size.pitch_) + (PIXEL_FMT_SIZE_RG * location_two.x);

		float temp_1 = field_left_up[theta_identifier_] + field_left_down[theta_identifier_] +  field_right_up[theta_identifier_] + field_right_down[theta_identifier_];
		float temp_2 = field_left_up_back[theta_identifier_] + field_left_down_back[theta_identifier_] + field_right_down_back[theta_identifier_] + field_right_up_back[theta_identifier_];

		temp_1 /=4.f;
		temp_2 /=4.f;
			
		float*output_thermo = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RG * x_iter);
		output_thermo[theta_advect_identifier_] = (temp_1 + temp_2)/2.f;
		
		if(x_iter == 0){
			output_thermo[theta_identifier_] = T0;
		}else if(x_iter + 1 == size.width_){
			output_thermo[theta_identifier_] = T0;
		}else if (y_iter == 0){
			if((x_iter >= size.width_/4.f  && x_iter <= 3*(size.width_/4.f)) && (z_iter <= size.width_/4.f  || x_iter >= 3*(size.width_/4.f))){
				output_thermo[theta_identifier_] = 310.f;
			}else{
				output_thermo[theta_identifier_] = 270.f;

			}
		}else if (y_iter + 1 == size.height_){
			output_thermo[theta_identifier_] = T0;
		}else if (z_iter == 0){
			output_thermo[theta_identifier_] = T0;
		}else if (z_iter + 1 == size.depth_){
			output_thermo[theta_identifier_] = T0;
		}
	}
}

extern "C"
void cuda_fluid_advect_thermo(void *input, void* velocity, Size size, float ambient_temp, float delta_time){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_advect_thermo<<<Dg,Db>>>((float*)input, (float*)velocity, size, ambient_temp, delta_time);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_advect() failed to launch error = %d\n", error);
	}
}
#endif