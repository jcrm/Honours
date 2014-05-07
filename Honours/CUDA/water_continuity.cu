#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include "math.h"
#include "../Source/CUDA/cuda_header.h"

__global__ void cuda_kernel_water_thermo(float *input, float *input_two, float *input_tree, Size size){
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size.depth_; ++z_iter){
		if(x_iter +1 < size.width_ && x_iter - 1 >= 0){
			if(y_iter + 1 < size.height_ && y_iter - 1 >= 0){
				if(z_iter + 1 < size.depth_ && z_iter - 1 >= 0){

					float* water = input_two + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RG * x_iter);
					float* rain = input_tree + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RG * x_iter);
					float* thermo = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RG * x_iter);

					float qv = water[qv_identifier_];
					float qc = water[qc_identifier_];
					float qr = rain[qr_identifier_];
					float theta =thermo[theta_identifier_];
					float theta_advect = thermo[theta_advect_identifier_];

					float K=beta*qc*qr*z_alt;
					float F=-V*qr/b1*z_alt;
					if(F != 0.f){
						F *= 1.f;
					}
					float A = 0;
					if(qc>aT){
						A=alpha*(qc-aT)*z_alt;
					}
					
					float p=p0*pow((T/T0),(g/R/gamma));
					float TEMP = theta * powf((p0/p),k);
					float est = (es0/p)*exp(a*(TEMP-273)/(TEMP-b));
					float C = g*p/(R*T*pow((p-est),2));
					C += (-a*gamma)*((273-b)/pow((TEMP-b),2))*((1.f/p)+(est/pow((p-est),2)));
					C *= -est*W*epsilon*z_alt;
					
					qv = -C/W;
					qc = (-A-K+C)/W;
					qr = (A+K+F)/W;
					float temp = (latent_heat / (cp * powf(p/p0,k)));
					temp *= C * time_step;
					temp = theta_advect - temp;
					theta = temp;

					water[qv_identifier_] = qv;
					water[qc_identifier_] = qc;
					rain[qr_identifier_] = qr;
					rain[F_identifier_] = F;
					float themp_F = rain[F_identifier_];
					thermo[theta_identifier_] = theta;
					thermo[theta_advect_identifier_] = theta_advect;
				}
			}
		}
	}
}
extern "C"
void cuda_fluid_water_thermo(void *input, void *input_two, void* input_three, Size size){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	//dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_water_thermo<<<Dg,Db>>>((float *)input, (float *)input_two, (float*)input_three, size);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_jacobi() failed to launch error = %d\n", error);
	}
}