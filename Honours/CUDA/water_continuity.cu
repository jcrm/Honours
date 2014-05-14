#ifndef _WATER_CONTINUITY_CUDA_
#define _WATER_CONTINUITY_CUDA_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "math.h"

#include "../Source/CUDA/cuda_header.h"

__global__ void cuda_kernel_water(float *input, float *input_two, float *input_three, Size size){
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size.depth_; z_iter++){
		float* water = input_two + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RG * x_iter);
		float* rain = input_three + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RG * x_iter);
		float* thermo = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RG * x_iter);

		float qv = water[qv_identifier_];
		float qc = water[qc_identifier_];
		float qr = rain[qr_identifier_];
		float theta = thermo[theta_identifier_];

		float K=beta*qc*qr;
		float F=-V*qr/b1;
		float A = 0;
		if(qc>aT){
			A=alpha*(qc-aT);
		}
		float temperature = theta * powf((p0/pressure),k);
		float est = (es0/pressure)*expf(a*(temperature-273.f)/(temperature-b));
		float pres_minus_est = powf((pressure-est),2);
		float C = (-est*W*epsilon*z_alt) * ((g*pressure/(R*T*pres_minus_est)) + (-a*gamma)*((273.f-b)/pow((temperature-b),2))*((1.f/pressure)+(est/pres_minus_est)));
		C *= 1000.f;
		qv = (-C/W);
		qc = ((-A-K+C)/W);
		qr = ((A+K+F)/W);

		water[qv_identifier_] = qv;
		water[qc_identifier_] = qc;
		rain[qr_identifier_] = qr;
		rain[F_identifier_] = F;
	}
}
extern "C"
void cuda_fluid_water(void *input, void *input_two, void* input_three, Size size){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	//dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_water<<<Dg,Db>>>((float *)input, (float *)input_two, (float*)input_three, size);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_jacobi() failed to launch error = %d\n", error);
	}
}
#endif