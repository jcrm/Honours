#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include "math.h"
#include "../Source/CUDA/cuda_header.h"

#define T0 295.f
#define gamma 6.f/1000.f
#define p0 100000
#define aT 5e-4
#define alpha 1e-3
#define beta 2.f
#define b1 1000.f
#define V 4.f
#define W 8.f
#define g 9.8f
#define R 287.f
#define epsilon 18.02f/29.87f
#define a 17.27f
#define b 35.86f
#define es0 100.f*3.8f
#define z_alt 1000
#define latent_heat 2.501
#define cp 1005
#define k 0.286
#define T T0-gamma*z_alt

#define PIXEL_FMT_SIZE 4
#define qv_identifier_ 0
#define qc_identifier_ 1
#define qr_identifier_ 2
#define F_identifier_ 3
#define theta_identifier_ 2
#define theta_advect_identifier_ 3
#define time_step 1.f

__global__ void cuda_kernel_water_thermo(unsigned char *input, unsigned char *input_two, Size size){
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size.depth_; ++z_iter){
		if(x_iter +1 < size.width_ && x_iter - 1 >= 0){
			if(y_iter + 1 < size.height_ && y_iter - 1 >= 0){
				if(z_iter + 1 < size.depth_ && z_iter - 1 >= 0){

					unsigned char* water = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE * x_iter);
					unsigned char* thermo = input_two + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE * x_iter);

					float qv = water[qv_identifier_];
					float qc = water[qc_identifier_];
					float qr = water[qr_identifier_];
					float theta = thermo[theta_identifier_];
					float theta_advect = thermo[theta_advect_identifier_];

					float K=beta*qc*qr;
					float F=-V*qr/b1;
					float A = 0;
					if(qc>aT){
						A=alpha*(qc-aT);
					}
					float p=p0*pow((T/T0),(g/R/gamma));
					float est = (es0/p)*exp(a*(theta-273)/(theta-b));
					float C = g*p/(R*T*pow((p-est),2));
					C += (-a*gamma)*((273-b)/pow((theta-b),2))*((1.f/p)+(est/pow((p-est),2)));
					C *= -est*W*epsilon;
					
					qv = -C/W;
					qc = (-A-K+C)/W;
					qr = (A+K+F)/W;

					theta = theta_advect + ((latent_heat / (cp * powf(p/p0,k))) * (time_step * C));

					water[qv_identifier_] = qv;
					water[qc_identifier_] = qc;
					water[qr_identifier_] = qr;
					water[F_identifier_] = F;
					thermo[theta_identifier_] = theta;
					thermo[theta_advect_identifier_] = theta_advect;
				}
			}
		}
	}
}
extern "C"
void cuda_fluid_water_thermo(unsigned char *input, unsigned char *input_two, Size size){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	//dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_water_thermo<<<Dg,Db>>>((unsigned char *)input, (unsigned char *)input_two, size);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_jacobi() failed to launch error = %d\n", error);
	}
}