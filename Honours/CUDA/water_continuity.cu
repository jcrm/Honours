#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#define T0 295.f
#define gamma 6.f/1000.f
#define p0 100000.f
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
#define es0 100.f*3.8f/0.62197f

__global__ void cuda_kernel_water(unsigned char *input, float3 size_WHD, size_t pitch, size_t pitch_slice, int pressure_index, int divergence_index){  
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size_WHD.z; ++z_iter){
		if(x_iter +1 < size_WHD.x && x_iter - 1 > 0){
			if(y_iter + 1 < size_WHD.y && y_iter - 1 > 0){
				if(z_iter + 1 < size_WHD.z && z_iter - 1 > 0){

% Height range (m)
z=[1000 10000];
% Initial conditions
qc0=0; qr0=0;
% SOLVE
[z,q]=ode45(’qdotw’,z,[qc0; qr0]);
qv=qs(z); qc=q(:,1); qr=q(:,2); qt=qv+qc+qr;
% Recalculate conversion processes (for plotting)
[T0,gamma,p0,W,aT,alpha,beta,b,V]=consts; % Load constants
K=beta*qc.*qr*60*1000; F=-V*qr/b1*60*1000;
A=alpha*(qc-aT)*60*1000; A(find(qc<aT))=0;
C=C(z)*60*1000;


function qdot=qdot(z,q)
[T0,gamma,p0,W,aT,alpha,beta,b,V]=consts;
qc=q(1); qr=q(2);
K=beta*qc*qr;
F=-V*qr/b1;
if qc>aT
A=alpha*(qc-aT);
else
A=0;
end
qdot=[(-A-K+C(z))/W; (A+K+F)/W];

function C=C(z)
% Condensation rate (s^-1)
% C=-W dqs/dz

T=T0-gamma*z;
p=p0*(T/T0).^(g/R/gamma);
C=-es(T).*( -a*gamma*(273-b)./(T-b).^2.*(1./p + es(T)./(p-es(T)).^2)+ g*p/R./T./(p-es(T)).^2 )*W*epsilon;

function es=es(T)
% Saturation vapor pressure (Pa)
% es=es(T)
% T in Kelvin
es=es0*exp(a*(T-273)./(T-b));
				}
			}
		}
	}
}
extern "C"
void cuda_fluid_water(void *pressuredivergence, float3 size_WHD, size_t pitch, size_t pitch_slice, int pressure_index, int divergence_index){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size_WHD.x+Db.x-1)/Db.x, (size_WHD.y+Db.y-1)/Db.y);

	cuda_kernel_water<<<Dg,Db>>>((unsigned char *)pressuredivergence,size_WHD, pitch, pitch_slice, pressure_index, divergence_index);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_jacobi() failed to launch error = %d\n", error);
	}
}