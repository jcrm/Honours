#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

__global__ void cuda_kernel_divergence(unsigned char* divergence, unsigned char* velocityInput,float3 size_WHD, size_t pitch, size_t pitch_slice, int divergence_index){
	int xIter = blockIdx.x*blockDim.x + threadIdx.x;
	int yIter = blockIdx.y*blockDim.y + threadIdx.y;
	int zIter = 0;

	for(zIter = 0; zIter < size_WHD.z; ++zIter){ 
		// Get velocity values from neighboring cells.
		unsigned char *fieldLeft = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
		unsigned char *fieldRight = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));

		unsigned char *fieldUp = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
		unsigned char *fieldDown = velocityInput + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);

		unsigned char *fieldTop = velocityInput + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
		unsigned char *fieldBottom = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter);

		unsigned char* cellDivergence = divergence + (zIter*pitch_slice) + (yIter*pitch) + (4*xIter);
		// Compute the velocity's divergence using central differences.  
		cellDivergence[divergence_index] =  0.5f * ((fieldRight[0] - fieldLeft[0])+  
				(fieldTop[1] - fieldBottom[1]) + (fieldUp[2] - fieldDown[2])); 
		if((xIter - 1 < 0) && (yIter - 1 < 0) && (zIter - 1 < 0)){

			fieldRight = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			fieldUp = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			fieldBottom = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellDivergence[divergence_index] = 0.5f * ((fieldRight[0]) + ( - fieldBottom[1]) + (fieldUp[2])); 

		}else if((xIter + 1 ==size_WHD.x) && (yIter - 1 < 0) && (zIter - 1 < 0)){

			fieldLeft = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			fieldUp = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			fieldBottom = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pLeft[pressure_index] + pBottom[pressure_index] + pUp[pressure_index] - dCentre)/3.f;

		}else if((xIter - 1 < 0) && (yIter + 1 ==size_WHD.y) && (zIter - 1 < 0)){

			pRight = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			pDown = pressuredivergence + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			pBottom = pressuredivergence + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pRight[pressure_index] + pBottom[pressure_index] + pDown[pressure_index] - dCentre)/3.f;

		}else if((xIter + 1 ==size_WHD.x) && (yIter + 1 ==size_WHD.y) && (zIter - 1 < 0)){

			pLeft = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			pDown = pressuredivergence + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			pBottom = pressuredivergence + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pLeft[pressure_index] + pBottom[pressure_index] + pDown[pressure_index] - dCentre)/3.f;

		}else if((xIter - 1 < 0) && (yIter - 1 < 0) && (zIter + 1 ==size_WHD.y)){

			pRight = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			pUp = pressuredivergence + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pRight[pressure_index] + pTop[pressure_index] + pUp[pressure_index] - dCentre)/3.f;

		}else if((xIter + 1 ==size_WHD.x) && (yIter - 1 < 0) && (zIter + 1 ==size_WHD.y)){

			pLeft = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			pUp = pressuredivergence + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pLeft[pressure_index] + pTop[pressure_index] + pUp[pressure_index] - dCentre)/3.f;

		}else if((xIter - 1 < 0) && (yIter + 1 ==size_WHD.y) && (zIter + 1 ==size_WHD.y)){

			pRight = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			pDown = pressuredivergence + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pRight[pressure_index] + pTop[pressure_index] + pDown[pressure_index] - dCentre)/3.f;

		}else if((xIter + 1 ==size_WHD.x) && (yIter + 1 ==size_WHD.y) && (zIter + 1 ==size_WHD.y)){

			pLeft = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			pDown = pressuredivergence + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pLeft[pressure_index] + pTop[pressure_index] + pDown[pressure_index] - dCentre)/3.f;

		}else if((xIter - 1 < 0) && (yIter - 1 < 0)){

			pRight = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			pUp = pressuredivergence + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			pBottom = pressuredivergence + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pRight[pressure_index] + pBottom[pressure_index] + pTop[pressure_index] + pUp[pressure_index] - dCentre)/4.f;

		}else if((xIter + 1 ==size_WHD.x) && (yIter - 1 < 0)){

			pLeft = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			pUp = pressuredivergence + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			pBottom = pressuredivergence + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pLeft[pressure_index] + pBottom[pressure_index] + pTop[pressure_index] + pUp[pressure_index] - dCentre)/4.f;

		}else if((xIter - 1 < 0) && (yIter + 1 ==size_WHD.y)){

			pRight = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			pDown = pressuredivergence + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			pBottom = pressuredivergence + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pRight[pressure_index] + pBottom[pressure_index] + pTop[pressure_index] + pDown[pressure_index] - dCentre)/4.f;

		}else if((xIter + 1 ==size_WHD.x) && (yIter + 1 ==size_WHD.y)){

			pLeft = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			pDown = pressuredivergence + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			pBottom = pressuredivergence + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pLeft[pressure_index] + pBottom[pressure_index] + pTop[pressure_index] + pDown[pressure_index] - dCentre)/4.f;

		}else if((xIter - 1 < 0) && (zIter - 1 < 0)){

			pRight = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			pDown = pressuredivergence + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			pUp = pressuredivergence + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			pBottom = pressuredivergence + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pRight[pressure_index] + pBottom[pressure_index] + pUp[pressure_index] + pDown[pressure_index] - dCentre)/4.f;

		}else if((xIter + 1 ==size_WHD.x) && (zIter - 1 < 0)){

			pLeft = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			pDown = pressuredivergence + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			pUp = pressuredivergence + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			pBottom = pressuredivergence + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pLeft[pressure_index] + pBottom[pressure_index] + pUp[pressure_index] + pDown[pressure_index] - dCentre)/4.f;

		}else if((xIter - 1 < 0) && (zIter + 1 ==size_WHD.y)){

			pRight = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			pDown = pressuredivergence + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			pUp = pressuredivergence + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pRight[pressure_index] + pTop[pressure_index] + pUp[pressure_index] + pDown[pressure_index] - dCentre)/4.f;

		}else if((xIter + 1 ==size_WHD.x) && (zIter + 1 ==size_WHD.y)){

			pLeft = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			pDown = pressuredivergence + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			pUp = pressuredivergence + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pLeft[pressure_index] + pTop[pressure_index] + pUp[pressure_index] + pDown[pressure_index] - dCentre)/4.f;

		}else if((yIter - 1 < 0) && (zIter - 1 < 0)){

			pLeft = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			pRight = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			pUp = pressuredivergence + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			pBottom = pressuredivergence + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pLeft[pressure_index] + pRight[pressure_index] + pBottom[pressure_index] + pUp[pressure_index] - dCentre)/4.f;

		}else if((yIter + 1 ==size_WHD.y) && (zIter - 1 < 0)){

			pLeft = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			pRight = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			pDown = pressuredivergence + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			pBottom = pressuredivergence + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pLeft[pressure_index] + pRight[pressure_index] + pBottom[pressure_index] + pDown[pressure_index] - dCentre)/4.f;

		}else if((yIter - 1 < 0) && (zIter + 1 ==size_WHD.y)){
			pLeft = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			pRight = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			pUp = pressuredivergence + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pLeft[pressure_index] + pRight[pressure_index]+ pTop[pressure_index] + pUp[pressure_index] - dCentre)/4.f;
		}else if((yIter + 1 ==size_WHD.y) && (zIter + 1 ==size_WHD.y)){

			pLeft = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			pRight = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			pDown = pressuredivergence + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pLeft[pressure_index] + pRight[pressure_index] + pTop[pressure_index] + pDown[pressure_index] - dCentre)/4.f;

		}else if(xIter - 1 < 0){

			pRight = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			pDown = pressuredivergence + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			pUp = pressuredivergence + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			pBottom = pressuredivergence + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pRight[pressure_index] + pBottom[pressure_index] + pTop[pressure_index] + pUp[pressure_index] + pDown[pressure_index] - dCentre)/5.f;

		}else if(xIter + 1 ==size_WHD.x){

			pLeft = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			pDown = pressuredivergence + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			pUp = pressuredivergence + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			pBottom = pressuredivergence + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pLeft[pressure_index] + pBottom[pressure_index] + pTop[pressure_index] + pUp[pressure_index] + pDown[pressure_index] - dCentre)/5.f;

		}else if(yIter - 1 < 0){

			pLeft = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			pRight = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			pUp = pressuredivergence + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			pBottom = pressuredivergence + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pLeft[pressure_index] + pRight[pressure_index] + pBottom[pressure_index] + pTop[pressure_index] + pUp[pressure_index] - dCentre)/5.f;

		}else if(yIter + 1 ==size_WHD.y){

			pLeft = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			pRight = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			pDown = pressuredivergence + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			pBottom = pressuredivergence + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pLeft[pressure_index] + pRight[pressure_index] + pBottom[pressure_index] + pTop[pressure_index] + pDown[pressure_index] - dCentre)/5.f;

		}else if(zIter - 1 < 0){

			pLeft = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			pRight = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			pDown = pressuredivergence + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			pUp = pressuredivergence + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			pBottom = pressuredivergence + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pLeft[pressure_index] + pRight[pressure_index] + pBottom[pressure_index] + pUp[pressure_index] + pDown[pressure_index] - dCentre)/5.f;

		}else if(zIter + 1 ==size_WHD.y){

			pLeft = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			pRight = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			pDown = pressuredivergence + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			pUp = pressuredivergence + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressure_index] = (pLeft[pressure_index] + pRight[pressure_index] + pTop[pressure_index] + pUp[pressure_index] + pDown[pressure_index] - dCentre)/5.f;

		}else{
			unsigned char *fieldLeft = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
		unsigned char *fieldRight = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));

		unsigned char *fieldUp = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
		unsigned char *fieldDown = velocityInput + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);

		unsigned char *fieldTop = velocityInput + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
		unsigned char *fieldBottom = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter);

		unsigned char* cellDivergence = divergence + (zIter*pitch_slice) + (yIter*pitch) + (4*xIter);
			fieldLeft = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			fieldRight = pressuredivergence + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			fieldDown = pressuredivergence + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			fieldUp = pressuredivergence + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			fieldTop = pressuredivergence + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			fieldBottom = pressuredivergence + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * ((fieldRight[0] - fieldLeft[0])+  
				(fieldTop[1] - fieldBottom[1]) + (fieldUp[2] - fieldDown[2])); 

		}
	}
}
extern "C"
void cuda_fluid_divergence(void *divergence, void *velocityInput, float3 size_WHD, size_t pitch, size_t pitch_slice, int divergence_index){
	cudaError_t error = cudaSuccess;

    dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((size_WHD.x+Db.x-1)/Db.x, (size_WHD.y+Db.y-1)/Db.y);

    cuda_kernel_divergence<<<Dg,Db>>>((unsigned char *)divergence, (unsigned char *)velocityInput,size_WHD, pitch, pitch_slice, divergence_index);

    error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("cuda_kernel_divergence() failed to launch error = %d\n", error);
    }
}
