#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

__global__ void cuda_kernel_jacobi(unsigned char *pressuredivergence, float3 sizeWHD, size_t pitch, size_t pitchSlice, int pressureIndex, int divergenceIndex){  
	int xIter = blockIdx.x*blockDim.x + threadIdx.x;
	int yIter = blockIdx.y*blockDim.y + threadIdx.y;
	int zIter = 0;

	for(zIter = 0; zIter < sizeWHD.z; ++zIter){
		unsigned char* cellDivergence = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*xIter);
		// Get the divergence at the current cell.  
		float dCentre = cellDivergence[divergenceIndex];
		// Get pressure values from neighboring cells. 
		unsigned char *pLeft, *pRight = NULL;
		unsigned char *pDown, *pUp = NULL;
		unsigned char *pBottom, *pTop = NULL;

		// Compute the new pressure value for the center cell.
		unsigned char* cellPressure = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*xIter);

		if((xIter - 1 < 0) && (yIter - 1 < 0) && (zIter - 1 < 0)){

			pRight = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter+1));
			pUp = pressuredivergence + (zIter*pitchSlice) + ((yIter+1)*pitch) + (4*xIter); 
			pBottom = pressuredivergence + ((zIter+1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pRight[pressureIndex] + pBottom[pressureIndex] + pUp[pressureIndex] - dCentre)/3.f;

		}else if((xIter + 1 ==sizeWHD.x) && (yIter - 1 < 0) && (zIter - 1 < 0)){

			pLeft = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter-1));
			pUp = pressuredivergence + (zIter*pitchSlice) + ((yIter+1)*pitch) + (4*xIter); 
			pBottom = pressuredivergence + ((zIter+1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pLeft[pressureIndex] + pBottom[pressureIndex] + pUp[pressureIndex] - dCentre)/3.f;

		}else if((xIter - 1 < 0) && (yIter + 1 ==sizeWHD.y) && (zIter - 1 < 0)){

			pRight = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter+1));
			pDown = pressuredivergence + (zIter*pitchSlice) + ((yIter-1)*pitch) + (4*xIter); 
			pBottom = pressuredivergence + ((zIter+1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pRight[pressureIndex] + pBottom[pressureIndex] + pDown[pressureIndex] - dCentre)/3.f;

		}else if((xIter + 1 ==sizeWHD.x) && (yIter + 1 ==sizeWHD.y) && (zIter - 1 < 0)){

			pLeft = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter-1));
			pDown = pressuredivergence + (zIter*pitchSlice) + ((yIter-1)*pitch) + (4*xIter); 
			pBottom = pressuredivergence + ((zIter+1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pLeft[pressureIndex] + pBottom[pressureIndex] + pDown[pressureIndex] - dCentre)/3.f;

		}else if((xIter - 1 < 0) && (yIter - 1 < 0) && (zIter + 1 ==sizeWHD.y)){

			pRight = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter+1));
			pUp = pressuredivergence + (zIter*pitchSlice) + ((yIter+1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pRight[pressureIndex] + pTop[pressureIndex] + pUp[pressureIndex] - dCentre)/3.f;

		}else if((xIter + 1 ==sizeWHD.x) && (yIter - 1 < 0) && (zIter + 1 ==sizeWHD.y)){

			pLeft = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter-1));
			pUp = pressuredivergence + (zIter*pitchSlice) + ((yIter+1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pLeft[pressureIndex] + pTop[pressureIndex] + pUp[pressureIndex] - dCentre)/3.f;

		}else if((xIter - 1 < 0) && (yIter + 1 ==sizeWHD.y) && (zIter + 1 ==sizeWHD.y)){

			pRight = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter+1));
			pDown = pressuredivergence + (zIter*pitchSlice) + ((yIter-1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pRight[pressureIndex] + pTop[pressureIndex] + pDown[pressureIndex] - dCentre)/3.f;

		}else if((xIter + 1 ==sizeWHD.x) && (yIter + 1 ==sizeWHD.y) && (zIter + 1 ==sizeWHD.y)){

			pLeft = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter-1));
			pDown = pressuredivergence + (zIter*pitchSlice) + ((yIter-1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pLeft[pressureIndex] + pTop[pressureIndex] + pDown[pressureIndex] - dCentre)/3.f;

		}else if((xIter - 1 < 0) && (yIter - 1 < 0)){

			pRight = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter+1));
			pUp = pressuredivergence + (zIter*pitchSlice) + ((yIter+1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			pBottom = pressuredivergence + ((zIter+1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pRight[pressureIndex] + pBottom[pressureIndex] + pTop[pressureIndex] + pUp[pressureIndex] - dCentre)/4.f;

		}else if((xIter + 1 ==sizeWHD.x) && (yIter - 1 < 0)){

			pLeft = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter-1));
			pUp = pressuredivergence + (zIter*pitchSlice) + ((yIter+1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			pBottom = pressuredivergence + ((zIter+1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pLeft[pressureIndex] + pBottom[pressureIndex] + pTop[pressureIndex] + pUp[pressureIndex] - dCentre)/4.f;

		}else if((xIter - 1 < 0) && (yIter + 1 ==sizeWHD.y)){

			pRight = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter+1));
			pDown = pressuredivergence + (zIter*pitchSlice) + ((yIter-1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			pBottom = pressuredivergence + ((zIter+1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pRight[pressureIndex] + pBottom[pressureIndex] + pTop[pressureIndex] + pDown[pressureIndex] - dCentre)/4.f;

		}else if((xIter + 1 ==sizeWHD.x) && (yIter + 1 ==sizeWHD.y)){

			pLeft = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter-1));
			pDown = pressuredivergence + (zIter*pitchSlice) + ((yIter-1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			pBottom = pressuredivergence + ((zIter+1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pLeft[pressureIndex] + pBottom[pressureIndex] + pTop[pressureIndex] + pDown[pressureIndex] - dCentre)/4.f;

		}else if((xIter - 1 < 0) && (zIter - 1 < 0)){

			pRight = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter+1));
			pDown = pressuredivergence + (zIter*pitchSlice) + ((yIter-1)*pitch) + (4*xIter); 
			pUp = pressuredivergence + (zIter*pitchSlice) + ((yIter+1)*pitch) + (4*xIter); 
			pBottom = pressuredivergence + ((zIter+1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pRight[pressureIndex] + pBottom[pressureIndex] + pUp[pressureIndex] + pDown[pressureIndex] - dCentre)/4.f;

		}else if((xIter + 1 ==sizeWHD.x) && (zIter - 1 < 0)){

			pLeft = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter-1));
			pDown = pressuredivergence + (zIter*pitchSlice) + ((yIter-1)*pitch) + (4*xIter); 
			pUp = pressuredivergence + (zIter*pitchSlice) + ((yIter+1)*pitch) + (4*xIter); 
			pBottom = pressuredivergence + ((zIter+1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pLeft[pressureIndex] + pBottom[pressureIndex] + pUp[pressureIndex] + pDown[pressureIndex] - dCentre)/4.f;

		}else if((xIter - 1 < 0) && (zIter + 1 ==sizeWHD.y)){

			pRight = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter+1));
			pDown = pressuredivergence + (zIter*pitchSlice) + ((yIter-1)*pitch) + (4*xIter); 
			pUp = pressuredivergence + (zIter*pitchSlice) + ((yIter+1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pRight[pressureIndex] + pTop[pressureIndex] + pUp[pressureIndex] + pDown[pressureIndex] - dCentre)/4.f;

		}else if((xIter + 1 ==sizeWHD.x) && (zIter + 1 ==sizeWHD.y)){

			pLeft = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter-1));
			pDown = pressuredivergence + (zIter*pitchSlice) + ((yIter-1)*pitch) + (4*xIter); 
			pUp = pressuredivergence + (zIter*pitchSlice) + ((yIter+1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pLeft[pressureIndex] + pTop[pressureIndex] + pUp[pressureIndex] + pDown[pressureIndex] - dCentre)/4.f;

		}else if((yIter - 1 < 0) && (zIter - 1 < 0)){

			pLeft = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter-1));
			pRight = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter+1));
			pUp = pressuredivergence + (zIter*pitchSlice) + ((yIter+1)*pitch) + (4*xIter); 
			pBottom = pressuredivergence + ((zIter+1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pLeft[pressureIndex] + pRight[pressureIndex] + pBottom[pressureIndex] + pUp[pressureIndex] - dCentre)/4.f;

		}else if((yIter + 1 ==sizeWHD.y) && (zIter - 1 < 0)){

			pLeft = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter-1));
			pRight = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter+1));
			pDown = pressuredivergence + (zIter*pitchSlice) + ((yIter-1)*pitch) + (4*xIter); 
			pBottom = pressuredivergence + ((zIter+1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pLeft[pressureIndex] + pRight[pressureIndex] + pBottom[pressureIndex] + pDown[pressureIndex] - dCentre)/4.f;

		}else if((yIter - 1 < 0) && (zIter + 1 ==sizeWHD.y)){
			pLeft = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter-1));
			pRight = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter+1));
			pUp = pressuredivergence + (zIter*pitchSlice) + ((yIter+1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pLeft[pressureIndex] + pRight[pressureIndex]+ pTop[pressureIndex] + pUp[pressureIndex] - dCentre)/4.f;
		}else if((yIter + 1 ==sizeWHD.y) && (zIter + 1 ==sizeWHD.y)){

			pLeft = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter-1));
			pRight = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter+1));
			pDown = pressuredivergence + (zIter*pitchSlice) + ((yIter-1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pLeft[pressureIndex] + pRight[pressureIndex] + pTop[pressureIndex] + pDown[pressureIndex] - dCentre)/4.f;

		}else if(xIter - 1 < 0){

			pRight = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter+1));
			pDown = pressuredivergence + (zIter*pitchSlice) + ((yIter-1)*pitch) + (4*xIter); 
			pUp = pressuredivergence + (zIter*pitchSlice) + ((yIter+1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			pBottom = pressuredivergence + ((zIter+1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pRight[pressureIndex] + pBottom[pressureIndex] + pTop[pressureIndex] + pUp[pressureIndex] + pDown[pressureIndex] - dCentre)/5.f;

		}else if(xIter + 1 ==sizeWHD.x){

			pLeft = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter-1));
			pDown = pressuredivergence + (zIter*pitchSlice) + ((yIter-1)*pitch) + (4*xIter); 
			pUp = pressuredivergence + (zIter*pitchSlice) + ((yIter+1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			pBottom = pressuredivergence + ((zIter+1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pLeft[pressureIndex] + pBottom[pressureIndex] + pTop[pressureIndex] + pUp[pressureIndex] + pDown[pressureIndex] - dCentre)/5.f;

		}else if(yIter - 1 < 0){

			pLeft = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter-1));
			pRight = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter+1));
			pUp = pressuredivergence + (zIter*pitchSlice) + ((yIter+1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			pBottom = pressuredivergence + ((zIter+1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pLeft[pressureIndex] + pRight[pressureIndex] + pBottom[pressureIndex] + pTop[pressureIndex] + pUp[pressureIndex] - dCentre)/5.f;

		}else if(yIter + 1 ==sizeWHD.y){

			pLeft = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter-1));
			pRight = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter+1));
			pDown = pressuredivergence + (zIter*pitchSlice) + ((yIter-1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			pBottom = pressuredivergence + ((zIter+1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pLeft[pressureIndex] + pRight[pressureIndex] + pBottom[pressureIndex] + pTop[pressureIndex] + pDown[pressureIndex] - dCentre)/5.f;

		}else if(zIter - 1 < 0){

			pLeft = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter-1));
			pRight = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter+1));
			pDown = pressuredivergence + (zIter*pitchSlice) + ((yIter-1)*pitch) + (4*xIter); 
			pUp = pressuredivergence + (zIter*pitchSlice) + ((yIter+1)*pitch) + (4*xIter); 
			pBottom = pressuredivergence + ((zIter+1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pLeft[pressureIndex] + pRight[pressureIndex] + pBottom[pressureIndex] + pUp[pressureIndex] + pDown[pressureIndex] - dCentre)/5.f;

		}else if(zIter + 1 ==sizeWHD.y){

			pLeft = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter-1));
			pRight = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter+1));
			pDown = pressuredivergence + (zIter*pitchSlice) + ((yIter-1)*pitch) + (4*xIter); 
			pUp = pressuredivergence + (zIter*pitchSlice) + ((yIter+1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pLeft[pressureIndex] + pRight[pressureIndex] + pTop[pressureIndex] + pUp[pressureIndex] + pDown[pressureIndex] - dCentre)/5.f;

		}else{

			pLeft = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter-1));
			pRight = pressuredivergence + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter+1));
			pDown = pressuredivergence + (zIter*pitchSlice) + ((yIter-1)*pitch) + (4*xIter); 
			pUp = pressuredivergence + (zIter*pitchSlice) + ((yIter+1)*pitch) + (4*xIter); 
			pTop = pressuredivergence + ((zIter-1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			pBottom = pressuredivergence + ((zIter+1)*pitchSlice) + (yIter*pitch) + (4*xIter);
			cellPressure[pressureIndex] = (pLeft[pressureIndex] + pRight[pressureIndex] + pBottom[pressureIndex] + pTop[pressureIndex] + pUp[pressureIndex] + pDown[pressureIndex] - dCentre)/6.f;

		}
	}
}
extern "C"
void cuda_fluid_jacobi(void *pressuredivergence, float3 sizeWHD, size_t pitch, size_t pitchSlice, int pressureIndex, int divergenceIndex){
	cudaError_t error = cudaSuccess;

    dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((sizeWHD.x+Db.x-1)/Db.x, (sizeWHD.y+Db.y-1)/Db.y);

    cuda_kernel_jacobi<<<Dg,Db>>>((unsigned char *)pressuredivergence,sizeWHD, pitch, pitchSlice, pressureIndex, divergenceIndex);

    error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("cuda_kernel_jacobi() failed to launch error = %d\n", error);
    }
}