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
		if(xIter +1 < size_WHD.x){
			if(xIter - 1 > 0){
				if(yIter + 1 < size_WHD.y){
					if(yIter - 1 > 0){
						if(zIter + 1 < size_WHD.z){
							if(zIter - 1 > 0){
								unsigned char *fieldLeft = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
								unsigned char *fieldRight = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
								unsigned char *fieldDown = velocityInput + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
								unsigned char *fieldUp = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
								unsigned char *fieldTop = velocityInput + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
								unsigned char *fieldBottom = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
								unsigned char* cellDivergence = divergence + (zIter*pitch_slice) + (yIter*pitch) + (4*xIter);
								cellDivergence[divergence_index] = signed int(0.5f * ((signed int(fieldRight[0]) - signed int(fieldLeft[0])) + 
									(signed int(fieldTop[1]) - signed int(fieldBottom[1])) + (signed int(fieldUp[2]) - signed int(fieldDown[2]))));
								// Compute the velocity's divergence using central differences.  
							}
						}
					}
				}
			}
		}
		/*
		if((xIter - 1 < 0) && (yIter - 1 < 0) && (zIter - 1 < 0)){

			fieldRight = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			fieldUp = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			fieldBottom = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			cellDivergence[divergence_index] = 0.5f * ((fieldRight[0]) + ( - fieldBottom[1]) + (fieldUp[2])); 

		}else if((xIter + 1 ==size_WHD.x) && (yIter - 1 < 0) && (zIter - 1 < 0)){
			fieldLeft = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			fieldUp = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			fieldBottom = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * ((- fieldLeft[0])+  
				( - fieldBottom[1]) + (fieldUp[2])); 
		}else if((xIter - 1 < 0) && (yIter + 1 ==size_WHD.y) && (zIter - 1 < 0)){
			fieldRight = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			fieldDown = velocityInput + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			fieldBottom = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * ((fieldRight[0])+  
				(fieldBottom[1]) + (fieldDown[2]));
		}else if((xIter + 1 ==size_WHD.x) && (yIter + 1 ==size_WHD.y) && (zIter - 1 < 0)){
			fieldLeft = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			fieldDown = velocityInput + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			fieldBottom = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * ((- fieldLeft[0])+  
				(- fieldBottom[1]) + (- fieldDown[2]));
		}else if((xIter - 1 < 0) && (yIter - 1 < 0) && (zIter + 1 ==size_WHD.y)){
			fieldRight = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			fieldUp = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			fieldTop = velocityInput + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * ((fieldRight[0])+  
				(fieldTop[1]) + (fieldUp[2]));
		}else if((xIter + 1 ==size_WHD.x) && (yIter - 1 < 0) && (zIter + 1 ==size_WHD.y)){
			fieldLeft = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			fieldUp = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			fieldTop = velocityInput + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * ((- fieldLeft[0])+  
				(fieldTop[1]) + (fieldUp[2]));
		}else if((xIter - 1 < 0) && (yIter + 1 ==size_WHD.y) && (zIter + 1 ==size_WHD.y)){
			fieldRight = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			fieldDown = velocityInput + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			fieldTop = velocityInput + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * ((fieldRight[0])+  
				(fieldTop[1]) + (- fieldDown[2])); 
		}else if((xIter + 1 ==size_WHD.x) && (yIter + 1 ==size_WHD.y) && (zIter + 1 ==size_WHD.y)){
			fieldLeft = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			fieldDown = velocityInput + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			fieldTop = velocityInput + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * (( - fieldLeft[0])+  
				(fieldTop[1]) + ( - fieldDown[2])); 
		}else if((xIter - 1 < 0) && (yIter - 1 < 0)){
			fieldRight = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			fieldUp = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			fieldTop = velocityInput + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			fieldBottom = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * ((fieldRight[0])+  
				(fieldTop[1] - fieldBottom[1]) + (fieldUp[2]));
		}else if((xIter + 1 ==size_WHD.x) && (yIter - 1 < 0)){
			fieldLeft = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			fieldUp = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			fieldTop = velocityInput + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			fieldBottom = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * (( - fieldLeft[0])+  
				(fieldTop[1] - fieldBottom[1]) + (fieldUp[2]));
		}else if((xIter - 1 < 0) && (yIter + 1 ==size_WHD.y)){
			fieldRight = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			fieldDown = velocityInput + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			fieldTop = velocityInput + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			fieldBottom = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * ((fieldRight[0])+  
				(fieldTop[1] - fieldBottom[1]) + (- fieldDown[2])); 
		}else if((xIter + 1 ==size_WHD.x) && (yIter + 1 ==size_WHD.y)){
			fieldLeft = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			fieldDown = velocityInput + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			fieldTop = velocityInput + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			fieldBottom = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * (fieldLeft[0])+  
				(fieldTop[1] - fieldBottom[1]) + (- fieldDown[2]); 
		}else if((xIter - 1 < 0) && (zIter - 1 < 0)){
			fieldRight = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			fieldDown = velocityInput + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			fieldUp = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			fieldBottom = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * ((fieldRight[0])+  
				( fieldBottom[1]) + (fieldUp[2] - fieldDown[2]));
		}else if((xIter + 1 ==size_WHD.x) && (zIter - 1 < 0)){
			fieldLeft = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			fieldDown = velocityInput + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			fieldUp = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			fieldBottom = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * ((- fieldLeft[0])+  
				(- fieldBottom[1]) + (fieldUp[2] - fieldDown[2]));
		}else if((xIter - 1 < 0) && (zIter + 1 ==size_WHD.y)){
			fieldRight = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			fieldDown = velocityInput + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			fieldUp = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			fieldTop = velocityInput + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * ((fieldRight[0])+  
				(fieldTop[1]) + (fieldUp[2] - fieldDown[2]));
		}else if((xIter + 1 ==size_WHD.x) && (zIter + 1 ==size_WHD.y)){
			fieldLeft = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			fieldDown = velocityInput + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			fieldUp = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			fieldTop = velocityInput + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * (( - fieldLeft[0])+  
				(fieldTop[1]) + (fieldUp[2] - fieldDown[2])); 
		}else if((yIter - 1 < 0) && (zIter - 1 < 0)){
			fieldLeft = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			fieldRight = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			fieldUp = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			fieldBottom = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * ((fieldRight[0] - fieldLeft[0])+  
				(-fieldBottom[1]) + (fieldUp[2])); 
		}else if((yIter + 1 ==size_WHD.y) && (zIter - 1 < 0)){
			fieldLeft = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			fieldRight = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			fieldDown = velocityInput + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			fieldBottom = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * ((fieldRight[0] - fieldLeft[0])+  
				(fieldBottom[1]) + (fieldDown[2])); 
		}else if((yIter - 1 < 0) && (zIter + 1 ==size_WHD.y)){
			fieldLeft = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			fieldRight = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			fieldUp = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			fieldTop = velocityInput + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * ((fieldRight[0] - fieldLeft[0])+  
				(fieldTop[1]) + (fieldUp[2])); 
		}else if((yIter + 1 ==size_WHD.y) && (zIter + 1 ==size_WHD.y)){
			fieldLeft = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			fieldRight = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			fieldDown = velocityInput + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			fieldUp = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * ((fieldRight[0] - fieldLeft[0]) + (fieldUp[2] - fieldDown[2]));
		}else if(xIter - 1 < 0){
			fieldRight = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			fieldDown = velocityInput + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			fieldUp = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			fieldTop = velocityInput + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			fieldBottom = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * ((fieldRight[0])+  
				(fieldTop[1] - fieldBottom[1]) + (fieldUp[2] - fieldDown[2])); 
		}else if(xIter + 1 ==size_WHD.x){
			fieldLeft = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			fieldDown = velocityInput + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			fieldUp = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			fieldTop = velocityInput + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			fieldBottom = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * ((- fieldLeft[0])+  
				(fieldTop[1] - fieldBottom[1]) + (fieldUp[2] - fieldDown[2])); 
		}else if(yIter - 1 < 0){
			fieldLeft = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			fieldRight = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			fieldUp = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			fieldTop = velocityInput + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			fieldBottom = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * ((fieldRight[0] - fieldLeft[0])+  
				(fieldTop[1] - fieldBottom[1]) + (fieldUp[2])); 
		}else if(yIter + 1 ==size_WHD.y){
			fieldLeft = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			fieldRight = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			fieldDown = velocityInput + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			fieldTop = velocityInput + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			fieldBottom = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * ((fieldRight[0] - fieldLeft[0])+  
				(fieldTop[1] - fieldBottom[1]) + (- fieldDown[2])); 
		}else if(zIter - 1 < 0){
			fieldLeft = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			fieldRight = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			fieldDown = velocityInput + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			fieldUp = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			fieldBottom = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * ((fieldRight[0] - fieldLeft[0])+  
				(-fieldBottom[1]) + (fieldUp[2] - fieldDown[2])); 
		}else if(zIter + 1 ==size_WHD.y){
			fieldLeft = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			fieldRight = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			fieldDown = velocityInput + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			fieldUp = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			fieldTop = velocityInput + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * ((fieldRight[0] - fieldLeft[0])+  
				(fieldTop[1]) + (fieldUp[2] - fieldDown[2])); 
		}else{
			fieldLeft = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter-1));
			fieldRight = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*(xIter+1));
			fieldDown = velocityInput + (zIter*pitch_slice) + ((yIter-1)*pitch) + (4*xIter); 
			fieldUp = velocityInput + (zIter*pitch_slice) + ((yIter+1)*pitch) + (4*xIter); 
			fieldTop = velocityInput + ((zIter-1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			fieldBottom = velocityInput + ((zIter+1)*pitch_slice) + (yIter*pitch) + (4*xIter);
			// Compute the velocity's divergence using central differences.  
			cellDivergence[divergence_index] =  0.5f * ((fieldRight[0] - fieldLeft[0])+  
				(fieldTop[1] - fieldBottom[1]) + (fieldUp[2] - fieldDown[2])); 
		}*/
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
