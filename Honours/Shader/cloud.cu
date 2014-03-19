#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

float3 cellIndex2TexCoord(float3 index, float3 sizeWHD);

__global__ void cuda_kernel_advect(unsigned char *output, unsigned char *velocityInput, float3 sizeWHD, size_t pitch, size_t pitchSlice){ 
	int xIter = blockIdx.x*blockDim.x + threadIdx.x;
	int yIter = blockIdx.y*blockDim.y + threadIdx.y;
	int zIter = 0;
	float timeStep = 1;
	//location is z slide + y position + variable size time x position
	int location =(zIter*pitchSlice) + (yIter*pitch) + (4*xIter);
	unsigned char* cellVelocity = velocityInput + location;
	float3 pos;
	pos.x = xIter;
	pos.y = yIter;
	pos.z = zIter;

	pos.x -= timeStep * cellVelocity[0];  
	pos.y -= timeStep * cellVelocity[1];  
	pos.z -= timeStep * cellVelocity[2];  

	pos.x /= sizeWHD.x;
	pos.y /= sizeWHD.y;
	pos.z += 0.5;
	pos.z /= sizeWHD.z;

	unsigned char* outputPixel = output + location;
	outputPixel[0] = pos.x;
	outputPixel[1] = pos.y;
	outputPixel[2] = pos.z;
	outputPixel[3] = 255; 
} 
__global__ void cuda_kernel_divergence(unsigned char* divergence, unsigned char* velocityInput, size_t pitch, size_t pitchSlice){
	int xIter = blockIdx.x*blockDim.x + threadIdx.x;
	int yIter = blockIdx.y*blockDim.y + threadIdx.y;
	int zIter = 0;

	// Get velocity values from neighboring cells.  
	unsigned char *fieldLeft = velocityInput + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter-1));
	unsigned char *fieldRight = velocityInput + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter+1));
	unsigned char *fieldBottom = velocityInput + ((zIter+1)*pitchSlice) + (yIter*pitch) + (4*xIter);  
	unsigned char *fieldTop = velocityInput + ((zIter-1)*pitchSlice) + (yIter*pitch) + (4*xIter);   
	unsigned char *fieldDown = velocityInput + (zIter*pitchSlice) + ((yIter-1)*pitch) + (4*xIter); 
	unsigned char *fieldUp = velocityInput + (zIter*pitchSlice) + ((yIter+1)*pitch) + (4*xIter);
	// Compute the velocity's divergence using central differences.  
	divergence[0] =  0.5 * ((fieldRight[0] - fieldLeft[0])+  
								(fieldTop[1] - fieldBottom[1])+  
								(fieldUp[2] - fieldDown[2]));  
}
__global__ void cuda_kernel_jacobi(unsigned char *pressure, unsigned char *divergence, size_t pitch, size_t pitchSlice){  
	int xIter = blockIdx.x*blockDim.x + threadIdx.x;
	int yIter = blockIdx.y*blockDim.y + threadIdx.y;
	int zIter = 0;
	unsigned char* cellDivergence = divergence + (zIter*pitchSlice) + (yIter*pitch) + (4*xIter);
	// Get the divergence at the current cell.  
	float dCentre = cellDivergence[0];  
	// Get pressure values from neighboring cells.  
	unsigned char *pLeft = pressure + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter-1));
	unsigned char *pRight = pressure + (zIter*pitchSlice) + (yIter*pitch) + (4*(xIter+1));
	unsigned char *pBottom = pressure + ((zIter+1)*pitchSlice) + (yIter*pitch) + (4*xIter);
	unsigned char *pTop = pressure + ((zIter-1)*pitchSlice) + (yIter*pitch) + (4*xIter);
	unsigned char *pDown = pressure + (zIter*pitchSlice) + ((yIter-1)*pitch) + (4*xIter);  
	unsigned char *pUp = pressure + (zIter*pitchSlice) + ((yIter+1)*pitch) + (4*xIter); 

	// Compute the new pressure value for the center cell.
	unsigned char* cellPressure = pressure + (zIter*pitchSlice) + (yIter*pitch) + (4*xIter);
	cellPressure[0] = (pLeft[0] + pRight[0] + pBottom[0] + pTop[0] + pUp[0] + pDown[0] - dCentre) / 6.0;  
} /*
__global__ void cuda_kernel_project(unsigned char *pressure, unsigned char* velocityInput, size_t pitch, size_t pitchSlice){  
	// Compute the gradient of pressure at the current cell by  
	// taking central differences of neighboring pressure values.  
	float pLeft = pressure.Sample(samPointClamp, in.LEFTCELL);  
	float pRight = pressure.Sample(samPointClamp, in.RIGHTCELL);  
	float pBottom = pressure.Sample(samPointClamp, in.BOTTOMCELL);  
	float pTop = pressure.Sample(samPointClamp, in.TOPCELL);  
	float pDown = pressure.Sample(samPointClamp, in.DOWNCELL);  
	float pUp = pressure.Sample(samPointClamp, in.UPCELL);  
	float3 gradP = 0.5*float3(pRright - pLeft, pTop - pBottom, pUp - pDown);  
	// Project the velocity onto its divergence-free component by  
	// subtracting the gradient of pressure.  
	float3 vOld = velocity.Sample(samPointClamp, in.texcoords);  
	float3 vNew = vOld - gradP;  
	return float4(vNew, 0);  
} */

extern "C"
void cuda_fluid_advect(void *output, void *velocityinput, float3 sizeWHD, size_t pitch, size_t pitchSlice){
	cudaError_t error = cudaSuccess;

    dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((sizeWHD.x+Db.x-1)/Db.x, (sizeWHD.y+Db.y-1)/Db.y);

    cuda_kernel_advect<<<Dg,Db>>>((unsigned char *)output, (unsigned char *)velocityinput, sizeWHD, pitch, pitchSlice);

    error = cudaGetLastError();

    if (error != cudaSuccess){
        printf("cuda_kernel_texture_3d() failed to launch error = %d\n", error);
    }
}
float3 cellIndex2TexCoord(float3 index, float3 sizeWHD){  
	// Convert a value in the range [0,gridSize] to one in the range [0,1].
	float3 temp = index;
	temp.x /= sizeWHD.x;
	temp.y /= sizeWHD.y;
	temp.z += 0.5;
	temp.z /= sizeWHD.z;
	return temp;  
}