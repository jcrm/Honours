#include <stdio.h>
#include <stdlib.h>
#include <string.h>
__global__ void cuda_kernel_fluid(unsigned char *surface, int width, int height, int depth, size_t pitch, size_t pitchSlice)
{
	
}
extern "C"
void cuda_fluid(void *output, void *velocityinput, int width, int height, int depth, size_t pitch, size_t pitchSlice)
{
	cudaError_t error = cudaSuccess;

    dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((width+Db.x-1)/Db.x, (height+Db.y-1)/Db.y);

    ADVECT_VEL<<<Dg,Db>>>();

    error = cudaGetLastError();

    if (error != cudaSuccess){
        printf("cuda_kernel_texture_3d() failed to launch error = %d\n", error);
    }
}

__global__ void ADVECT_VEL(GS_OUTPUT_FLUIDSIM in, Texture3D velocity){  
  float3 pos = in.cellIndex;  //position in array
  float3 cellVelocity = velocity.Sample(samPointClamp, in.CENTERCELL).xyz;  //value of current position
  pos -= timeStep * cellVelocity;  
  pos = cellIndex2TexCoord(pos);  
  return velocity.Sample(samLinear, pos);  
} 
__global__ void PS_DIVERGENCE(Texture3D divergence, GS_OUTPUT_FLUIDSIM in, Texture3D velocity) : SV_Target  
{  
  // Get velocity values from neighboring cells.  
  float4 fieldL = velocity.Sample(samPointClamp, in.LEFTCELL);  
  float4 fieldR = velocity.Sample(samPointClamp, in.RIGHTCELL);  
  float4 fieldB = velocity.Sample(samPointClamp, in.BOTTOMCELL);  
  float4 fieldT = velocity.Sample(samPointClamp, in.TOPCELL);  
  float4 fieldD = velocity.Sample(samPointClamp, in.DOWNCELL);  
  float4 fieldU = velocity.Sample(samPointClamp, in.UPCELL);  
  // Compute the velocity's divergence using central differences.  
   float divergence =  0.5 * ((fieldR.x - fieldL.x)+  
                             (fieldT.y - fieldB.y)+  
                             (fieldU.z - fieldD.z));  
  return divergence;  
} 
/*
struct GS_OUTPUT_FLUIDSIM  
{  
  // Index of the current grid cell (i,j,k in [0,gridSize] range)  
  float3 cellIndex : TEXCOORD0;  
  // Texture coordinates (x,y,z in [0,1] range) for the  
   // current grid cell and its immediate neighbors  
  float3 CENTERCELL : TEXCOORD1;  
  float3 LEFTCELL   : TEXCOORD2;  
  float3 RIGHTCELL  : TEXCOORD3;  
  float3 BOTTOMCELL : TEXCOORD4;  
  float3 TOPCELL    : TEXCOORD5;  
  float3 DOWNCELL   : TEXCOORD6;  
  float3 UPCELL     : TEXCOORD7;  
  float4 pos        : SV_Position; // 2D slice vertex in  
   // homogeneous clip space  
   uint RTIndex    : SV_RenderTargetArrayIndex;  // Specifies  
   // destination slice  
};  
float3 cellIndex2TexCoord(float3 index)  
{  
  // Convert a value in the range [0,gridSize] to one in the range [0,1].  
   return float3(index.x / textureWidth,  
                index.y / textureHeight,  
                (index.z+0.5) / textureDepth);  
}  
float4 PS_ADVECT_VEL(GS_OUTPUT_FLUIDSIM in,  
                     Texture3D velocity) : SV_Target  
{  
  float3 pos = in.cellIndex;  
  float3 cellVelocity = velocity.Sample(samPointClamp,  
                                        in.CENTERCELL).xyz;  
  pos -= timeStep * cellVelocity;  
  pos = cellIndex2TexCoord(pos);  
  return velocity.Sample(samLinear, pos);  
}  
float PS_DIVERGENCE(GS_OUTPUT_FLUIDSIM in,  
                    Texture3D velocity) : SV_Target  
{  
  // Get velocity values from neighboring cells.  
   float4 fieldL = velocity.Sample(samPointClamp, in.LEFTCELL);  
  float4 fieldR = velocity.Sample(samPointClamp, in.RIGHTCELL);  
  float4 fieldB = velocity.Sample(samPointClamp, in.BOTTOMCELL);  
  float4 fieldT = velocity.Sample(samPointClamp, in.TOPCELL);  
  float4 fieldD = velocity.Sample(samPointClamp, in.DOWNCELL);  
  float4 fieldU = velocity.Sample(samPointClamp, in.UPCELL);  
  // Compute the velocity's divergence using central differences.  
   float divergence =  0.5 * ((fieldR.x - fieldL.x)+  
                             (fieldT.y - fieldB.y)+  
                             (fieldU.z - fieldD.z));  
  return divergence;  
}  
float PS_JACOBI(GS_OUTPUT_FLUIDSIM in,  
                Texture3D pressure,  
                Texture3D divergence) : SV_Target  
{  
  // Get the divergence at the current cell.  
   float dC = divergence.Sample(samPointClamp, in.CENTERCELL);  
  // Get pressure values from neighboring cells.  
   float pL = pressure.Sample(samPointClamp, in.LEFTCELL);  
  float pR = pressure.Sample(samPointClamp, in.RIGHTCELL);  
  float pB = pressure.Sample(samPointClamp, in.BOTTOMCELL);  
  float pT = pressure.Sample(samPointClamp, in.TOPCELL);  
  float pD = pressure.Sample(samPointClamp, in.DOWNCELL);  
  float pU = pressure.Sample(samPointClamp, in.UPCELL);  
  // Compute the new pressure value for the center cell.  
   return(pL + pR + pB + pT + pU + pD - dC) / 6.0;  
}  
float4 PS_PROJECT(GS_OUTPUT_FLUIDSIM in,  
                  Texture3D pressure,  
                  Texture3D velocity): SV_Target  
{  
  // Compute the gradient of pressure at the current cell by  
   // taking central differences of neighboring pressure values.  
   float pL = pressure.Sample(samPointClamp, in.LEFTCELL);  
  float pR = pressure.Sample(samPointClamp, in.RIGHTCELL);  
  float pB = pressure.Sample(samPointClamp, in.BOTTOMCELL);  
  float pT = pressure.Sample(samPointClamp, in.TOPCELL);  
  float pD = pressure.Sample(samPointClamp, in.DOWNCELL);  
  float pU = pressure.Sample(samPointClamp, in.UPCELL);  
  float3 gradP = 0.5*float3(pR - pL, pT - pB, pU - pD);  
  // Project the velocity onto its divergence-free component by  
   // subtracting the gradient of pressure.  
   float3 vOld = velocity.Sample(samPointClamp, in.texcoords);  
  float3 vNew = vOld - gradP;  
  return float4(vNew, 0);  
}  */

float3 cellIndex2TexCoord(float3 index, int textureWidth, int textureHeight, int textureDepth){  
  // Convert a value in the range [0,gridSize] to one in the range [0,1].  
   return float3(index.x / textureWidth,  
                index.y / textureHeight,  
                (index.z+0.5) / textureDepth);  
}