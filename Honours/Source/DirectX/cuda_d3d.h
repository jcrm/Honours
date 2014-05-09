#ifndef _CUDA_D3D_H
#define _CUDA_D3D_H

#include "d3dclass.h"

class CUDAD3D :	public D3DClass
{
public:
	CUDAD3D(void);
	CUDAD3D(const D3DClass&);
	~CUDAD3D(void);
	bool Initialize(int, int, bool, HWND, bool, float, float);
	bool findCUDADevice();
	bool findDXDevice(char*);
protected:
	bool InitDisplayMode(int, int, unsigned int&, unsigned int&);
	bool InitSwapChain(HWND, int, int,unsigned int&, unsigned int&, bool);
private:
	IDXGIAdapter *cuda_capable_adapter_;  // Adapter to use
};
#endif