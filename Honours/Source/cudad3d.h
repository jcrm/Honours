#pragma once
#include "d3dclass.h"
class CUDAD3D :
	public D3DClass
{
public:
	CUDAD3D(void);
	CUDAD3D(const D3DClass&);
	~CUDAD3D(void);
	bool Initialize(int, int, bool, HWND, bool, float, float);
	bool findCUDADevice();
	bool findDXDevice( char* dev_name );

	IDXGIAdapter *g_pCudaCapableAdapter;  // Adapter to use
};

