////////////////////////////////////////////////////////////////////////////////
// Filename: inputclass.h
////////////////////////////////////////////////////////////////////////////////
#ifndef _INPUTCLASS_H_
#define _INPUTCLASS_H_
///////////////////////////////
// PRE-PROCESSING DIRECTIVES //
///////////////////////////////
#define DIRECTINPUT_VERSION 0x0800
/////////////
// LINKING //
/////////////
#pragma comment(lib, "dinput8.lib")
#pragma comment(lib, "dxguid.lib")
//////////////
// INCLUDES //
//////////////
#include <dinput.h>
////////////////////////////////////////////////////////////////////////////////
// Class name: InputClass
////////////////////////////////////////////////////////////////////////////////
class InputClass
{
public:
	InputClass();
	InputClass(const InputClass&);
	~InputClass();
	bool Initialize(HINSTANCE, HWND, int, int);
	void Shutdown();
	bool Frame();
	void GetMouseLocation(int&, int&);
	bool IsSpacePressed();
	bool IsEscapePressed();
	bool IsLeftPressed();
	bool IsRightPressed();
	bool IsUpPressed();
	bool IsDownPressed();
	bool IsAPressed();
	bool IsZPressed();
	bool IsPgUpPressed();
	bool IsPgDownPressed();
	//added extra key presses for different functionality 
	bool IsHPressed();
	bool IsRPressed();
	bool IsWPressed();
	bool IsSPressed();
	bool IsDPressed();
	bool IsEPressed();
	bool IsQPressed();
	bool IsXPressed();
private:
	bool ReadKeyboard();
	bool ReadMouse();
	void ProcessInput();
private:
	IDirectInput8* direct_input_;
	IDirectInputDevice8* keyboard_;
	IDirectInputDevice8* mouse_;
	unsigned char keyboard_state_[256];
	DIMOUSESTATE mouse_state_;
	int screen_width_, screen_height_;
	int mouse_x_, mouse_y_;
};
#endif