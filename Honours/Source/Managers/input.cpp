////////////////////////////////////////////////////////////////////////////////
// Filename: inputclass.cpp
////////////////////////////////////////////////////////////////////////////////
#include "input.h"
InputClass::InputClass(){
	direct_input_ = 0;
	keyboard_ = 0;
	mouse_ = 0;
}
InputClass::InputClass(const InputClass& other){
}
InputClass::~InputClass(){
}
bool InputClass::Initialize(HINSTANCE hinstance, HWND hwnd, int screen_width, int screen_height){
	HRESULT result;
	// Store the screen size which will be used for positioning the mouse cursor.
	screen_width_ = screen_width;
	screen_height_ = screen_height;
	// Initialize the location of the mouse on the screen.
	mouse_x_ = 0;
	mouse_y_ = 0;
	// Initialize the main direct input interface.
	result = DirectInput8Create(hinstance, DIRECTINPUT_VERSION, IID_IDirectInput8, (void**)&direct_input_, NULL);
	if(FAILED(result)){
		return false;
	}
	// Initialize the direct input interface for the keyboard.
	result = direct_input_->CreateDevice(GUID_SysKeyboard, &keyboard_, NULL);
	if(FAILED(result)){
		return false;
	}
	// Set the data format.  In this case since it is a keyboard we can use the predefined data format.
	result = keyboard_->SetDataFormat(&c_dfDIKeyboard);
	if(FAILED(result)){
		return false;
	}
	// Set the cooperative level of the keyboard to not share with other programs.
	result = keyboard_->SetCooperativeLevel(hwnd, DISCL_FOREGROUND | DISCL_EXCLUSIVE);
	if(FAILED(result)){
		return false;
	}
	// Now acquire the keyboard.
	result = keyboard_->Acquire();
	if(FAILED(result)){
		return false;
	}
	// Initialize the direct input interface for the mouse.
	result = direct_input_->CreateDevice(GUID_SysMouse, &mouse_, NULL);
	if(FAILED(result)){
		return false;
	}
	// Set the data format for the mouse using the pre-defined mouse data format.
	result = mouse_->SetDataFormat(&c_dfDIMouse);
	if(FAILED(result)){
		return false;
	}
	// Set the cooperative level of the mouse to share with other programs.
	result = mouse_->SetCooperativeLevel(hwnd, DISCL_FOREGROUND | DISCL_NONEXCLUSIVE);
	if(FAILED(result)){
		return false;
	}
	// Acquire the mouse.
	result = mouse_->Acquire();
	if(FAILED(result)){
		return false;
	}
	return true;
}
void InputClass::Shutdown(){
	// Release the mouse.
	if(mouse_){
		mouse_->Unacquire();
		mouse_->Release();
		mouse_ = 0;
	}
	// Release the keyboard.
	if(keyboard_){
		keyboard_->Unacquire();
		keyboard_->Release();
		keyboard_ = 0;
	}
	// Release the main interface to direct input.
	if(direct_input_){
		direct_input_->Release();
		direct_input_ = 0;
	}
	return;
}
bool InputClass::Frame(){
	bool result;
	// Read the current state of the keyboard.
	result = ReadKeyboard();
	if(!result){
		return false;
	}
	// Read the current state of the mouse.
	result = ReadMouse();
	if(!result){
		return false;
	}
	// Process the changes in the mouse and keyboard.
	ProcessInput();
	return true;
}
bool InputClass::ReadKeyboard(){
	HRESULT result;
	
	// Read the keyboard device.
	result = keyboard_->GetDeviceState(sizeof(keyboard_state_), (LPVOID)&keyboard_state_);
	if(FAILED(result)){
		// If the keyboard lost focus or was not acquired then try to get control back.
		if((result == DIERR_INPUTLOST) || (result == DIERR_NOTACQUIRED)){
			keyboard_->Acquire();
		}else{
			return false;
		}
	}
		
	return true;
}
bool InputClass::ReadMouse(){
	HRESULT result;
	// Read the mouse device.
	result = mouse_->GetDeviceState(sizeof(DIMOUSESTATE), (LPVOID)&mouse_state_);
	if(FAILED(result)){
		// If the mouse lost focus or was not acquired then try to get control back.
		if((result == DIERR_INPUTLOST) || (result == DIERR_NOTACQUIRED)){
			mouse_->Acquire();
		}else{
			return false;
		}
	}
	return true;
}
void InputClass::ProcessInput(){
	// Update the location of the mouse cursor based on the change of the mouse location during the frame.
	mouse_x_ += mouse_state_.lX;
	mouse_y_ += mouse_state_.lY;
	// Ensure the mouse location doesn't exceed the screen width or height.
	if(mouse_x_ < 0){
		mouse_x_ = 0;
	}
	if(mouse_y_ < 0){
		mouse_y_ = 0;
	}
	if(mouse_x_ > screen_width_){
		mouse_x_ = screen_width_;
	}
	if(mouse_y_ > screen_height_){
		mouse_y_ = screen_height_;
	}
	return;
}
void InputClass::GetMouseLocation(int& mouseX, int& mouseY){
	mouseX = mouse_x_;
	mouseY = mouse_y_;
	return;
}
bool InputClass::IsEscapePressed(){
	// Do a bitwise and on the keyboard state to check if the escape key is currently being pressed.
	if(keyboard_state_[DIK_ESCAPE] & 0x80){
		return true;
	}
	return false;
}
bool InputClass::IsSpacePressed(){
	// Do a bitwise and on the keyboard state to check if the key is currently being pressed.
	if(keyboard_state_[DIK_SPACE] & 0x80){
		return true;
	}
	return false;
}
bool InputClass::IsLeftPressed(){
	// Do a bitwise and on the keyboard state to check if the key is currently being pressed.
	if(keyboard_state_[DIK_LEFT] & 0x80){
		return true;
	}
	return false;
}
bool InputClass::IsRightPressed(){
	// Do a bitwise and on the keyboard state to check if the key is currently being pressed.
	if(keyboard_state_[DIK_RIGHT] & 0x80){
		return true;
	}
	return false;
}
bool InputClass::IsUpPressed(){
	// Do a bitwise and on the keyboard state to check if the key is currently being pressed.
	if(keyboard_state_[DIK_UP] & 0x80){
		return true;
	}
	return false;
}
bool InputClass::IsDownPressed(){
	// Do a bitwise and on the keyboard state to check if the key is currently being pressed.
	if(keyboard_state_[DIK_DOWN] & 0x80){
		return true;
	}
	return false;
}
bool InputClass::IsAPressed(){
	// Do a bitwise and on the keyboard state to check if the key is currently being pressed.
	if(keyboard_state_[DIK_A] & 0x80){
		return true;
	}
	return false;
}
bool InputClass::IsZPressed(){
	// Do a bitwise and on the keyboard state to check if the key is currently being pressed.
	if(keyboard_state_[DIK_Z] & 0x80){
		return true;
	}
	return false;
}
bool InputClass::IsPgUpPressed(){
	// Do a bitwise and on the keyboard state to check if the key is currently being pressed.
	if(keyboard_state_[DIK_PGUP] & 0x80){
		return true;
	}
	return false;
}
bool InputClass::IsPgDownPressed(){
	// Do a bitwise and on the keyboard state to check if the key is currently being pressed.
	if(keyboard_state_[DIK_PGDN] & 0x80){
		return true;
	}
	return false;
}
bool InputClass::IsHPressed(){
	// Do a bitwise and on the keyboard state to check if the key is currently being pressed.
	if(keyboard_state_[DIK_H] & 0x80){
		return true;
	}
	return false;
}
bool InputClass::IsRPressed(){
	// Do a bitwise and on the keyboard state to check if the key is currently being pressed.
	if(keyboard_state_[DIK_R] & 0x80){
		return true;
	}
	return false;
}
bool InputClass::IsWPressed(){
	// Do a bitwise and on the keyboard state to check if the key is currently being pressed.
	if(keyboard_state_[DIK_W] & 0x80){
		return true;
	}
	return false;
}
bool InputClass::IsQPressed(){
	// Do a bitwise and on the keyboard state to check if the key is currently being pressed.
	if(keyboard_state_[DIK_Q] & 0x80){
		return true;
	}
	return false;
}
bool InputClass::IsEPressed(){
	// Do a bitwise and on the keyboard state to check if the key is currently being pressed.
	if(keyboard_state_[DIK_E] & 0x80){
		return true;
	}
	return false;
}
bool InputClass::IsSPressed(){
	// Do a bitwise and on the keyboard state to check if the key is currently being pressed.
	if(keyboard_state_[DIK_S] & 0x80){
		return true;
	}
	return false;
}
bool InputClass::IsDPressed(){
	// Do a bitwise and on the keyboard state to check if the key is currently being pressed.
	if(keyboard_state_[DIK_D] & 0x80){
		return true;
	}
	return false;
}
bool InputClass::IsXPressed(){
	// Do a bitwise and on the keyboard state to check if the key is currently being pressed.
	if(keyboard_state_[DIK_X] & 0x80){
		return true;
	}
	return false;
}