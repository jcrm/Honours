////////////////////////////////////////////////////////////////////////////////
// Filename: textclass.h
////////////////////////////////////////////////////////////////////////////////
#ifndef _TEXTCLASS_H_
#define _TEXTCLASS_H_
///////////////////////
// MY CLASS INCLUDES //
///////////////////////
#include "../Text/fontclass.h"
#include "../Shaders/fontshaderclass.h"
////////////////////////////////////////////////////////////////////////////////
// Class name: TextClass
////////////////////////////////////////////////////////////////////////////////
class TextClass
{
private:
	struct SentenceType
	{
		ID3D11Buffer *vertex_buffer_, *index_buffer_;
		int vertex_count_, index_count_, max_length_;
		float red_, green_, blue_;
	};
	struct VertexType
	{
		D3DXVECTOR3 position_;
	    D3DXVECTOR2 texture_;
	};
public:
	TextClass();
	TextClass(const TextClass&);
	~TextClass();
	bool Initialize(ID3D11Device*, ID3D11DeviceContext*, HWND, int, int, D3DXMATRIX);
	void Shutdown();
	bool Render(ID3D11DeviceContext*, FontShaderClass*, D3DXMATRIX, D3DXMATRIX);
	bool SetVideoCardInfo(char*, int, ID3D11DeviceContext*);
	bool SetFps(int, ID3D11DeviceContext*);
	bool SetCpu(int, ID3D11DeviceContext*);
	bool SetCameraPosition(float, float, float, ID3D11DeviceContext*);
	bool SetCameraRotation(float, float, float, ID3D11DeviceContext*);
	bool SetTime(float, ID3D11DeviceContext*);
private:
	bool InitializeSentence(SentenceType**, int, ID3D11Device*);
	bool UpdateSentence(SentenceType*, char*, int, int, float, float, float, ID3D11DeviceContext*);
	void ReleaseSentence(SentenceType**);
	bool RenderSentence(SentenceType*, ID3D11DeviceContext*, FontShaderClass*, D3DXMATRIX, D3DXMATRIX);
private:
	int screen_width_, screen_height_;
	D3DXMATRIX base_view_matrix_;
	FontClass* font_;
	SentenceType *sentence_one_, *sentence_two_, *sentence_three_, *sentence_four_, *sentence_five_;
	SentenceType *sentence_six_, *sentence_seven_, *sentence_eight_, *sentence_nine_, *sentence_tem_;
	SentenceType *timer_sentence_;
};
#endif