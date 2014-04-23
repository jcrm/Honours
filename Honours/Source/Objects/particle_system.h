////////////////////////////////////////////////////////////////////////////////
// Filename: particlesystemclass.h
////////////////////////////////////////////////////////////////////////////////
#ifndef _PARTICLESYSTEMCLASS_H_
#define _PARTICLESYSTEMCLASS_H_


//////////////
// INCLUDES //
//////////////
#include <d3d11.h>
#include <d3dx10math.h>


///////////////////////
// MY CLASS INCLUDES //
///////////////////////
#include "../Textures/texture.h"


////////////////////////////////////////////////////////////////////////////////
// Class name: ParticleSystemClass
////////////////////////////////////////////////////////////////////////////////
class ParticleSystemClass
{
private:
	struct ParticleType{
		float position_x_, position_y_, position_z_;
		float red_, green_, blue_;
		float velocity_;
		bool active_;
	};

	struct VertexType{
		D3DXVECTOR3 position_;
		D3DXVECTOR2 texture_;
		D3DXVECTOR4 color_;
	};

public:
	ParticleSystemClass();
	ParticleSystemClass(const ParticleSystemClass&);
	~ParticleSystemClass();

	bool Initialize(ID3D11Device*, WCHAR*);
	void Shutdown();
	bool Frame(float, ID3D11DeviceContext*);
	void Render(ID3D11DeviceContext*);

	ID3D11ShaderResourceView* GetTexture();
	int GetIndexCount();
	inline int GetKillCount(){return kill_count_;}
	inline D3DXMATRIX GetTranslation() {return translation_;}
	inline void RandomizeTranslation() {D3DXMatrixTranslation(&translation_, float(rand()%129), float(rand()%10), float(rand()%129));}
	inline void SetKillCount(int count){kill_count_ = count;}
private:
	bool LoadTexture(ID3D11Device*, WCHAR*);
	void ReleaseTexture();

	bool InitializeParticleSystem();
	void ShutdownParticleSystem();

	bool InitializeBuffers(ID3D11Device*);
	void ShutdownBuffers();

	void EmitParticles(float);
	void UpdateParticles(float);
	void KillParticles();

	bool UpdateBuffers(ID3D11DeviceContext*);

	void RenderBuffers(ID3D11DeviceContext*);

private:
	float particle_deviation_x_, particle_deviation_y_, particle_deviation_z_;
	float particle_velocity_, particle_velocity_variation_;
	float particle_size_, particles_per_second_;
	int max_particles_;

	int current_particle_count_;
	float accumulated_time_;

	TextureClass* texture_;
	ParticleType* particle_list_;
	int vertex_count_, index_count_;
	VertexType* vertices_;
	ID3D11Buffer *vertex_buffer_, *index_buffer_;
	int kill_count_;
	//where the model is in 3d space
	D3DXMATRIX translation_;
};

#endif