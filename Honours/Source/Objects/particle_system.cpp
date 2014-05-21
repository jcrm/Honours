////////////////////////////////////////////////////////////////////////////////
// Filename: particlesystemclass.cpp
////////////////////////////////////////////////////////////////////////////////
#include "particle_system.h"

ParticleSystemClass::ParticleSystemClass(){
	texture_ = 0;
	particle_list_ = 0;
	vertex_buffer_ = 0;
	instance_buffer_ = 0;
	D3DXMatrixIdentity(&translation_);
	system_position_ = D3DXVECTOR3(float(rand()%256), 20.f, float(rand()%256));
	D3DXMatrixTranslation(&translation_, system_position_.x,system_position_.y,system_position_.z);
}

ParticleSystemClass::ParticleSystemClass(const ParticleSystemClass& other){
}

ParticleSystemClass::~ParticleSystemClass(){
}

bool ParticleSystemClass::Initialize(ID3D11Device* device, WCHAR* texture_filename){
	bool result;
	is_clear_ = false;
	// Load the texture that is used for the particles.
	result = LoadTexture(device, texture_filename);
	if(!result)	{
		return false;
	}

	// Initialize the particle system.
	result = InitializeParticleSystem();
	if(!result)	{
		return false;
	}

	// Create the buffers that will be used to render the particles with.
	result = InitializeBuffers(device);
	if(!result)	{
		return false;
	}

	return true;
}
void ParticleSystemClass::Shutdown(){
	// Release the buffers.
	ShutdownBuffers();

	// Release the particle system.
	ShutdownParticleSystem();

	// Release the texture used for the particles.
	ReleaseTexture();

	return;
}

bool ParticleSystemClass::Frame(float frame_time, ID3D11DeviceContext* device_context){
	bool result;
	if(is_clear_ == false){
		// Release old particles.
		KillParticles();

		// Emit new particles.
		EmitParticles(frame_time);
	
		// Update the position of the particles.
		UpdateParticles(frame_time);

		// Update the dynamic vertex buffer with the new position of each particle.
		result = UpdateBuffers(device_context);
		if(!result){
			return false;
		}
	}
	return true;
}

void ParticleSystemClass::Render(ID3D11DeviceContext* device_context){
	if(is_clear_ == false){
		// Put the vertex and index buffers on the graphics pipeline to prepare them for drawing.
		RenderBuffers(device_context);
	}
	return;
}

bool ParticleSystemClass::LoadTexture(ID3D11Device* device, WCHAR* filename){
	bool result;

	// Create the texture object.
	texture_ = new TextureClass;
	if(!texture_){
		return false;
	}

	// Initialize the texture object.
	result = texture_->Initialize(device, filename);
	if(!result){
		return false;
	}

	return true;
}

void ParticleSystemClass::ReleaseTexture(){
	// Release the texture object.
	if(texture_){
		texture_->Shutdown();
		delete texture_;
		texture_ = 0;
	}

	return;
}
void ParticleSystemClass::UpdateParticleSystem(D3DXVECTOR3 part_dev, D3DXVECTOR2 velo, D3DXVECTOR3 part_feat){
	// Set the random deviation of where the particles can be located when emitted.
	particle_deviation_x_ = part_dev.x;
	particle_deviation_y_ = part_dev.y;
	particle_deviation_z_ = part_dev.z;

	// Set the speed and speed variation of particles.
	particle_velocity_ = velo.x;
	particle_velocity_variation_ = velo.y;

	// Set the physical size of the particles.
	particle_size_ = part_feat.x;

	// Set the number of particles to emit per second.
	particles_per_second_ = part_feat.y;

	// Set the maximum number of particles allowed in the particle system.
	max_particles_ = int(part_feat.z);

	// Initialize the particle list.
	for(int i=0; i<MAX_NUM_PARTICLES; i++){
		particle_list_[i].active_ = false;
	}

	// Initialize the current particle count to zero since none are emitted yet.
	current_particle_count_ = 0;

	// Clear the initial accumulated time for the particle per second emission rate.
	accumulated_time_ = 0.0f;

}
bool ParticleSystemClass::InitializeParticleSystem(){
	// Set the random deviation of where the particles can be located when emitted.
	particle_deviation_x_ = 1.5f;
	particle_deviation_y_ = 1.1f;
	particle_deviation_z_ = 2.0f;

	// Set the speed and speed variation of particles.
	particle_velocity_ = 16.0f;
	particle_velocity_variation_ = 0.0f;

	// Set the physical size of the particles.
	particle_size_ =0.04f;

	// Set the number of particles to emit per second.
	particles_per_second_ = 50.0f;

	// Set the maximum number of particles allowed in the particle system.
	max_particles_ = MAX_NUM_PARTICLES;

	// Create the particle list.
	particle_list_ = new ParticleType[MAX_NUM_PARTICLES];
	if(!particle_list_){
		return false;
	}

	// Initialize the particle list.
	for(int i=0; i<max_particles_; i++){
		particle_list_[i].active_ = false;
		particle_list_[i].position_x_ = 0.0f;
		particle_list_[i].position_y_ = 0.0f;
		particle_list_[i].position_z_ = -5.0f;
		particle_list_[i].red_ = 0.f;
		particle_list_[i].green_ = 0.f;
		particle_list_[i].blue_ = 0.f;
		particle_list_[i].velocity_ = particle_velocity_;
	}


	// Initialize the current particle count to zero since none are emitted yet.
	current_particle_count_ = 0;

	// Clear the initial accumulated time for the particle per second emission rate.
	accumulated_time_ = 0.0f;

	return true;
}

void ParticleSystemClass::ShutdownParticleSystem(){
	// Release the particle list.
	if(particle_list_){
		delete [] particle_list_;
		particle_list_ = 0;
	}

	return;
}

bool ParticleSystemClass::InitializeBuffers(ID3D11Device* device){
	D3D11_BUFFER_DESC vertex_buffer_desc, instance_buffer_desc;
	D3D11_SUBRESOURCE_DATA vertex_data, instance_data;
	HRESULT result;

	// Set the maximum number of vertices in the vertex array.
	vertex_count_ = 6;

	int index = 0;
	int i = 0;

	// Bottom left.
	vertices_[index].position_ = D3DXVECTOR3(particle_list_[i].position_x_ - particle_size_, particle_list_[i].position_y_ - (particle_size_*2), particle_list_[i].position_z_);
	vertices_[index].texture_ = D3DXVECTOR2(0.0f, 1.0f);
	vertices_[index].color_ = D3DXVECTOR4(particle_list_[i].red_, particle_list_[i].green_, particle_list_[i].blue_, 1.0f);
	index++;

	// Top left.
	vertices_[index].position_ = D3DXVECTOR3(particle_list_[i].position_x_ - particle_size_, particle_list_[i].position_y_ + (particle_size_*2), particle_list_[i].position_z_);
	vertices_[index].texture_ = D3DXVECTOR2(0.0f, 0.0f);
	vertices_[index].color_ = D3DXVECTOR4(particle_list_[i].red_, particle_list_[i].green_, particle_list_[i].blue_,1.0f);
	index++;

	// Top right.
	vertices_[index].position_ = D3DXVECTOR3(particle_list_[i].position_x_ + particle_size_, particle_list_[i].position_y_ + (particle_size_*2), particle_list_[i].position_z_);
	vertices_[index].texture_ = D3DXVECTOR2(1.0f, 0.0f);
	vertices_[index].color_ = D3DXVECTOR4(particle_list_[i].red_, particle_list_[i].green_, particle_list_[i].blue_, 1.0f);
	index++;

	// Top right.
	vertices_[index].position_ = D3DXVECTOR3(particle_list_[i].position_x_ + particle_size_, particle_list_[i].position_y_ + (particle_size_*2), particle_list_[i].position_z_);
	vertices_[index].texture_ = D3DXVECTOR2(1.0f, 0.0f);
	vertices_[index].color_ = D3DXVECTOR4(particle_list_[i].red_, particle_list_[i].green_, particle_list_[i].blue_,1.0f);
	index++;

	// Bottom right.
	vertices_[index].position_ = D3DXVECTOR3(particle_list_[i].position_x_ + particle_size_, particle_list_[i].position_y_ - (particle_size_*2), particle_list_[i].position_z_);
	vertices_[index].texture_ = D3DXVECTOR2(1.0f, 1.0f);
	vertices_[index].color_ = D3DXVECTOR4(particle_list_[i].red_, particle_list_[i].green_, particle_list_[i].blue_, 1.0f);
	index++;

	// Bottom left.
	vertices_[index].position_ = D3DXVECTOR3(particle_list_[i].position_x_ - particle_size_, particle_list_[i].position_y_ - (particle_size_*2), particle_list_[i].position_z_);
	vertices_[index].texture_ = D3DXVECTOR2(0.0f, 1.0f);
	vertices_[index].color_ = D3DXVECTOR4(particle_list_[i].red_, particle_list_[i].green_, particle_list_[i].blue_,1.0f);
	index++;

	// Set up the description of the dynamic vertex buffer.
	vertex_buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
	vertex_buffer_desc.ByteWidth = sizeof(VertexType) * vertex_count_;
	vertex_buffer_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vertex_buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	vertex_buffer_desc.MiscFlags = 0;
	vertex_buffer_desc.StructureByteStride = 0;

	// Give the subresource structure a pointer to the vertex data.
	vertex_data.pSysMem = vertices_;
	vertex_data.SysMemPitch = 0;
	vertex_data.SysMemSlicePitch = 0;

	// Now finally create the vertex buffer.
	result = device->CreateBuffer(&vertex_buffer_desc, &vertex_data, &vertex_buffer_);
	if(FAILED(result)){
		return false;
	}

	// Set the maximum number of indices in the index array.
	instance_count_ = max_particles_;

	for(int i=0; i<instance_count_; i++){
		instances_[i].position_ = D3DXVECTOR4(particle_list_[i].position_x_ , particle_list_[i].position_y_, particle_list_[i].position_z_,-1);
	}

	// Set up the description of the instance buffer.
	instance_buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
	instance_buffer_desc.ByteWidth = sizeof(InstanceType) * instance_count_;
	instance_buffer_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	instance_buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	instance_buffer_desc.MiscFlags = 0;
	instance_buffer_desc.StructureByteStride = 0;
	
	// Give the subresource structure a pointer to the instance data.
	instance_data.pSysMem = instances_;
	instance_data.SysMemPitch = 0;
	instance_data.SysMemSlicePitch = 0;

	// Create the instance buffer.
	result = device->CreateBuffer(&instance_buffer_desc, &instance_data, &instance_buffer_);
	if(FAILED(result)){
		return false;
	}
	return true;
}

void ParticleSystemClass::ShutdownBuffers(){
	// Release the index buffer.
	if(instance_buffer_){
		instance_buffer_->Release();
		instance_buffer_ = 0;
	}

	// Release the vertex buffer.
	if(vertex_buffer_){
		vertex_buffer_->Release();
		vertex_buffer_ = 0;
	}

	return;
}

void ParticleSystemClass::EmitParticles(float frame_time){
	bool emitParticle, found;
	float position_x_, position_y_, position_z_, velocity, red, green, blue;
	int index, i, j;


	// Increment the frame time.
	accumulated_time_ += frame_time;

	// Set emit particle to false for now.
	emitParticle = false;
	
	// Check if it is time to emit a new particle or not.
	if(accumulated_time_ > (1000.0f / particles_per_second_)){
		accumulated_time_ = 0.0f;
		emitParticle = true;
	}

	// If there are particles to emit then emit one per frame.
	if((emitParticle == true) && (current_particle_count_ < (max_particles_ - 1))){
		current_particle_count_++;

		// Now generate the randomized particle properties.
		position_x_ = (((float)rand()-(float)rand())/RAND_MAX) * particle_deviation_x_;
		position_y_ = (((float)rand()-(float)rand())/RAND_MAX) * particle_deviation_y_;
		position_z_ = (((float)rand()-(float)rand())/RAND_MAX) * particle_deviation_z_;

		velocity = particle_velocity_ + (((float)rand()-(float)rand())/RAND_MAX) * particle_velocity_variation_;

		red   = (((float)rand()-(float)rand())/RAND_MAX) + 0.5f;
		green = (((float)rand()-(float)rand())/RAND_MAX) + 0.5f;
		blue  = (((float)rand()-(float)rand())/RAND_MAX) + 0.5f;

		// Now since the particles need to be rendered from back to front for blending we have to sort the particle array.
		// We will sort using Z depth so we need to find where in the list the particle should be inserted.
		index = 0;
		found = false;
		while(!found){
			if((particle_list_[index].active_ == false) || (particle_list_[index].position_z_ < position_z_)){
				found = true;
			}else{
				index++;
			}
		}

		// Now that we know the location to insert into we need to copy the array over by one position from the index to make room for the new particle.
		i = current_particle_count_;
		j = i - 1;

		while(i != index){
			particle_list_[i].position_x_ = particle_list_[j].position_x_;
			particle_list_[i].position_y_ = particle_list_[j].position_y_;
			particle_list_[i].position_z_ = particle_list_[j].position_z_;
			particle_list_[i].red_       = particle_list_[j].red_;
			particle_list_[i].green_     = particle_list_[j].green_;
			particle_list_[i].blue_      = particle_list_[j].blue_;
			particle_list_[i].velocity_  = particle_list_[j].velocity_;
			particle_list_[i].active_    = particle_list_[j].active_;
			i--;
			j--;
		}

		// Now insert it into the particle array in the correct depth order.
		particle_list_[index].position_x_ = position_x_;
		particle_list_[index].position_y_ = position_y_;
		particle_list_[index].position_z_ = position_z_;
		particle_list_[index].red_       = red;
		particle_list_[index].green_     = green;
		particle_list_[index].blue_      = blue;
		particle_list_[index].velocity_  = velocity;
		particle_list_[index].active_    = true;
	}

	return;
}


void ParticleSystemClass::UpdateParticles(float frame_time){
	// Each frame we update all the particles by making them move downwards using their position, velocity, and the frame time.
	for(int i=0; i<current_particle_count_; i++){
		particle_list_[i].position_y_ = particle_list_[i].position_y_ - (particle_list_[i].velocity_ * frame_time * 0.001f);
	}

	return;
}

void ParticleSystemClass::KillParticles(){
	// Kill all the particles that have gone below a certain height range.
	for(int i=0; i<max_particles_; i++){
		if((particle_list_[i].active_ == true) && (particle_list_[i].position_y_ < -20.05f)){
			particle_list_[i].active_ = false;
			current_particle_count_--;
			kill_count_++;
			// Now shift all the live particles back up the array to erase the destroyed particle and keep the array sorted correctly.
			for(int j=i; j<max_particles_-1; j++){
				particle_list_[j].position_x_ = particle_list_[j+1].position_x_;
				particle_list_[j].position_y_ = particle_list_[j+1].position_y_;
				particle_list_[j].position_z_ = particle_list_[j+1].position_z_;
				particle_list_[j].red_       = particle_list_[j+1].red_;
				particle_list_[j].green_     = particle_list_[j+1].green_;
				particle_list_[j].blue_      = particle_list_[j+1].blue_;
				particle_list_[j].velocity_  = particle_list_[j+1].velocity_;
				particle_list_[j].active_    = particle_list_[j+1].active_;
			}
		}
	}

	return;
}

bool ParticleSystemClass::UpdateBuffers(ID3D11DeviceContext* device_context){
	HRESULT result;
	D3D11_MAPPED_SUBRESOURCE mapped_resource;
	InstanceType* instance_ptr;

	for(int i=0; i<instance_count_; i++){
		if(particle_list_[i].active_ == false){
			instances_[i].position_ = D3DXVECTOR4(particle_list_[i].position_x_ , particle_list_[i].position_y_, particle_list_[i].position_z_,-1.f);
		}else{
			instances_[i].position_ = D3DXVECTOR4(particle_list_[i].position_x_ , particle_list_[i].position_y_, particle_list_[i].position_z_,1.f);
		}
	}
	
	// Lock the vertex buffer.
	result = device_context->Map(instance_buffer_, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped_resource);
	if(FAILED(result)){
		return false;
	}

	// Get a pointer to the data in the vertex buffer.
	instance_ptr = (InstanceType*)mapped_resource.pData;

	// Copy the data into the vertex buffer.
	memcpy(instance_ptr, (void*)instances_, (sizeof(InstanceType) * instance_count_));

	// Unlock the vertex buffer.
	device_context->Unmap(instance_buffer_, 0);
	if(FAILED(result)){
		return false;
	}
	return true;
}

void ParticleSystemClass::RenderBuffers(ID3D11DeviceContext* device_context){
	unsigned int strides[2];
	unsigned int offsets[2];
	ID3D11Buffer* bufferPointers[2];

	// Set the buffer strides.
	strides[0] = sizeof(VertexType); 
	strides[1] = sizeof(InstanceType); 

	// Set the buffer offsets.
	offsets[0] = 0;
	offsets[1] = 0;
	
	// Set the array of pointers to the vertex and instance buffers.
	bufferPointers[0] = vertex_buffer_;
	bufferPointers[1] = instance_buffer_;

	// Set the vertex buffer to active in the input assembler so it can be rendered.
	device_context->IASetVertexBuffers(0, 2, bufferPointers, strides, offsets);

	// Set the type of primitive that should be rendered from this vertex buffer.
	device_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	return;
}