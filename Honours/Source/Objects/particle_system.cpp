////////////////////////////////////////////////////////////////////////////////
// Filename: particlesystemclass.cpp
////////////////////////////////////////////////////////////////////////////////
#include "particle_system.h"

ParticleSystemClass::ParticleSystemClass(){
	texture_ = 0;
	particle_list_ = 0;
	vertices_ = 0;
	vertex_buffer_ = 0;
	index_buffer_ = 0;
	D3DXMatrixTranslation(&translation_, float(rand()%129), float(rand()%10), float(rand()%129));
}

ParticleSystemClass::ParticleSystemClass(const ParticleSystemClass& other){
}

ParticleSystemClass::~ParticleSystemClass(){
}

bool ParticleSystemClass::Initialize(ID3D11Device* device, WCHAR* texture_filename){
	bool result;

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

	// Release old particles.
	KillParticles();

	// Emit new particles.
	EmitParticles(frame_time);
	
	// Update the position of the particles.
	UpdateParticles(frame_time);

	// Update the dynamic vertex buffer with the new position of each particle.
	result = UpdateBuffers(device_context);
	if(!result)	{
		return false;
	}

	return true;
}

void ParticleSystemClass::Render(ID3D11DeviceContext* device_context){
	// Put the vertex and index buffers on the graphics pipeline to prepare them for drawing.
	RenderBuffers(device_context);

	return;
}

ID3D11ShaderResourceView* ParticleSystemClass::GetTexture(){
	return texture_->GetTexture();
}

int ParticleSystemClass::GetIndexCount(){
	return index_count_;
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

bool ParticleSystemClass::InitializeParticleSystem(){
	// Set the random deviation of where the particles can be located when emitted.
	particle_deviation_x_ = 1.5f;
	particle_deviation_y_ = 1.1f;
	particle_deviation_z_ = 2.0f;

	// Set the speed and speed variation of particles.
	particle_velocity_ = 1.0f;
	particle_velocity_variation_ = 0.2f;

	// Set the physical size of the particles.
	particle_size_ = 0.1f;

	// Set the number of particles to emit per second.
	particles_per_second_ = 500.0f;

	// Set the maximum number of particles allowed in the particle system.
	max_particles_ = 10000;

	// Create the particle list.
	particle_list_ = new ParticleType[max_particles_];
	if(!particle_list_){
		return false;
	}

	// Initialize the particle list.
	for(int i=0; i<max_particles_; i++){
		particle_list_[i].active_ = false;
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
	unsigned long* indices;
	D3D11_BUFFER_DESC vertex_buffer_desc, index_buffer_desc;
    D3D11_SUBRESOURCE_DATA vertex_data, index_data;
	HRESULT result;

	// Set the maximum number of vertices in the vertex array.
	vertex_count_ = max_particles_ * 6;

	// Set the maximum number of indices in the index array.
	index_count_ = vertex_count_;

	// Create the vertex array for the particles that will be rendered.
	vertices_ = new VertexType[vertex_count_];
	if(!vertices_){
		return false;
	}

	// Create the index array.
	indices = new unsigned long[index_count_];
	if(!indices){
		return false;
	}

	// Initialize vertex array to zeros at first.
	memset(vertices_, 0, (sizeof(VertexType) * vertex_count_));

	// Initialize the index array.
	for(int i=0; i<index_count_; i++){
		indices[i] = i;
	}

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

	// Set up the description of the static index buffer.
    index_buffer_desc.Usage = D3D11_USAGE_DEFAULT;
    index_buffer_desc.ByteWidth = sizeof(unsigned long) * index_count_;
    index_buffer_desc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    index_buffer_desc.CPUAccessFlags = 0;
    index_buffer_desc.MiscFlags = 0;
	index_buffer_desc.StructureByteStride = 0;

	// Give the subresource structure a pointer to the index data.
    index_data.pSysMem = indices;
	index_data.SysMemPitch = 0;
	index_data.SysMemSlicePitch = 0;

	// Create the index buffer.
	result = device->CreateBuffer(&index_buffer_desc, &index_data, &index_buffer_);
	if(FAILED(result)){
		return false;
	}

	// Release the index array since it is no longer needed.
	delete [] indices;
	indices = 0;

	return true;
}

void ParticleSystemClass::ShutdownBuffers(){
	// Release the index buffer.
	if(index_buffer_){
		index_buffer_->Release();
		index_buffer_ = 0;
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
		if((particle_list_[i].active_ == true) && (particle_list_[i].position_y_ < -3.0f)){
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
	int index;
	HRESULT result;
	D3D11_MAPPED_SUBRESOURCE mapped_resource;
	VertexType* verticesPtr;

	// Initialize vertex array to zeros at first.
	memset(vertices_, 0, (sizeof(VertexType) * vertex_count_));

	// Now build the vertex array from the particle list array.  Each particle is a quad made out of two triangles.
	index = 0;

	for(int i=0; i<current_particle_count_; i++){
		// Bottom left.
		vertices_[index].position_ = D3DXVECTOR3(particle_list_[i].position_x_ - particle_size_, particle_list_[i].position_y_ - particle_size_, particle_list_[i].position_z_);
		vertices_[index].texture_ = D3DXVECTOR2(0.0f, 1.0f);
		vertices_[index].color_ = D3DXVECTOR4(particle_list_[i].red_, particle_list_[i].green_, particle_list_[i].blue_, 1.0f);
		index++;

		// Top left.
		vertices_[index].position_ = D3DXVECTOR3(particle_list_[i].position_x_ - particle_size_, particle_list_[i].position_y_ + particle_size_, particle_list_[i].position_z_);
		vertices_[index].texture_ = D3DXVECTOR2(0.0f, 0.0f);
		vertices_[index].color_ = D3DXVECTOR4(particle_list_[i].red_, particle_list_[i].green_, particle_list_[i].blue_, 1.0f);
		index++;

		// Bottom right.
		vertices_[index].position_ = D3DXVECTOR3(particle_list_[i].position_x_ + particle_size_, particle_list_[i].position_y_ - particle_size_, particle_list_[i].position_z_);
		vertices_[index].texture_ = D3DXVECTOR2(1.0f, 1.0f);
		vertices_[index].color_ = D3DXVECTOR4(particle_list_[i].red_, particle_list_[i].green_, particle_list_[i].blue_, 1.0f);
		index++;

		// Bottom right.
		vertices_[index].position_ = D3DXVECTOR3(particle_list_[i].position_x_ + particle_size_, particle_list_[i].position_y_ - particle_size_, particle_list_[i].position_z_);
		vertices_[index].texture_ = D3DXVECTOR2(1.0f, 1.0f);
		vertices_[index].color_ = D3DXVECTOR4(particle_list_[i].red_, particle_list_[i].green_, particle_list_[i].blue_, 1.0f);
		index++;

		// Top left.
		vertices_[index].position_ = D3DXVECTOR3(particle_list_[i].position_x_ - particle_size_, particle_list_[i].position_y_ + particle_size_, particle_list_[i].position_z_);
		vertices_[index].texture_ = D3DXVECTOR2(0.0f, 0.0f);
		vertices_[index].color_ = D3DXVECTOR4(particle_list_[i].red_, particle_list_[i].green_, particle_list_[i].blue_, 1.0f);
		index++;

		// Top right.
		vertices_[index].position_ = D3DXVECTOR3(particle_list_[i].position_x_ + particle_size_, particle_list_[i].position_y_ + particle_size_, particle_list_[i].position_z_);
		vertices_[index].texture_ = D3DXVECTOR2(1.0f, 0.0f);
		vertices_[index].color_ = D3DXVECTOR4(particle_list_[i].red_, particle_list_[i].green_, particle_list_[i].blue_, 1.0f);
		index++;
	}
	
	// Lock the vertex buffer.
	result = device_context->Map(vertex_buffer_, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped_resource);
	if(FAILED(result)){
		return false;
	}

	// Get a pointer to the data in the vertex buffer.
	verticesPtr = (VertexType*)mapped_resource.pData;

	// Copy the data into the vertex buffer.
	memcpy(verticesPtr, (void*)vertices_, (sizeof(VertexType) * vertex_count_));

	// Unlock the vertex buffer.
	device_context->Unmap(vertex_buffer_, 0);

	return true;
}

void ParticleSystemClass::RenderBuffers(ID3D11DeviceContext* device_context){
	unsigned int stride;
	unsigned int offset;

	// Set vertex buffer stride and offset.
    stride = sizeof(VertexType); 
	offset = 0;
    
	// Set the vertex buffer to active in the input assembler so it can be rendered.
	device_context->IASetVertexBuffers(0, 1, &vertex_buffer_, &stride, &offset);

    // Set the index buffer to active in the input assembler so it can be rendered.
    device_context->IASetIndexBuffer(index_buffer_, DXGI_FORMAT_R32_UINT, 0);

    // Set the type of primitive that should be rendered from this vertex buffer.
    device_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	return;
}