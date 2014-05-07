#pragma once

struct Size{
	int width_;
	int height_;
	int depth_;
	size_t pitch_;
	size_t pitch_slice_;
};

#define PIXEL_FMT_SIZE_RGBA 4
#define PIXEL_FMT_SIZE_RG 2

#define x_identifier_ 0
#define y_identifier_ 1
#define z_identifier_ 2

#define qv_identifier_ 0
#define qc_identifier_ 1
#define qr_identifier_ 0
#define F_identifier_ 1

#define theta_identifier_ 0
#define theta_advect_identifier_ 1

#define time_step 1.f
#define dx 1.f
#define T0 295.f
#define gamma 10.f/100.f
#define p0 100000
#define aT 5e-4
#define alpha 1e-3
#define beta 2.f
#define b1 1000.f
#define V 4.f
#define W 8.f
#define g 9.8f
#define R 287.f
#define epsilon 18.02f/29.87f
#define a 17.27f
#define b 35.86f
#define es0 100.f*3.8f
#define z_alt 1000
#define latent_heat 2.501
#define cp 1005
#define k 0.286
#define T T0-gamma*z_alt