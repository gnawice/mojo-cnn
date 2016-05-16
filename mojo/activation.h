// == mojo ====================================================================
//
//    Copyright (c) gnawice@gnawice.com. All rights reserved.
//	  See LICENSE in root folder
//
//    Permission is hereby granted, free of charge, to any person obtaining a
//    copy of this software and associated documentation files(the "Software"),
//    to deal in the Software without restriction, including without 
//    limitation the rights to use, copy, modify, merge, publish, distribute,
//    sublicense, and/or sell copies of the Software, and to permit persons to
//    whom the Software is furnished to do so, subject to the following 
//    conditions :
//
//    The above copyright notice and this permission notice shall be included
//    in all copies or substantial portions of the Software.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
//    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
//    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
//    OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
//    THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// ============================================================================
//    activation.h:  neuron activation functions
// ==================================================================== mojo ==

#pragma once

#include <math.h>
#include <algorithm>
#include <string>

namespace mojo {

#ifdef MOJO_LUTS
		const float_t tanh_lut[1024] = { -1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,
		-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,
		-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,
		-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999999f,-0.999998f,-0.999998f,
		-0.999998f,-0.999998f,-0.999998f,-0.999998f,-0.999998f,-0.999998f,-0.999998f,-0.999998f,-0.999998f,-0.999998f,-0.999998f,-0.999998f,-0.999998f,-0.999997f,-0.999997f,-0.999997f,
		-0.999997f,-0.999997f,-0.999997f,-0.999997f,-0.999997f,-0.999997f,-0.999997f,-0.999996f,-0.999996f,-0.999996f,-0.999996f,-0.999996f,-0.999996f,-0.999996f,-0.999996f,-0.999995f,
		-0.999995f,-0.999995f,-0.999995f,-0.999995f,-0.999995f,-0.999995f,-0.999994f,-0.999994f,-0.999994f,-0.999994f,-0.999994f,-0.999993f,-0.999993f,-0.999993f,-0.999993f,-0.999992f,
		-0.999992f,-0.999992f,-0.999992f,-0.999992f,-0.999991f,-0.999991f,-0.999991f,-0.99999f,-0.99999f,-0.99999f,-0.99999f,-0.999989f,-0.999989f,-0.999988f,-0.999988f,-0.999988f,
		-0.999987f,-0.999987f,-0.999987f,-0.999986f,-0.999986f,-0.999985f,-0.999985f,-0.999984f,-0.999984f,-0.999983f,-0.999983f,-0.999982f,-0.999981f,-0.999981f,-0.99998f,-0.99998f,
		-0.999979f,-0.999978f,-0.999978f,-0.999977f,-0.999976f,-0.999975f,-0.999975f,-0.999974f,-0.999973f,-0.999972f,-0.999971f,-0.99997f,-0.99997f,-0.999969f,-0.999968f,-0.999967f,
		-0.999966f,-0.999964f,-0.999963f,-0.999962f,-0.999961f,-0.99996f,-0.999958f,-0.999957f,-0.999956f,-0.999954f,-0.999953f,-0.999951f,-0.99995f,-0.999948f,-0.999947f,-0.999945f,
		-0.999943f,-0.999941f,-0.99994f,-0.999938f,-0.999936f,-0.999934f,-0.999931f,-0.999929f,-0.999927f,-0.999925f,-0.999922f,-0.99992f,-0.999917f,-0.999915f,-0.999912f,-0.999909f,
		-0.999906f,-0.999903f,-0.9999f,-0.999897f,-0.999894f,-0.99989f,-0.999887f,-0.999884f,-0.99988f,-0.999876f,-0.999872f,-0.999868f,-0.999864f,-0.999859f,-0.999855f,-0.99985f,
		-0.999846f,-0.999841f,-0.999836f,-0.99983f,-0.999825f,-0.99982f,-0.999814f,-0.999808f,-0.999802f,-0.999795f,-0.999789f,-0.999782f,-0.999775f,-0.999768f,-0.999761f,-0.999753f,
		-0.999745f,-0.999737f,-0.999729f,-0.99972f,-0.999712f,-0.999702f,-0.999693f,-0.999683f,-0.999673f,-0.999663f,-0.999652f,-0.999641f,-0.99963f,-0.999618f,-0.999606f,-0.999593f,
		-0.99958f,-0.999567f,-0.999553f,-0.999539f,-0.999524f,-0.999509f,-0.999494f,-0.999478f,-0.999461f,-0.999444f,-0.999426f,-0.999408f,-0.999389f,-0.99937f,-0.99935f,-0.999329f,
		-0.999308f,-0.999286f,-0.999263f,-0.99924f,-0.999216f,-0.999191f,-0.999165f,-0.999139f,-0.999112f,-0.999083f,-0.999054f,-0.999024f,-0.998993f,-0.998961f,-0.998929f,-0.998894f,
		-0.998859f,-0.998823f,-0.998786f,-0.998747f,-0.998708f,-0.998667f,-0.998624f,-0.998581f,-0.998536f,-0.998489f,-0.998441f,-0.998392f,-0.998341f,-0.998288f,-0.998234f,-0.998178f,
		-0.99812f,-0.998061f,-0.997999f,-0.997936f,-0.99787f,-0.997803f,-0.997733f,-0.997661f,-0.997587f,-0.99751f,-0.997431f,-0.99735f,-0.997266f,-0.997179f,-0.99709f,-0.996998f,
		-0.996902f,-0.996804f,-0.996703f,-0.996599f,-0.996491f,-0.99638f,-0.996265f,-0.996146f,-0.996024f,-0.995898f,-0.995769f,-0.995635f,-0.995496f,-0.995354f,-0.995207f,-0.995055f,
		-0.994898f,-0.994737f,-0.99457f,-0.994398f,-0.994221f,-0.994038f,-0.993849f,-0.993655f,-0.993454f,-0.993247f,-0.993033f,-0.992813f,-0.992585f,-0.992351f,-0.992109f,-0.99186f,
		-0.991602f,-0.991337f,-0.991063f,-0.990781f,-0.99049f,-0.990189f,-0.989879f,-0.98956f,-0.98923f,-0.98889f,-0.98854f,-0.988178f,-0.987805f,-0.98742f,-0.987023f,-0.986614f,
		-0.986192f,-0.985757f,-0.985308f,-0.984845f,-0.984368f,-0.983876f,-0.983368f,-0.982845f,-0.982305f,-0.981749f,-0.981175f,-0.980583f,-0.979973f,-0.979344f,-0.978695f,-0.978026f,
		-0.977336f,-0.976626f,-0.975892f,-0.975137f,-0.974357f,-0.973554f,-0.972726f,-0.971873f,-0.970993f,-0.970086f,-0.969151f,-0.968187f,-0.967194f,-0.96617f,-0.965115f,-0.964028f,
		-0.962907f,-0.961752f,-0.960562f,-0.959335f,-0.958072f,-0.956769f,-0.955428f,-0.954045f,-0.952621f,-0.951154f,-0.949642f,-0.948085f,-0.946481f,-0.944829f,-0.943128f,-0.941376f,
		-0.939571f,-0.937712f,-0.935799f,-0.933828f,-0.931799f,-0.92971f,-0.92756f,-0.925346f,-0.923068f,-0.920722f,-0.918309f,-0.915825f,-0.913269f,-0.910638f,-0.907932f,-0.905148f,
		-0.902284f,-0.899339f,-0.896309f,-0.893193f,-0.889989f,-0.886695f,-0.883308f,-0.879827f,-0.876248f,-0.87257f,-0.86879f,-0.864907f,-0.860916f,-0.856818f,-0.852607f,-0.848284f,
		-0.843844f,-0.839285f,-0.834605f,-0.829802f,-0.824872f,-0.819814f,-0.814624f,-0.809301f,-0.803841f,-0.798243f,-0.792503f,-0.786619f,-0.780588f,-0.774409f,-0.768079f,-0.761594f,
		-0.754954f,-0.748155f,-0.741195f,-0.734071f,-0.726783f,-0.719328f,-0.711702f,-0.703906f,-0.695935f,-0.68779f,-0.679468f,-0.670967f,-0.662286f,-0.653424f,-0.644378f,-0.635149f,
		-0.625735f,-0.616134f,-0.606348f,-0.596374f,-0.586212f,-0.575862f,-0.565325f,-0.5546f,-0.543687f,-0.532587f,-0.521301f,-0.50983f,-0.498174f,-0.486336f,-0.474316f,-0.462117f,
		-0.449741f,-0.437189f,-0.424464f,-0.41157f,-0.398509f,-0.385284f,-0.371899f,-0.358357f,-0.344663f,-0.330821f,-0.316835f,-0.30271f,-0.28845f,-0.274062f,-0.259549f,-0.244919f,
		-0.230176f,-0.215326f,-0.200377f,-0.185333f,-0.170202f,-0.154991f,-0.139705f,-0.124353f,-0.108941f,-0.0934763f,-0.0779665f,-0.0624188f,-0.0468407f,-0.0312398f,-0.0156237f,0.f,
		0.0156237f,0.0312398f,0.0468407f,0.0624188f,0.0779665f,0.0934763f,0.108941f,0.124353f,0.139705f,0.154991f,0.170202f,0.185333f,0.200377f,0.215326f,0.230176f,0.244919f,
		0.259549f,0.274062f,0.28845f,0.30271f,0.316835f,0.330821f,0.344663f,0.358357f,0.371899f,0.385284f,0.398509f,0.41157f,0.424464f,0.437189f,0.449741f,0.462117f,
		0.474316f,0.486336f,0.498174f,0.50983f,0.521301f,0.532587f,0.543687f,0.5546f,0.565325f,0.575862f,0.586212f,0.596374f,0.606348f,0.616134f,0.625735f,0.635149f,
		0.644378f,0.653424f,0.662286f,0.670967f,0.679468f,0.68779f,0.695935f,0.703906f,0.711702f,0.719328f,0.726783f,0.734071f,0.741195f,0.748155f,0.754954f,0.761594f,
		0.768079f,0.774409f,0.780588f,0.786619f,0.792503f,0.798243f,0.803841f,0.809301f,0.814624f,0.819814f,0.824872f,0.829802f,0.834605f,0.839285f,0.843844f,0.848284f,
		0.852607f,0.856818f,0.860916f,0.864907f,0.86879f,0.87257f,0.876248f,0.879827f,0.883308f,0.886695f,0.889989f,0.893193f,0.896309f,0.899339f,0.902284f,0.905148f,
		0.907932f,0.910638f,0.913269f,0.915825f,0.918309f,0.920722f,0.923068f,0.925346f,0.92756f,0.92971f,0.931799f,0.933828f,0.935799f,0.937712f,0.939571f,0.941376f,
		0.943128f,0.944829f,0.946481f,0.948085f,0.949642f,0.951154f,0.952621f,0.954045f,0.955428f,0.956769f,0.958072f,0.959335f,0.960562f,0.961752f,0.962907f,0.964028f,
		0.965115f,0.96617f,0.967194f,0.968187f,0.969151f,0.970086f,0.970993f,0.971873f,0.972726f,0.973554f,0.974357f,0.975137f,0.975892f,0.976626f,0.977336f,0.978026f,
		0.978695f,0.979344f,0.979973f,0.980583f,0.981175f,0.981749f,0.982305f,0.982845f,0.983368f,0.983876f,0.984368f,0.984845f,0.985308f,0.985757f,0.986192f,0.986614f,
		0.987023f,0.98742f,0.987805f,0.988178f,0.98854f,0.98889f,0.98923f,0.98956f,0.989879f,0.990189f,0.99049f,0.990781f,0.991063f,0.991337f,0.991602f,0.99186f,
		0.992109f,0.992351f,0.992585f,0.992813f,0.993033f,0.993247f,0.993454f,0.993655f,0.993849f,0.994038f,0.994221f,0.994398f,0.99457f,0.994737f,0.994898f,0.995055f,
		0.995207f,0.995354f,0.995496f,0.995635f,0.995769f,0.995898f,0.996024f,0.996146f,0.996265f,0.99638f,0.996491f,0.996599f,0.996703f,0.996804f,0.996902f,0.996998f,
		0.99709f,0.997179f,0.997266f,0.99735f,0.997431f,0.99751f,0.997587f,0.997661f,0.997733f,0.997803f,0.99787f,0.997936f,0.997999f,0.998061f,0.99812f,0.998178f,
		0.998234f,0.998288f,0.998341f,0.998392f,0.998441f,0.998489f,0.998536f,0.998581f,0.998624f,0.998667f,0.998708f,0.998747f,0.998786f,0.998823f,0.998859f,0.998894f,
		0.998929f,0.998961f,0.998993f,0.999024f,0.999054f,0.999083f,0.999112f,0.999139f,0.999165f,0.999191f,0.999216f,0.99924f,0.999263f,0.999286f,0.999308f,0.999329f,
		0.99935f,0.99937f,0.999389f,0.999408f,0.999426f,0.999444f,0.999461f,0.999478f,0.999494f,0.999509f,0.999524f,0.999539f,0.999553f,0.999567f,0.99958f,0.999593f,
		0.999606f,0.999618f,0.99963f,0.999641f,0.999652f,0.999663f,0.999673f,0.999683f,0.999693f,0.999702f,0.999712f,0.99972f,0.999729f,0.999737f,0.999745f,0.999753f,
		0.999761f,0.999768f,0.999775f,0.999782f,0.999789f,0.999795f,0.999802f,0.999808f,0.999814f,0.99982f,0.999825f,0.99983f,0.999836f,0.999841f,0.999846f,0.99985f,
		0.999855f,0.999859f,0.999864f,0.999868f,0.999872f,0.999876f,0.99988f,0.999884f,0.999887f,0.99989f,0.999894f,0.999897f,0.9999f,0.999903f,0.999906f,0.999909f,
		0.999912f,0.999915f,0.999917f,0.99992f,0.999922f,0.999925f,0.999927f,0.999929f,0.999931f,0.999934f,0.999936f,0.999938f,0.99994f,0.999941f,0.999943f,0.999945f,
		0.999947f,0.999948f,0.99995f,0.999951f,0.999953f,0.999954f,0.999956f,0.999957f,0.999958f,0.99996f,0.999961f,0.999962f,0.999963f,0.999964f,0.999966f,0.999967f,
		0.999968f,0.999969f,0.99997f,0.99997f,0.999971f,0.999972f,0.999973f,0.999974f,0.999975f,0.999975f,0.999976f,0.999977f,0.999978f,0.999978f,0.999979f,0.99998f,
		0.99998f,0.999981f,0.999981f,0.999982f,0.999983f,0.999983f,0.999984f,0.999984f,0.999985f,0.999985f,0.999986f,0.999986f,0.999987f,0.999987f,0.999987f,0.999988f,
		0.999988f,0.999988f,0.999989f,0.999989f,0.99999f,0.99999f,0.99999f,0.99999f,0.999991f,0.999991f,0.999991f,0.999992f,0.999992f,0.999992f,0.999992f,0.999992f,
		0.999993f,0.999993f,0.999993f,0.999993f,0.999994f,0.999994f,0.999994f,0.999994f,0.999994f,0.999995f,0.999995f,0.999995f,0.999995f,0.999995f,0.999995f,0.999995f,
		0.999996f,0.999996f,0.999996f,0.999996f,0.999996f,0.999996f,0.999996f,0.999996f,0.999997f,0.999997f,0.999997f,0.999997f,0.999997f,0.999997f,0.999997f,0.999997f,
		0.999997f,0.999997f,0.999998f,0.999998f,0.999998f,0.999998f,0.999998f,0.999998f,0.999998f,0.999998f,0.999998f,0.999998f,0.999998f,0.999998f,0.999998f,0.999998f,
		0.999998f,0.999999f,0.999999f,0.999999f,0.999999f,0.999999f,0.999999f,0.999999f,0.999999f,0.999999f,0.999999f,0.999999f,0.999999f,0.999999f,0.999999f,0.999999f,
		0.999999f,0.999999f,0.999999f,0.999999f,0.999999f,0.999999f,0.999999f,0.999999f,0.999999f,0.999999f,0.999999f,0.999999f,0.999999f,0.999999f,0.999999f,0.999999f,
		0.999999f,0.999999f,0.999999f,0.999999f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,
		1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f };
#endif

// not using class because I thought this may be faster than vptrs
namespace tan_h 
{
#ifndef MOJO_LUTS
	inline float f(float *in, int i, int size, float bias) // this is activation f(x)
	{
		// in[i] += bias;
        const float ep = std::exp((in[i]+bias));
        const float em = std::exp(-(in[i] + bias));
        return (ep - em) / (ep + em);
    }
#else
	inline float  f(float *in, int i, int size, float bias) // this is activation f(x)
	{
		//in[i] += bias;
		const int index = (int)((in[i] + bias) * 64.0 + 512.);
		if (index >= 1024) return 1.f; // iff exceed max index size
		else if (index<0) return -1.f; // or below min index 0
		return tanh_lut[index];
	}
#endif // MOJO_LUTS	
	inline float  df(float *in, int i, int size) { return (1.f - in[i]*in[i]); }  // this is df(x), but we pass in the activated value f(x) and not x 
	const char name[]="tanh";
}

namespace elu 
{
	inline float  f(float *in, int i, int size, float bias) { if ((in[i] + bias) < 0) return 0.1f*(std::exp((in[i] + bias)) - 1.f); return (in[i] + bias); }
	inline float  df(float *in, int i, int size) { if(in[i] > 0) return 1.f; else return 0.1f*std::exp(in[i]);}
	const char name[]="elu";
}

namespace identity 
{
	inline float  f(float *in, int i, const int size, const float bias) { return (in[i] + bias); }
	inline float  df(float *in, int i, const int size){return 1.f;};
	const char name[]="identity";
}
namespace relu 
{
	inline float  f(float *in, int i, const int size, const float bias) { if ((in[i] + bias) < 0) return 0; return (in[i] + bias); }
	inline float  df(float *in, int i, const int size) {if(in[i] > 0) return 1.0f; else return 0.0f; }
	const char name[]="relu";
};
namespace lrelu 
{
	inline float  f(float *in, int i, const int size, const float bias) { if((in[i] + bias) < 0) return 0.01f*(in[i] + bias); return (in[i] + bias); }
	inline float  df(float *in, int i, const int size) {if(in[i] > 0) return 1.0f; else return 0.01f; }
	const char name[]="lrelu";
};
namespace vlrelu 
{
	inline float  f(float *in, int i, const int size, const float bias) { if((in[i] + bias) < 0) return 0.33f*(in[i] + bias); return (in[i] + bias); }
	inline float  df(float *in, int i, const int size) {if(in[i] > 0) return 1.0f; else return 0.33f; }
	const char name[]="vlrelu";
};

namespace sigmoid
{
	inline float  f(float *in, int i, const int size, const float bias) { return 1.0f/(1.0f+exp(-(in[i] + bias)));}
	inline float df(float *in, int i, const int size) {return in[i]*(1.f-in[i]); }
	const char name[]="sigmoid";
};

namespace softmax
{
	inline float f(float *in, int i, const int size, const float bias)
	{
		float max = in[0];
		for (int j = 1; j<size; j++) if (in[j] > max) max = in[j];

		float denom = 0;
		for (int j = 0; j<size; j++) denom += std::exp(in[j] - max);

		return std::exp(in[i] - max) / denom;
	}

	inline float df(float *in, int i, const int size)
	{
		return in[i] * (1.f - in[i]);
		//		for(int j=0; j<size; j++) 
		//		{
		//			if(i==j) in[i]= in[i] * (1.f - in[i]);
		//			else in[i] = in[i]*in[j];
		//		}
	}

	const char name[] = "softmax";
};

namespace none
{
	inline float f(float *in, int i, int size, float bias) {return 0;};
	inline float df(float *in, int i, int size) {return 0;};
	const char name[]="none";
};

typedef struct 
{
public:
	float (*f)(float *, int, const int, const float);
	float (*df)(float *, int, const int);
	const char *name;
} activation_function;

activation_function* new_activation_function(std::string act)
{
	activation_function *p = new activation_function;
	if(act.compare(tan_h::name)==0) { p->f = &tan_h::f; p->df = &tan_h::df; p->name=tan_h::name;return p;}
	if(act.compare(identity::name)==0) { p->f = &identity::f; p->df = &identity::df; p->name=identity::name; return p;}
	if(act.compare(vlrelu::name)==0) { p->f = &vlrelu::f; p->df = &vlrelu::df; p->name=vlrelu::name; return p;}
	if(act.compare(lrelu::name)==0) { p->f = &lrelu::f; p->df = &lrelu::df; p->name=lrelu::name; return p;}
	if(act.compare(relu::name)==0) { p->f = &relu::f; p->df = &relu::df; p->name=relu::name;return p;}
	if(act.compare(sigmoid::name)==0) { p->f = &sigmoid::f; p->df = &sigmoid::df; p->name=sigmoid::name; return p;}
	if(act.compare(elu::name)==0) { p->f = &elu::f; p->df = &elu::df; p->name=elu::name; return p;}
	if(act.compare(none::name)==0) { p->f = &none::f; p->df = &none::df; p->name=none::name; return p;}
	if(act.compare(softmax::name) == 0) { p->f = &softmax::f; p->df = &softmax::df; p->name = softmax::name; return p; }
	delete p;
	return NULL;
}

activation_function* new_activation_function(const char *type)
{
	std::string act(type);
	return new_activation_function(act);
}

} // namespace