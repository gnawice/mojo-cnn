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
//    solver.h: stochastic optimization approaches
// ==================================================================== mojo ==

#pragma once

#include <math.h>
#include <algorithm>
#include <string>
#include <vector>
#include <cstdlib>

#include "core_math.h"

// hack for VS2010 to handle c++11 for(:)
#if (_MSC_VER  == 1600)
	#ifndef __for__
	#define __for__ for each
	#define __in__ in
	#endif
#else
	#ifndef __for__
	#define __for__ for
	#define __in__ :
	#endif
#endif

namespace mojo {



class solver
{
public:
	// learning rates are 'tweaked' in inc_w function so that they can be similar for all solvers
	float learning_rate;
	solver(): learning_rate(0.01f) {}
	virtual ~solver(){}
	virtual void reset() {}
	// this increments the weight matrix w, which corresponds to connection index 'g'
	// bottom is the number of grads coming up from the lower layer
	// top is the current output node value of the upper layer
	virtual void increment_w(matrix *w,  int g, const matrix &dW, const float custom_factor=1.0f){}//, matrix *top){}
	virtual void push_back(int w, int h, int c){}	
};

#ifndef MOJO_NO_TRAINING


class sgd: public solver
{
public:
	static const char *name(){return "sgd";}

	virtual void increment_w(matrix *w,  int g, const matrix &dW, const float custom_factor = 1.0f)
	{
		const float w_decay=0.01f;//1;
		const float lr=custom_factor*learning_rate;
		for(int s=0; s<w->size(); s++)	
			w->x[s] -= lr*(dW.x[s] + w_decay*w->x[s]);
	}
};

class adagrad: public solver
{
	// persistent variables that mirror size of weight matrix
	std::vector<matrix *> G1;
public:
	static const char *name(){return "adagrad";}

	virtual ~adagrad(){__for__(auto g __in__ G1) delete g;}
	virtual void push_back(int w, int h, int c) {
		G1.push_back(new matrix(w, h, c));
	}// G1[G1.size() - 1]->fill(0);

	
	virtual void reset() { __for__(auto g __in__ G1) g->fill(0.f);}
	virtual void increment_w(matrix *w,  int g, const matrix &dW, const float custom_factor = 1.0f)
	{
		float *g1 = G1[g]->x;
		//float min, max;
		//G1[g]->min_max(&min, &max);
		//std::cout << "((" << min << "," << max << ")";
		const float eps = 1.e-8f;
		// if (G1[g]->size() != w->size()) throw;
		const float lr = custom_factor*learning_rate;
		for(int s=0; s<w->size(); s++) 
		{
			g1[s] += dW.x[s] * dW.x[s];
			//if (g1[s] < 1) throw;
			w->x[s] -= lr*dW.x[s]/(std::sqrt(g1[s]) + eps);
		}	
	};
};

class rmsprop: public solver
{
	// persistent variables that mirror size of weight matrix
	std::vector<matrix *> G1;
public:
	static const char *name(){return "rmsprop";}
	virtual ~rmsprop(){__for__(auto g __in__ G1) delete g;}

	virtual void push_back(int w, int h, int c){ G1.push_back(new matrix(w,h,c)); G1[G1.size() - 1]->fill(0);}
	virtual void reset() { __for__(auto g __in__ G1) g->fill(0.f);}
	virtual void increment_w(matrix *w,  int g, const matrix &dW, const float custom_factor = 1.0f)
	{
		float *g1 = G1[g]->x;
		const float eps = 1.e-8f;
		const float mu = 0.999f;
		const float lr = 0.01f*custom_factor*learning_rate;

		for(int s=0; s<(int)w->size(); s++)
		{
			g1[s] = mu * g1[s]+(1-mu) * dW.x[s] * dW.x[s];
			w->x[s] -= lr*dW.x[s]/(std::sqrt(g1[s]) + eps);
		}	
	};

};

class adam: public solver
{
	float b1_t, b2_t;
	const float b1, b2;
	// persistent variables that mirror size of weight matrix
	std::vector<matrix *> G1;
	std::vector<matrix *> G2;
public:
	static const char *name(){return "adam";}
	adam(): b1(0.9f), b1_t(0.9f), b2(0.999f), b2_t(0.999f), solver()	{}
	virtual ~adam(){__for__(auto g __in__ G1) delete g; __for__(auto g __in__ G2) delete g;}

	virtual void reset()
	{
		b1_t*=b1; b2_t*=b2;
		__for__(auto g __in__ G1) g->fill(0.f);
		__for__(auto g __in__ G2) g->fill(0.f);
	}

	virtual void push_back(int w, int h, int c)
	{
		G1.push_back(new matrix(w,h,c)); G1[G1.size() - 1]->fill(0); 
		G2.push_back(new matrix(w,h,c)); G2[G2.size() - 1]->fill(0);
	}

	virtual void increment_w(matrix *w,  int g, const matrix &dW, const float custom_factor = 1.0f)
	{
		float *g1 = G1[g]->x;
		float *g2 = G2[g]->x;
		const float eps = 1.e-8f;
		const float b1=0.9f, b2=0.999f;
		const float lr = 0.1f*custom_factor*learning_rate;
		for(int s=0; s<(int)w->size(); s++)
			{
				g1[s] = b1* g1[s]+(1-b1) * dW.x[s];
				g2[s] = b2* g2[s]+(1-b2) * dW.x[s]*dW.x[s];
				w->x[s] -= lr* (g1[s]/(1.f-b1_t)) / ((float)std::sqrt(g2[s]/(1.-b2_t)) + eps);
			}	
	};

};


solver* new_solver(const char *type)
{
	if(type==NULL) return NULL;
	std::string act(type);
	if(act.compare(sgd::name())==0) { return new sgd();}
	if(act.compare(rmsprop::name())==0) { return new rmsprop();}
	if(act.compare(adagrad::name())==0) { return new adagrad();}
	if(act.compare(adam::name())==0) { return new adam();}

	return NULL;
}

#else


solver* new_solver(const char *type) {return NULL;}
solver* new_solver(std::string act){return NULL;}

#endif


} // namespace