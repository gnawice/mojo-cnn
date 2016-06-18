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
//    core_math.h: defines matrix class and math functions
// ==================================================================== mojo ==


#pragma once

#include <math.h>
#include <string.h>
#include <string>
#include <cstdlib>
#include <random>
#include <algorithm> 

namespace mojo
{

enum pad_type { zero = 0, edge = 1, median_edge = 2 };

inline float dot(const float *x1, const float *x2, const int size)
{
	switch (size)
	{
	case 1: return x1[0] * x2[0];
	case 2: return x1[0] * x2[0] + x1[1] * x2[1];
	case 3: return x1[0] * x2[0] + x1[1] * x2[1] + x1[2] * x2[2];
	case 4: return x1[0] * x2[0] + x1[1] * x2[1] + x1[2] * x2[2] + x1[3] * x2[3];
	case 5: return x1[0] * x2[0] + x1[1] * x2[1] + x1[2] * x2[2] + x1[3] * x2[3] + x1[4] * x2[4];
	default:
		float v = 0;
		for (int i = 0; i<size; i++) v += x1[i] * x2[i];
		return v;
	};
}
// in_size is 'line length'
inline bool unwrap_aligned_5x5(float *aligned_out, const float *in, const int in_size, const int stride=1)
{
	const int node_size = in_size - 5 + 1;
	int c1 = 0;
	float mpty = 0;

//	for (int k = 0; k < node_size*node_size; k++) mpty += in[k]* in[k];
//	if (mpty < 1e-6) return false;

	for (int j = 0; j < node_size; j += stride) // intput w
	{
		for (int i = 0; i < node_size; i += stride) // intput w
		{
			const float *tn = in + j*in_size + i;
			int c2 = 0;
			memcpy(&aligned_out[c1], &tn[c2], 5 * sizeof(float)); c1 += 5; c2 += in_size;
			memcpy(&aligned_out[c1], &tn[c2], 5 * sizeof(float)); c1 += 5; c2 += in_size;
			memcpy(&aligned_out[c1], &tn[c2], 5 * sizeof(float)); c1 += 5; c2 += in_size;
			memcpy(&aligned_out[c1], &tn[c2], 5 * sizeof(float)); c1 += 5; c2 += in_size;
			memcpy(&aligned_out[c1], &tn[c2], 5 * sizeof(float)); c1 += 5; c2 += in_size;
			c1 += 3;
		}
	}
	return true;
}

inline void unwrap_aligned_3x3(float *aligned_out, const float *in, const int in_size, const int stride=1)
{
	const int node_size = (in_size - 3) + 1;
	int c1 = 0;
	for (int j = 0; j < node_size; j += stride) // intput w
	{
		for (int i = 0; i < node_size; i += stride) // intput w
		{
			const float *tn = in + j*in_size + i;
			int c2 = 0;
			memcpy(&aligned_out[c1], &tn[c2], 3 * sizeof(float)); c1 += 3; c2 += in_size;
			memcpy(&aligned_out[c1], &tn[c2], 3 * sizeof(float)); c1 += 3; c2 += in_size;
			memcpy(&aligned_out[c1], &tn[c2], 3 * sizeof(float)); c1 += 3; c2 += in_size;
			c1 += 3;
		}
	}
}

inline void unwrap_aligned_4x4(float *aligned_out, const float *in, const int in_size, const int stride = 1)
{
	const int node_size = (in_size - 4) + 1;
	int c1 = 0;
	for (int j = 0; j < node_size; j += stride) // intput w
	{
		for (int i = 0; i < node_size; i += stride) // intput w
		{
			const float *tn = in + j*in_size + i;
			int c2 = 0;
			memcpy(&aligned_out[c1], &tn[c2], 2 * sizeof(float)); c1 += 4; c2 += in_size;
			memcpy(&aligned_out[c1], &tn[c2], 2 * sizeof(float)); c1 += 4; c2 += in_size;
			memcpy(&aligned_out[c1], &tn[c2], 2 * sizeof(float)); c1 += 4; c2 += in_size;
			memcpy(&aligned_out[c1], &tn[c2], 2 * sizeof(float)); c1 += 4; c2 += in_size;
		}
	}
}
inline void unwrap_aligned_2x2(float *aligned_out, const float *in, const int in_size, const int stride = 1)
{
	const int node_size = (in_size - 2) + 1;
	int c1 = 0;
	for (int j = 0; j < node_size; j += stride) // intput w
	{
		for (int i = 0; i < node_size; i += stride) // intput w
		{
			const float *tn = in + j*in_size + i;
			int c2 = 0;
			memcpy(&aligned_out[c1], &tn[c2], 2 * sizeof(float)); c1 +=2; c2 += in_size;
			memcpy(&aligned_out[c1], &tn[c2], 2 * sizeof(float)); c1 +=2; c2 += in_size;
		}
	}
}

#ifndef MOJO_SSE3
inline void dot_unwrapped_5x5(const float *_img, const float *filter_ptr, float *out, const int outsize)
{
	const float *_filt = filter_ptr;
	for (int j = 0; j < outsize; j += 1)//stride) // intput w
	{
		float c0 = _img[0] * _filt[0] + _img[1] * _filt[1] + _img[2] * _filt[2] + _img[3] * _filt[3];
		_img += 4; _filt += 4;
		c0 += _img[0] * _filt[0] + _img[1] * _filt[1] + _img[2] * _filt[2] + _img[3] * _filt[3];
		_img += 4; _filt += 4;
		c0 += _img[0] * _filt[0] + _img[1] * _filt[1] + _img[2] * _filt[2] + _img[3] * _filt[3];
		_img += 4; _filt += 4;
		c0 += _img[0] * _filt[0] + _img[1] * _filt[1] + _img[2] * _filt[2] + _img[3] * _filt[3];
		_img += 4; _filt += 4;
		c0 += _img[0] * _filt[0] + _img[1] * _filt[1] + _img[2] * _filt[2] + _img[3] * _filt[3];
		_img += 4; _filt += 4;
		c0 += _img[0] * _filt[0] + _img[1] * _filt[1] + _img[2] * _filt[2] + _img[3] * _filt[3];
		_img += 4; _filt += 4;
		c0 += _img[0] * _filt[0];// +_img[1] * _filt[1] + _img[2] * _filt[2] + _img[3] * _filt[3];
		_img += 4; _filt = filter_ptr;
		out[j] = c0;
	}
}


inline void dot_unwrapped_3x3(const float *_img, const float *filter_ptr, float *out, const int outsize)
{
	const float *_filt = filter_ptr;


	for (int j = 0; j < outsize; j += 1)//stride) // intput w
	{
		float c0 = _img[0] * _filt[0] + _img[1] * _filt[1] + _img[2] * _filt[2] + _img[3] * _filt[3];
		_img += 4; _filt += 4;
		c0 += _img[0] * _filt[0] + _img[1] * _filt[1] + _img[2] * _filt[2] + _img[3] * _filt[3];
		_img += 4; _filt += 4;
		c0 += _img[0] * _filt[0];
		_img += 4; _filt = filter_ptr;
		out[j] = c0;

	}
}

inline void dot_unwrapped_2x2(const float *_img, const float *filter_ptr, float *out, const int outsize)
{
	const float *_filt = filter_ptr;


	for (int j = 0; j < outsize; j += 1)//stride) // intput w
	{
		float c0 = _img[0] * _filt[0] + _img[1] * _filt[1] + _img[2] * _filt[2] + _img[3] * _filt[3];
		_img += 4; 
		out[j] = c0;

	}
}

#endif 

inline void unwrap_aligned(float *aligned_out, const float *in, const int in_size, const int kernel_size)
{
	const int node_size = in_size - kernel_size + 1;
	int c1 = 0;
	int mul = (int)((kernel_size*kernel_size +3) / 4);
	int leftover = mul * 4- kernel_size*kernel_size;
	for (int j = 0; j < node_size; j += 1)//stride) // intput w
	{
		for (int i = 0; i < node_size; i += 1)//stride) // intput w
		{
			const float *tn = in + j*in_size + i;
			int c2 = 0;
			for (int ii = 0; ii < kernel_size; ii += 1)//stride) // intput w
			{
				memcpy(&aligned_out[c1], &tn[c2], kernel_size * sizeof(float)); c1 += kernel_size; c2 += in_size;
			}
			c1 += leftover;
		}
	}
}
#ifdef MOJO_SSE3
inline void dot_unwrapped_5x5(const float *_img, const float *filter_ptr, float *out, const int outsize)
{
	_mm_prefetch((const char *)(out), _MM_HINT_T0);
	for (int j = 0; j < outsize; j += 1)//stride) // intput w
	{
		__m128 a, b, c0, c1;
		//_mm_prefetch((const char *)(filter_ptr + 12), _MM_HINT_T0);
		a = _mm_load_ps(_img); b = _mm_load_ps(filter_ptr);
		c0 = _mm_mul_ps(a, b);

		a = _mm_load_ps(_img + 4); b = _mm_load_ps(filter_ptr + 4);
		c1 = _mm_mul_ps(a, b);
		c0 = _mm_add_ps(c0, c1);

		a = _mm_load_ps(_img + 8); b = _mm_load_ps(filter_ptr + 8);
		c1 = _mm_mul_ps(a, b);
		c0 = _mm_add_ps(c0, c1);

		a = _mm_load_ps(_img + 12); b = _mm_load_ps(filter_ptr + 12);
		c1 = _mm_mul_ps(a, b);
		c0 = _mm_add_ps(c0, c1);

		a = _mm_load_ps(_img + 16); b = _mm_load_ps(filter_ptr + 16);
		c1 = _mm_mul_ps(a, b);
		c0 = _mm_add_ps(c0, c1);

		a = _mm_load_ps(_img + 20); b = _mm_load_ps(filter_ptr + 20);
		c1 = _mm_mul_ps(a, b);
		c0 = _mm_add_ps(c0, c1);


		//	a = _mm_load_ps(_img + 24); b = _mm_load_ps(filter_ptr + 24);
		//	c1 = _mm_mul_ps(a, b);
		//	a = _mm_set_ps(0, 0, 0, 1);
		//	c1 = _mm_mul_ps(a, c1);
		//	c0 = _mm_add_ps(c0, c1);

		c0 = _mm_hadd_ps(c0, c0);
		c0 = _mm_hadd_ps(c0, c0);

		_mm_store_ss(&out[j], c0);
		out[j] += _img[24] * filter_ptr[24];
		_img += 28;
	}
}

inline void dot_unwrapped_3x3(const float *_img, const float *filter_ptr, float *out, const int outsize)
{
	_mm_prefetch((const char *)(out), _MM_HINT_T0);
	for (int j = 0; j < outsize; j += 1)//stride) // intput w
	{
		__m128 a, b, c0, c1;
		_mm_prefetch((const char *)(out + j), _MM_HINT_T0);
		a = _mm_load_ps(_img); b = _mm_load_ps(filter_ptr);
		c0 = _mm_mul_ps(a, b);

		a = _mm_load_ps(_img + 4); b = _mm_load_ps(filter_ptr + 4);
		c1 = _mm_mul_ps(a, b);
		c0 = _mm_add_ps(c0, c1);

//		a = _mm_load_ps(_img + 8); b = _mm_load_ps(filter_ptr + 8);
//		c1 = _mm_mul_ps(a, b);
//		c0 = _mm_add_ps(c0, c1);

		c0 = _mm_hadd_ps(c0, c0);
		c0 = _mm_hadd_ps(c0, c0);

		_mm_store_ss(&out[j], c0);

		out[j] += _img[8] * filter_ptr[8];

		_img += 12;
	}
}
inline void dot_unwrapped_2x2(const float *_img, const float *filter_ptr, float *out, const int outsize)
{
	_mm_prefetch((const char *)(out), _MM_HINT_T0);
	for (int j = 0; j < outsize; j += 1)//stride) // intput w
	{
		__m128 a, b, c0;
		_mm_prefetch((const char *)(out + j), _MM_HINT_T0);
		a = _mm_load_ps(_img); b = _mm_load_ps(filter_ptr);
		c0 = _mm_mul_ps(a, b);

		c0 = _mm_hadd_ps(c0, c0);
		c0 = _mm_hadd_ps(c0, c0);

		_mm_store_ss(&out[j], c0);

		_img += 4;
	}
}
// not finished
inline void dot_unwrapped_sse(const float *_img, const float *filter_ptr, float *out, const int outsize, const int filtersize)
{
	__m128 a, b, c0, c1;
	for (int j = 0; j < outsize; j += 1)//stride) // intput w
	{
		_mm_prefetch((const char *)(out + j), _MM_HINT_T0);
		a = _mm_load_ps(_img); b = _mm_load_ps(filter_ptr);
		c0 = _mm_mul_ps(a, b);
		int i;
		for (i = 4; i < filtersize*filtersize; i += 4)
		{
			a = _mm_load_ps(_img + i); b = _mm_load_ps(filter_ptr + i);
			c1 = _mm_mul_ps(a, b);
			c0 = _mm_add_ps(c0, c1);
		}
		const int aligned_size = i;
		i -= 4;
		int leftover = filtersize*filtersize-i ;

		c0 = _mm_hadd_ps(c0, c0);
		c0 = _mm_hadd_ps(c0, c0);
		_mm_store_ss(&out[j], c0);
		
		//i = aligned_size - 1;
		while (leftover > 0)
		{
			out[j] += _img[i + leftover] * filter_ptr[i + leftover];
			leftover--;
		}
		_img += aligned_size;
	}
}
#endif

// second item is rotated 180 (this is a convolution)
inline float dot_rot180(const float *x1, const float *x2, const int size)	
{	
	switch(size)
	{
	case 1:  return x1[0]*x2[0]; 
	case 2:  return x1[0]*x2[1]+x1[1]*x2[0]; 
	case 3:  return x1[0]*x2[2]+x1[1]*x2[1]+x1[2]*x2[0]; 
	case 4:  return x1[0]*x2[3]+x1[1]*x2[2]+x1[2]*x2[1]+x1[3]*x2[0]; 
	case 5:  return x1[0]*x2[4]+x1[1]*x2[3]+x1[2]*x2[2]+x1[3]*x2[1]+x1[4]*x2[0]; 
	default: 
		float v=0;
		for(int i=0; i<size; i++) v+=x1[i]*x2[size-i-1];
		return v;	
	};

}

inline float unwrap_2d_dot(const float *x1, const float *x2, const int size, int stride1, int stride2)	
{	
	float v=0;	

	for(int j=0; j<size; j++) 
		v+= dot(&x1[stride1*j],&x2[stride2*j],size);
	return v;
}

//#include <smmintrin.h>

inline float unwrap_2d_dot_5x5(const float *x1, const float *x2,  int stride1, int stride2)	
{
	const float *f1 = &x1[0]; const float *f2 = &x2[0];
	float v;
	v = f1[0]*f2[0]+f1[1]*f2[1]+f1[2]*f2[2]+f1[3]*f2[3]+f1[4]*f2[4]; f1+=stride1; f2+=stride2;
	v+= f1[0]*f2[0]+f1[1]*f2[1]+f1[2]*f2[2]+f1[3]*f2[3]+f1[4]*f2[4]; f1+=stride1; f2+=stride2;
	v+= f1[0]*f2[0]+f1[1]*f2[1]+f1[2]*f2[2]+f1[3]*f2[3]+f1[4]*f2[4]; f1+=stride1; f2+=stride2;
	v+= f1[0]*f2[0]+f1[1]*f2[1]+f1[2]*f2[2]+f1[3]*f2[3]+f1[4]*f2[4]; f1+=stride1; f2+=stride2;
	v+= f1[0]*f2[0]+f1[1]*f2[1]+f1[2]*f2[2]+f1[3]*f2[3]+f1[4]*f2[4]; 
	return v;
}

inline float unwrap_2d_dot_3x3(const float *x1, const float *x2,  int stride1, int stride2)	
{	
	const float *f1=&x1[0]; const float *f2=&x2[0];
	float v;
	v = f1[0]*f2[0]+f1[1]*f2[1]+f1[2]*f2[2]; f1+=stride1; f2+=stride2;
	v+= f1[0]*f2[0]+f1[1]*f2[1]+f1[2]*f2[2]; f1+=stride1; f2+=stride2;
	v+= f1[0]*f2[0]+f1[1]*f2[1]+f1[2]*f2[2]; 
	return v;
}

// second item is rotated 180
inline float unwrap_2d_dot_rot180_5x5(const float *x1, const float *x2,  int stride1, int stride2)	
{	
	const float *f1=&x1[0]; const float *f2=&x2[stride2*4];
	float v = f1[0]*f2[4]+f1[1]*f2[3]+f1[2]*f2[2]+f1[3]*f2[1]+f1[4]*f2[0]; f1+=stride1; f2-=stride2;
	v+= f1[0]*f2[4]+f1[1]*f2[3]+f1[2]*f2[2]+f1[3]*f2[1]+f1[4]*f2[0]; f1+=stride1; f2-=stride2;
	v+= f1[0]*f2[4]+f1[1]*f2[3]+f1[2]*f2[2]+f1[3]*f2[1]+f1[4]*f2[0]; f1+=stride1; f2-=stride2;
	v+= f1[0]*f2[4]+f1[1]*f2[3]+f1[2]*f2[2]+f1[3]*f2[1]+f1[4]*f2[0]; f1+=stride1; f2-=stride2;
	v+= f1[0]*f2[4]+f1[1]*f2[3]+f1[2]*f2[2]+f1[3]*f2[1]+f1[4]*f2[0]; 
	return v;
}

// second item is rotated 180
inline float unwrap_2d_dot_rot180_3x3(const float *x1, const float *x2,  int stride1, int stride2)	
{	
	const float *f1=&x1[0]; const float *f2=&x2[stride2*2];
	float v = f1[0]*f2[2]+f1[1]*f2[1]+f1[2]*f2[0]; f1+=stride1; f2-=stride2;
	v+= f1[0]*f2[2]+f1[1]*f2[1]+f1[2]*f2[0]; f1+=stride1; f2-=stride2;
	v+= f1[0]*f2[2]+f1[1]*f2[1]+f1[2]*f2[0]; 
	return v;
}

// second item is rotated 180
inline float unwrap_2d_dot_rot180(const float *x1, const float *x2, const int size, int stride1, int stride2)	
{	
	float v=0;	
	for(int j=0; j<size; j++) 
	{
		v+= dot_rot180(&x1[stride1*j],&x2[stride2*(size-j-1)],size);
	}
	return v;
}

// matrix class ---------------------------------------------------
// should use opencv if available
//

class matrix
{
	int _size;
	int _capacity;
	float *_x_mem;
	void delete_x() { delete[] _x_mem; x = NULL;  _x_mem = NULL; }
	// 4 extra for alignment and 4 for 3 padding for SSE
	float *new_x(const int size) { _x_mem = new float[size + 4+3];  x = (float *)(((uintptr_t)_x_mem + 16) & ~(uintptr_t)0x0F); return x; }
public:
	std::string _name;
	int cols, rows, chans;
	float *x;
	//unsigned char *empty_chan;

	matrix( ): cols(0), rows(0), chans(0), _size(0), _capacity(0), x(NULL)/*, empty_chan(NULL)*/{}

	matrix( int _w, int _h, int _c=1, const float *data=NULL): cols(_w), rows(_h), chans(_c) 
	{
		_size=cols*rows*chans; _capacity=_size; x = new_x(_size); 
		if(data!=NULL) memcpy(x,data,_size*sizeof(float));
		
//		empty_chan = new unsigned char[chans];
//		memset(empty_chan, 0, chans);
	}

	//inline void reset_empty_chans(){ memset(empty_chan, 0, chans); }

	// copy constructor - deep copy
	matrix( const matrix &m) : cols(m.cols), rows(m.rows), chans(m.chans), _size(m._size), _capacity(m._size)   {x = new_x(_size); memcpy(x,m.x,sizeof(float)*_size); /*empty_chan = new unsigned char[chans]; memcpy(empty_chan, m.empty_chan, chans);*/} // { v=m.v; x=(float*)v.data();}
	// copy and pad constructor
	matrix( const matrix &m, int pad_cols, int pad_rows, mojo::pad_type padding= mojo::pad_type::zero) : cols(m.cols), rows(m.rows), chans(m.chans), _size(m._size), _capacity(m._size)  
	{
		x = new_x(_size); memcpy(x, m.x, sizeof(float)*_size);
		*this = pad(pad_cols, pad_rows, padding);
/*
		_size = cols*rows*chans;
		_capacity = _size;
		x = new_x(_size); 
	//	*this = m;

		*
		
//		empty_chan = new unsigned char[chans];
//		memcpy(empty_chan, m.empty_chan, chans);
		fill(0);
		for(int c=0; c<m.chans; c++)
		for(int j=0; j<m.rows; j++)
		{
			memcpy(x+pad_cols+(pad_rows+j)*cols+c*cols*rows,m.x+j*m.cols +c*m.cols*m.rows,sizeof(float)*m.cols);
		}
	*/	 
	} // { v=m.v; x=(float*)v.data();}

	~matrix() { if (x) delete_x(); /*if (empty_chan) delete[] empty_chan; */}
	
	matrix get_chans(int start_channel, int num_chans=1) const
	{
		return matrix(cols,rows,num_chans,&x[start_channel*cols*rows]);
	}


	// if edge_pad==0, then the padded area is just 0. 
	// if edge_pad==1 it fills with edge pixel colors
	// if edge_pad==2 it fills with median edge pixel color
	matrix pad(int dx, int dy, mojo::pad_type edge_pad = mojo::pad_type::zero) const
	{
		return pad(dx, dy, dx, dy, edge_pad);
	}
	matrix pad(int dx, int dy, int dx_right, int dy_bottom, mojo::pad_type edge_pad = mojo::pad_type::zero) const
	{
		matrix v(cols+dx+dx_right,rows+dy+dy_bottom,chans);
		v.fill(0);
	
		//float *new_x = new float[chans*w*h]; 
		for(int k=0; k<chans; k++)
		{
			const int v_chan_offset=k*v.rows*v.cols;
			const int chan_offset=k*cols*rows;
			// find median color of perimeter
			float median = 0.f;
			if (edge_pad == mojo::pad_type::median_edge)
			{
				int perimeter = 2 * (cols + rows - 2);
				std::vector<float> d(perimeter);
				for (int i = 0; i < cols; i++)
				{
					d[i] = x[i+ chan_offset]; d[i + cols] = x[i + cols*(rows - 1)+ chan_offset];
				}
				for (int i = 1; i < (rows - 1); i++)
				{
					d[i + cols * 2] = x[cols*i+ chan_offset];
					// file from back so i dont need to cal index
					d[perimeter - i] = x[cols - 1 + cols*i+ chan_offset];
				}

				std::nth_element(d.begin(), d.begin() + perimeter / 2, d.end());
				median = d[perimeter / 2];
				//for (int i = 0; i < v.rows*v.cols; i++) v.x[v_chan_offset + i] = solid_fill;
			}

			for(int j=0; j<rows; j++)
			{
				memcpy(&v.x[dx+(j+dy)*v.cols+v_chan_offset], &x[j*cols+chan_offset], sizeof(float)*cols);
				if(edge_pad== mojo::pad_type::edge)
				{
					// do left/right side
					for(int i=0; i<dx; i++) v.x[i+(j+dy)*v.cols+v_chan_offset]=x[0+j*cols+chan_offset];
					for (int i = 0; i<dx_right; i++) v.x[i + dx + cols + (j + dy)*v.cols + v_chan_offset] = x[(cols - 1) + j*cols + chan_offset];
				}
				else if (edge_pad == mojo::pad_type::median_edge)
				{
					for (int i = 0; i < dx; i++) v.x[i + (j + dy)*v.cols + v_chan_offset] = median;
					for (int i = 0; i < dx_right; i++) v.x[i + dx + cols + (j + dy)*v.cols + v_chan_offset] = median;
				}
			}
			// top bottom pad
			if(edge_pad== mojo::pad_type::edge)
			{
				for(int j=0; j<dy; j++)	memcpy(&v.x[(j)*v.cols+v_chan_offset],&v.x[(dy)*v.cols+v_chan_offset], sizeof(float)*v.cols);
				for (int j = 0; j<dy_bottom; j++) memcpy(&v.x[(j + dy + rows)*v.cols + v_chan_offset], &v.x[(rows - 1 + dy)*v.cols + v_chan_offset], sizeof(float)*v.cols);
			}
			if (edge_pad == mojo::pad_type::median_edge)
			{
				for (int j = 0; j<dy; j++)	
					for (int i = 0; i<v.cols; i++) 
						v.x[i + j*v.cols + v_chan_offset] = median;
				for (int j = 0; j<dy_bottom; j++) 
					for (int i = 0; i<v.cols; i++) 
						v.x[i + (j + dy + rows)*v.cols + v_chan_offset] = median;
			}
		}

		return v;
	}

	matrix crop(int dx, int dy, int w, int h) const
	{
		matrix v(w,h,chans);

		//float *new_x = new float[chans*w*h]; 
		for(int k=0; k<chans; k++)
		for(int j=0; j<h; j++)
		{
			memcpy(&v.x[j*w+k*w*h], &x[dx+(j+dy)*cols+k*rows*cols], sizeof(float)*w);
		}

		return v;
	}

	mojo::matrix shift(int dx, int dy, mojo::pad_type edge_pad=mojo::pad_type::zero)
	{
		int orig_cols=cols;
		int orig_rows=rows;
		int off_x=abs(dx);
		int off_y=abs(dy);

		mojo::matrix shifted= pad(off_x, off_y, edge_pad);

		return shifted.crop(off_x-dx, off_y-dy,orig_cols,orig_rows);
	}

	mojo::matrix flip_cols()
	{
		mojo::matrix v(cols,rows,chans);
		for(int k=0; k<chans; k++)
			for(int j=0; j<rows; j++)
				for(int i=0; i<cols; i++)
					v.x[i+j*cols+k*cols*rows]=x[(cols-i-1)+j*cols+k*cols*rows];

		return v;
	}
	mojo::matrix flip_rows()
	{
		mojo::matrix v(cols, rows, chans);
		
		for (int k = 0; k<chans; k++)
			for (int j = 0; j<rows; j++)
				memcpy(&v.x[(rows-1-j)*cols + k*cols*rows],&x[j*cols + k*cols*rows], cols*sizeof(float));

		return v;
	}

	void clip(float min, float max)
	{
		int s = rows*cols*chans;
		for (int i = 0; i < s; i++)
		{
			if (x[i] < min) x[i] = min;
			if (x[i] > max) x[i]=max;
		}
	}


	void min_max(float *min, float *max, int *min_i=NULL, int *max_i=NULL)
	{
		int s = rows*cols*chans;
		int mini = 0;
		int maxi = 0; 
		for (int i = 0; i < s; i++)
		{
			if (x[i] < x[mini]) mini = i;
			if (x[i] > x[maxi]) maxi = i;
		}
		*min = x[mini];
		*max = x[maxi];
		if (min_i) *min_i = mini;
		if (max_i) *max_i = maxi;
	}

	float remove_mean(int channel)
	{
		int s = rows*cols;
		int offset = channel*s;
		float average=0;
		for(int i=0; i<s; i++) average+=x[i+offset];		
		average= average/(float)s;
		for(int i=0; i<s; i++) x[i+offset]-=average;		
		return average;
	}

	float remove_mean()
	{
		int s = rows*cols*chans;
		//int offset = channel*s;
		float average=0;
		for(int i=0; i<s; i++) average+=x[i];		
		average= average/(float)s;
		for(int i=0; i<s; i++) x[i]-=average;		
		return average;
	}
	void fill(float val) { for(int i=0; i<_size; i++) x[i]=val; }
	void fill_random_uniform(float range)
	{
		std::mt19937 gen(0);
		std::uniform_real_distribution<float> dst(-range, range);
		for (int i = 0; i<_size; i++) x[i] = dst(gen);
	}
	void fill_random_normal(float std)
	{
		std::mt19937 gen(0);
		std::normal_distribution<float> dst(0, std);
		for (int i = 0; i<_size; i++) x[i] = dst(gen);
	}


	// deep copy
	inline matrix& operator =(const matrix &m)
	{
		resize(m.cols, m.rows, m.chans);
		memcpy(x,m.x,sizeof(float)*_size);
//		memcpy(empty_chan, m.empty_chan, chans);
		return *this;
	}

	int  size() const {return _size;} 
	
	void resize(int _w, int _h, int _c) { 
		int s = _w*_h*_c;
		if(s>_capacity) 
		{ 
			if(_capacity>0) delete_x(); _size = s; _capacity=_size; x = new_x(_size);
		}
/*		if (_c > chans)
		{
			if (empty_chan) delete[] empty_chan;
			empty_chan = new unsigned char[_c];
		}
	*/	cols=_w; rows=_h; chans=_c; _size=s;
	} 
	
	// dot vector to 2d mat
	inline matrix dot_1dx2d(const matrix &m_2d) const
	{
		mojo::matrix v(m_2d.rows, 1, 1);
		for(int j=0; j<m_2d.rows; j++)	v.x[j]=dot(x,&m_2d.x[j*m_2d.cols],_size);
		return v;
	}

	// +=
	inline matrix& operator+=(const matrix &m2){
	  for(int i = 0; i < _size; i++) x[i] += m2.x[i];
	  return *this;
	}
	// -=
	inline matrix& operator-=(const matrix &m2) {
		for (int i = 0; i < _size; i++) x[i] -= m2.x[i];
		return *this;
	}
#ifndef MOJO_SSE3
	// *= float
	inline matrix operator *=(const float v) {
		for (int i = 0; i < _size; i++) x[i] = x[i] * v;
		return *this;
	}


#else
	inline matrix operator *=(const float v) {
		__m128  b;
		b = _mm_set_ps(v, v, v, v);
		for (int j = 0; j < _size; j += 4)
			_mm_store_ps(x + j, _mm_mul_ps(_mm_load_ps(x + j), b));
		return *this;
	}
#endif
	// *= matrix
	inline matrix operator *=(const matrix &v) {
		for (int i = 0; i < _size; i++) x[i] = x[i] * v.x[i];
		return *this;
	}
	inline matrix operator *(const matrix &v) {
		matrix T(cols, rows, chans);
		for (int i = 0; i < _size; i++) T.x[i] = x[i] * v.x[i];
		return T;
	}
	// * float
	inline matrix operator *(const float v) {
		matrix T(cols, rows, chans);
		for (int i = 0; i < _size; i++) T.x[i] = x[i] * v;
		return T;
	}

	// + float
	inline matrix operator +(const float v) {
		matrix T(cols, rows, chans);
		for (int i = 0; i < _size; i++) T.x[i] = x[i] + v;
		return T;
	}

	// +
	inline matrix operator +(matrix m2)
	{
		matrix T(cols,rows,chans);
		for(int i = 0; i < _size; i++) T.x[i] = x[i] + m2.x[i]; 
		return T;
	}
};

}// namespace

