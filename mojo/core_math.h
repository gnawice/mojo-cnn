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

inline float unwrap_2d_dot(const float *x1, const float *x2, const int size, int stride1, int stride2)	
{	
	float v=0;	

	for(int j=0; j<size; j++) 
		v+= dot(&x1[stride1*j],&x2[stride2*j],size);
	return v;
}


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
inline float unwrap_2d_dot_rot180(const float *x1, const float *x2, const int size, int stride1, int stride2)	
{	
	float v=0;	
	for(int j=0; j<size; j++) 
	{
		v+= dot_rot180(&x1[stride1*j],&x2[stride2*(size-j-1)],size);
	}
	return v;
}


inline void unwrap_aligned_NxN(const int N, float *aligned_out,  const float *in, const int in_size, const int stride = 1)
{
	const int node_size = (in_size - N) + 1;
	int c1 = 0;
	int off = 0;
	const int inc_off = N*N*8;

	for (int j = 0; j < node_size; j += 1) // intput h
	{
		for (int i = 0; i < node_size; i += 1) // intput w
		{
			const float *tn = in + j*in_size + i;

			if(N==5)
			{
				for (int k = 0; k < 5; k++)
				{
					aligned_out[c1 + 0 +  k * 40 + off] = tn[0 + 0 + in_size*k];
					aligned_out[c1 + 8 +  k * 40 + off] = tn[0 + 1 + in_size*k];
					aligned_out[c1 + 16 + k * 40 + off] = tn[0 + 2 + in_size*k];
					aligned_out[c1 + 24 + k * 40 + off] = tn[0 + 3 + in_size*k];
					aligned_out[c1 + 32 + k * 40 + off] = tn[0 + 4 + in_size*k];
				}
			}
			else if(N==3)
			{
				aligned_out[c1 + off] = tn[0];
				aligned_out[c1 + 8 + off] = tn[0 + 1];
				aligned_out[c1 + 16 + off] = tn[0 + 2];

				aligned_out[c1 + 24 + off] = tn[0 +     in_size];
				aligned_out[c1 + 32 + off] = tn[0 + 1 + in_size];
				aligned_out[c1 + 40 + off] = tn[0 + 2 + in_size];

				aligned_out[c1 + 48 + off] = tn[0 +     2 * in_size];
				aligned_out[c1 + 56 + off] = tn[0 + 1 + 2 * in_size];
				aligned_out[c1 + 64 + off] = tn[0 + 2 + 2 * in_size];
			}
			else
			{
				int cnt=0;
				for (int k = 0; k < N; k++)
				{
					for (int m = 0; m < N; m++) 
					{
						aligned_out[c1 + cnt*8 + off] = tn[0 + m + in_size*k];
						cnt++;
					}
				}
			}


			off++;
			if (off > 7) { off = 0; c1 += inc_off; }
		}

	}
}


inline void dotsum_unwrapped_NxN(const int N, const float *im,  const float *filter_ptr, float *out, const int outsize)
{
	const int NN=N*N;
	for (int j = 0; j < outsize; j += 8)
	{
		float *c = out+j;
		for(int i=0; i<NN; i++) 
		{ 
			const float f = filter_ptr[i];
			c[0]+=im[0]*f; c[1]+=im[1]*f; c[2]+=im[2]*f; c[3]+=im[3]*f; c[4]+=im[4]*f; c[5]+=im[5]*f; c[6]+=im[6]*f; c[7]+=im[7]*f;
			im+=8; 
		}
	}
}

#ifdef MOJO_AVX

inline void dotsum_unwrapped_2x2(const float *_img, const float *filter_ptr, float *out, const int outsize)
{
	_mm256_zeroupper();

	const __m256 f0 = _mm256_broadcast_ss(&filter_ptr[0]); const __m256 f1 = _mm256_broadcast_ss(&filter_ptr[1]);
	const __m256 f2 = _mm256_broadcast_ss(&filter_ptr[2]); const __m256 f3 = _mm256_broadcast_ss(&filter_ptr[3]);

	for (int j = 0; j < outsize; j += 8)
	{
		__m256 a, c0, c1;
		// multiply filter
		a = _mm256_load_ps(_img); c0 = _mm256_mul_ps(a, f0);
		a = _mm256_load_ps(_img +  8); c1 = _mm256_mul_ps(a, f1); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 16); c1 = _mm256_mul_ps(a, f2); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 24); c1 = _mm256_mul_ps(a, f3); c0 = _mm256_add_ps(c0, c1);
		
		// add result to output
		a = _mm256_load_ps(out + j); 
		c0 = _mm256_add_ps(c0, a);
		_mm256_stream_ps(out + j, c0);
		_img += 32;
	}
	_mm256_zeroupper();
}

inline void dotsum_unwrapped_3x3(const float *_img, const float *filter_ptr, float *out, const int outsize)
{
	_mm256_zeroupper();

	const __m256 f0 = _mm256_broadcast_ss(&filter_ptr[0]); const __m256 f1 = _mm256_broadcast_ss(&filter_ptr[1]);
	const __m256 f2 = _mm256_broadcast_ss(&filter_ptr[2]); const __m256 f3 = _mm256_broadcast_ss(&filter_ptr[3]);
	const __m256 f4 = _mm256_broadcast_ss(&filter_ptr[4]); const __m256 f5 = _mm256_broadcast_ss(&filter_ptr[5]);
	const __m256 f6 = _mm256_broadcast_ss(&filter_ptr[6]); const __m256 f7 = _mm256_broadcast_ss(&filter_ptr[7]);
	const __m256 f8 = _mm256_broadcast_ss(&filter_ptr[8]);

	for (int j = 0; j < outsize; j += 8)//stride) // intput w
	{
		__m256 a, c0, c1;
		// multiply filter
		a = _mm256_load_ps(_img); c0 = _mm256_mul_ps(a, f0);
		a = _mm256_load_ps(_img +  8); c1 = _mm256_mul_ps(a, f1); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 16); c1 = _mm256_mul_ps(a, f2); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 24); c1 = _mm256_mul_ps(a, f3); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 32); c1 = _mm256_mul_ps(a, f4); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 40); c1 = _mm256_mul_ps(a, f5); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 48); c1 = _mm256_mul_ps(a, f6); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 56); c1 = _mm256_mul_ps(a, f7); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 64); c1 = _mm256_mul_ps(a, f8); c0 = _mm256_add_ps(c0, c1);
		// add result to output
		a = _mm256_load_ps(out + j); 
		c0 = _mm256_add_ps(c0, a);
		_mm256_stream_ps(out + j, c0);
		_img += 72;
	}
	_mm256_zeroupper();

}

inline void dotsum_unwrapped_4x4(const float *_img, const float *filter_ptr, float *out, const int outsize)
{
	_mm256_zeroupper();

	const __m256 f0 = _mm256_broadcast_ss(&filter_ptr[0]); const __m256 f1 = _mm256_broadcast_ss(&filter_ptr[1]);
	const __m256 f2 = _mm256_broadcast_ss(&filter_ptr[2]); const __m256 f3 = _mm256_broadcast_ss(&filter_ptr[3]);
	const __m256 f4 = _mm256_broadcast_ss(&filter_ptr[4]); const __m256 f5 = _mm256_broadcast_ss(&filter_ptr[5]);
	const __m256 f6 = _mm256_broadcast_ss(&filter_ptr[6]); const __m256 f7 = _mm256_broadcast_ss(&filter_ptr[7]);
	const __m256 f8 = _mm256_broadcast_ss(&filter_ptr[8]); const __m256 f9 = _mm256_broadcast_ss(&filter_ptr[9]);
	const __m256 f10 = _mm256_broadcast_ss(&filter_ptr[10]); const __m256 f11 = _mm256_broadcast_ss(&filter_ptr[11]);
	const __m256 f12 = _mm256_broadcast_ss(&filter_ptr[12]); const __m256 f13 = _mm256_broadcast_ss(&filter_ptr[13]);
	const __m256 f14 = _mm256_broadcast_ss(&filter_ptr[14]); const __m256 f15 = _mm256_broadcast_ss(&filter_ptr[15]);

	for (int j = 0; j < outsize; j += 8)//stride) // intput w
	{
		__m256 a, c0, c1;
		// multiply filter
		a = _mm256_load_ps(_img); c0 = _mm256_mul_ps(a, f0);
		a = _mm256_load_ps(_img + 8); c1 = _mm256_mul_ps(a, f1); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 16); c1 = _mm256_mul_ps(a, f2); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 24); c1 = _mm256_mul_ps(a, f3); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 32); c1 = _mm256_mul_ps(a, f4); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 40); c1 = _mm256_mul_ps(a, f5); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 48); c1 = _mm256_mul_ps(a, f6); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 56); c1 = _mm256_mul_ps(a, f7); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 64); c1 = _mm256_mul_ps(a, f8); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 72); c1 = _mm256_mul_ps(a, f9); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 80); c1 = _mm256_mul_ps(a, f10); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 88); c1 = _mm256_mul_ps(a, f11); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 96); c1 = _mm256_mul_ps(a, f12); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 104); c1 = _mm256_mul_ps(a, f13); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 112); c1 = _mm256_mul_ps(a, f14); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 120); c1 = _mm256_mul_ps(a, f15); c0 = _mm256_add_ps(c0, c1);
		// add result to output
		a = _mm256_load_ps(out + j); 
		c0 = _mm256_add_ps(c0, a);
		_mm256_stream_ps(out + j, c0);
		_img += 128;
	}
	_mm256_zeroupper();

}

inline void dotsum_unwrapped_5x5(const float *_img, const float *filter_ptr, float *out, const int outsize)
{
	_mm256_zeroupper();

	const __m256 f0 = _mm256_broadcast_ss(&filter_ptr[0]); const __m256 f1 = _mm256_broadcast_ss(&filter_ptr[1]);
	const __m256 f2 = _mm256_broadcast_ss(&filter_ptr[2]); const __m256 f3 = _mm256_broadcast_ss(&filter_ptr[3]);
	const __m256 f4 = _mm256_broadcast_ss(&filter_ptr[4]); const __m256 f5 = _mm256_broadcast_ss(&filter_ptr[5]);
	const __m256 f6 = _mm256_broadcast_ss(&filter_ptr[6]); const __m256 f7 = _mm256_broadcast_ss(&filter_ptr[7]);
	const __m256 f8 = _mm256_broadcast_ss(&filter_ptr[8]); const __m256 f9 = _mm256_broadcast_ss(&filter_ptr[9]);
	const __m256 f10 = _mm256_broadcast_ss(&filter_ptr[10]); const __m256 f11 = _mm256_broadcast_ss(&filter_ptr[11]);
	const __m256 f12 = _mm256_broadcast_ss(&filter_ptr[12]); const __m256 f13 = _mm256_broadcast_ss(&filter_ptr[13]);
	const __m256 f14 = _mm256_broadcast_ss(&filter_ptr[14]); const __m256 f15 = _mm256_broadcast_ss(&filter_ptr[15]);
	const __m256 f16 = _mm256_broadcast_ss(&filter_ptr[16]); const __m256 f17 = _mm256_broadcast_ss(&filter_ptr[17]);
	const __m256 f18 = _mm256_broadcast_ss(&filter_ptr[18]); const __m256 f19 = _mm256_broadcast_ss(&filter_ptr[19]);
	const __m256 f20 = _mm256_broadcast_ss(&filter_ptr[20]); const __m256 f21 = _mm256_broadcast_ss(&filter_ptr[21]);
	const __m256 f22 = _mm256_broadcast_ss(&filter_ptr[22]); const __m256 f23 = _mm256_broadcast_ss(&filter_ptr[23]);
	const __m256 f24 = _mm256_broadcast_ss(&filter_ptr[24]); 

	for (int j = 0; j < outsize; j += 8)
	{
		__m256 a, c0, c1;

		a = _mm256_load_ps(_img); c0 = _mm256_mul_ps(a, f0);

		a = _mm256_load_ps(_img + 8); c1 = _mm256_mul_ps(a, f1); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 16); c1 = _mm256_mul_ps(a, f2); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 24); c1 = _mm256_mul_ps(a, f3); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 32); c1 = _mm256_mul_ps(a, f4); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 40); c1 = _mm256_mul_ps(a, f5); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 48); c1 = _mm256_mul_ps(a, f6); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 56); c1 = _mm256_mul_ps(a, f7); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 64); c1 = _mm256_mul_ps(a, f8); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 72); c1 = _mm256_mul_ps(a, f9); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 80); c1 = _mm256_mul_ps(a, f10); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 88); c1 = _mm256_mul_ps(a, f11); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 96); c1 = _mm256_mul_ps(a, f12); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 104); c1 = _mm256_mul_ps(a, f13); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 112); c1 = _mm256_mul_ps(a, f14); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 120); c1 = _mm256_mul_ps(a, f15); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 128); c1 = _mm256_mul_ps(a, f16); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 136); c1 = _mm256_mul_ps(a, f17); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 144); c1 = _mm256_mul_ps(a, f18); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 152); c1 = _mm256_mul_ps(a, f19); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 160); c1 = _mm256_mul_ps(a, f20); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 168); c1 = _mm256_mul_ps(a, f21); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 176); c1 = _mm256_mul_ps(a, f22); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 184); c1 = _mm256_mul_ps(a, f23); c0 = _mm256_add_ps(c0, c1);
		a = _mm256_load_ps(_img + 192); c1 = _mm256_mul_ps(a, f24); c0 = _mm256_add_ps(c0, c1);

		a = _mm256_load_ps(out + j);
		c0 = _mm256_add_ps(c0, a);

		_mm256_stream_ps(out + j, c0);
		_img += 200;

	}

	_mm256_zeroupper();
}

inline void dotsum_unwrapped_7x7(const float *_img,  const float *filter_ptr, float *out, const int outsize)
{

	_mm256_zeroupper();
	__m256 f[49];//=new __m256(s);
	for(int i=0; i<49; i++) f[i]= _mm256_broadcast_ss(&filter_ptr[i]); 

	for (int j = 0; j < outsize; j += 8)
	{
		__m256 a, c0, c1;

		a = _mm256_load_ps(_img);
		c0 = _mm256_mul_ps(a, f[0]);
		for(int i=1; i<49;i++)
		{
			a = _mm256_load_ps(_img + 8*i); c1 = _mm256_mul_ps(a, f[i]); c0 = _mm256_add_ps(c0, c1);
		}

		a = _mm256_load_ps(out + j);
		c0 = _mm256_add_ps(c0, a);

		_mm256_stream_ps(out + j, c0);
		_img += 49*8;

	}
	_mm256_zeroupper();
	//delete [] f;
}

#else // no AVX

inline void dotsum_unwrapped_2x2(const float *_img, const float *filter_ptr, float *out, const int outsize)
{
	dotsum_unwrapped_NxN(2, _img,  filter_ptr, out, outsize);
}
inline void dotsum_unwrapped_3x3(const float *_img, const float *filter_ptr, float *out, const int outsize)
{
	dotsum_unwrapped_NxN(3, _img,  filter_ptr, out, outsize);
}
inline void dotsum_unwrapped_4x4(const float *_img, const float *filter_ptr, float *out, const int outsize)
{
	dotsum_unwrapped_NxN(4, _img,  filter_ptr, out, outsize);
}
inline void dotsum_unwrapped_5x5(const float *_img, const float *filter_ptr, float *out, const int outsize)
{
	dotsum_unwrapped_NxN(5, _img,  filter_ptr, out, outsize);
}
inline void dotsum_unwrapped_7x7(const float *_img, const float *filter_ptr, float *out, const int outsize)
{
	dotsum_unwrapped_NxN(7, _img,  filter_ptr, out, outsize);
}

#endif


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
	//float *new_x(const int size) { _x_mem = new float[size + 4+3];  x = (float *)(((uintptr_t)_x_mem + 16) & ~(uintptr_t)0x0F); return x; }
	// avx mem aligment
	float *new_x(const int size) { _x_mem = new float[size + 8 + 7];  x = (float *)(((uintptr_t)_x_mem + 32) & ~(uintptr_t)0x1F); return x; }
public:
	std::string _name;
	int cols, rows, chans;
	int chan_stride;
	int chan_aligned;
	float *x;
	// size must be divisible by 8 for AVX
	virtual int calc_chan_stride(int w, int h)
	{
		if (chan_aligned)
		{
			int s = w*h;
			const int remainder = s % 8;
			if (remainder > 0) s += 8 - remainder;
			return s;
		}
		else return w*h;
	}

	matrix( ): cols(0), rows(0), chans(0), _size(0), _capacity(0), chan_stride(0), x(NULL), chan_aligned(0)/*, empty_chan(NULL)*/{}


	matrix( int _w, int _h, int _c=1, const float *data=NULL, int align_chan=0): cols(_w), rows(_h), chans(_c)
	{
		chan_aligned = align_chan;
		chan_stride = calc_chan_stride(cols, rows);
		_size= chan_stride*chans; _capacity=_size; x = new_x(_size);
		if(data!=NULL) memcpy(x,data,_size*sizeof(float));
	}

	// copy constructor - deep copy
	matrix( const matrix &m) : cols(m.cols), rows(m.rows), chan_aligned(m.chan_aligned), chans(m.chans), chan_stride(m.chan_stride), _size(m._size), _capacity(m._size)   {x = new_x(_size); memcpy(x,m.x,sizeof(float)*_size); /*empty_chan = new unsigned char[chans]; memcpy(empty_chan, m.empty_chan, chans);*/} // { v=m.v; x=(float*)v.data();}
	// copy and pad constructor
	matrix( const matrix &m, int pad_cols, int pad_rows, mojo::pad_type padding= mojo::zero, int threads=1) : cols(m.cols), rows(m.rows), chans(m.chans), chan_aligned(m.chan_aligned), chan_stride(m.chan_stride), _size(m._size), _capacity(m._size)
	{
		x = new_x(_size); memcpy(x, m.x, sizeof(float)*_size);
		*this = pad(pad_cols, pad_rows, padding, threads);
	}

	~matrix() { if (x) delete_x(); }
	
	matrix get_chans(int start_channel, int num_chans=1) const
	{
		return matrix(cols,rows,num_chans,&x[start_channel*chan_stride]);
	}


	// if edge_pad==0, then the padded area is just 0. 
	// if edge_pad==1 it fills with edge pixel colors
	// if edge_pad==2 it fills with median edge pixel color
	matrix pad(int dx, int dy, mojo::pad_type edge_pad = mojo::zero, int threads=1) const
	{
		return pad(dx, dy, dx, dy, edge_pad, threads);
	}
	matrix pad(int dx, int dy, int dx_right, int dy_bottom, mojo::pad_type edge_pad = mojo::zero, int threads=1) const
	{
		matrix v(cols+dx+dx_right,rows+dy+dy_bottom,chans);//,NULL,this->chan_aligned);
		v.fill(0);
	
		//float *new_x = new float[chans*w*h]; 
#pragma omp parallel for num_threads(threads)
		for(int k=0; k<chans; k++)
		{
			const int v_chan_offset=k*v.chan_stride;
			const int chan_offset=k*chan_stride;
			// find median color of perimeter
			float median = 0.f;
			if (edge_pad == mojo::median_edge)
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
				if(edge_pad== mojo::edge)
				{
					// do left/right side
					for(int i=0; i<dx; i++) v.x[i+(j+dy)*v.cols+v_chan_offset]=x[0+j*cols+chan_offset];
					for (int i = 0; i<dx_right; i++) v.x[i + dx + cols + (j + dy)*v.cols + v_chan_offset] = x[(cols - 1) + j*cols + chan_offset];
				}
				else if (edge_pad == mojo::median_edge)
				{
					for (int i = 0; i < dx; i++) v.x[i + (j + dy)*v.cols + v_chan_offset] = median;
					for (int i = 0; i < dx_right; i++) v.x[i + dx + cols + (j + dy)*v.cols + v_chan_offset] = median;
				}
			}
			// top bottom pad
			if(edge_pad== mojo::edge)
			{
				for(int j=0; j<dy; j++)	memcpy(&v.x[(j)*v.cols+v_chan_offset],&v.x[(dy)*v.cols+v_chan_offset], sizeof(float)*v.cols);
				for (int j = 0; j<dy_bottom; j++) memcpy(&v.x[(j + dy + rows)*v.cols + v_chan_offset], &v.x[(rows - 1 + dy)*v.cols + v_chan_offset], sizeof(float)*v.cols);
			}
			if (edge_pad == mojo::median_edge)
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

	matrix crop(int dx, int dy, int w, int h, int threads=1) const
	{
		matrix v(w,h,chans);

#pragma omp parallel for num_threads(threads)
		for(int k=0; k<chans; k++)
		{
			for(int j=0; j<h; j++)
			{
				memcpy(&v.x[j*w+k*v.chan_stride], &x[dx+(j+dy)*cols+k*chan_stride], sizeof(float)*w);
			}
		}

		return v;
	}

	mojo::matrix shift(int dx, int dy, mojo::pad_type edge_pad=mojo::zero)
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
					v.x[i+j*cols+k*chan_stride]=x[(cols-i-1)+j*cols+k*chan_stride];

		return v;
	}
	mojo::matrix flip_rows()
	{
		mojo::matrix v(cols, rows, chans);
		
		for (int k = 0; k<chans; k++)
			for (int j = 0; j<rows; j++)
				memcpy(&v.x[(rows-1-j)*cols + k*chan_stride],&x[j*cols + k*chan_stride], cols*sizeof(float));

		return v;
	}

	void clip(float min, float max)
	{
		int s = chan_stride*chans;
		for (int i = 0; i < s; i++)
		{
			if (x[i] < min) x[i] = min;
			if (x[i] > max) x[i]=max;
		}
	}


	void min_max(float *min, float *max, int *min_i=NULL, int *max_i=NULL)
	{
		int s = rows*cols;
		int mini = 0;
		int maxi = 0; 
		for (int c = 0; c < chans; c++)
		{
			const int t = chan_stride*c;
			for (int i = t; i < t+s; i++)
		{
			if (x[i] < x[mini]) mini = i;
			if (x[i] > x[maxi]) maxi = i;
		}
		}
		*min = x[mini];
		*max = x[maxi];
		if (min_i) *min_i = mini;
		if (max_i) *max_i = maxi;
	}

	float mean()
	{
		const int s = rows*cols;
		int cnt = 0;// channel*s;
		float average = 0;
		for (int c = 0; c < chans; c++)
		{
			const int t = chan_stride*c;
			for (int i = 0; i < s; i++)
				average += x[i + t];
		}
		average = average / (float)(s*chans);
		return average;
	}
	float remove_mean(int channel)
	{
		int s = rows*cols;
		int offset = channel*chan_stride;
		float average=0;
		for(int i=0; i<s; i++) average+=x[i+offset];		
		average= average/(float)s;
		for(int i=0; i<s; i++) x[i+offset]-=average;		
		return average;
	}

	float remove_mean()
	{
		float m=mean();
		int s = chan_stride*chans;
		//int offset = channel*s;
		for(int i=0; i<s; i++) x[i]-=m;		
		return m;
	}
	void fill(float val) { for(int i=0; i<_size; i++) x[i]=val; 
	}
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
		resize(m.cols, m.rows, m.chans, m.chan_aligned);
		memcpy(x,m.x,sizeof(float)*_size);
//		memcpy(empty_chan, m.empty_chan, chans);
		return *this;
	}

	int  size() const {return _size;} 
	
	void resize(int _w, int _h, int _c, int align_chans=0) { 
		chan_aligned = align_chans;
		int new_stride = calc_chan_stride(_w,_h);
		int s = new_stride*_c;
		if(s>_capacity) 
		{ 
			if(_capacity>0) delete_x(); _size = s; _capacity=_size; x = new_x(_size);
		}
		cols = _w; rows = _h; chans = _c; _size = s; chan_stride = new_stride; 
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
#ifndef MOJO_AVX
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

