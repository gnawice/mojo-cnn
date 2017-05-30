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
//    mojo.h:  include this one file to use mojo cnn without OpenMP
// ==================================================================== mojo ==

#pragma once

#define MOJO_AVX	// turn on AVX / SSE3 / SIMD optimizations
#define MOJO_OMP	// allow multi-threading through openmp
//#define MOJO_LUTS	// use look up tables, uses more memory
//#define MOJO_CV3	// use OpenCV 3.x utilities
//#define MOJO_CV2	// use OpenCV 2.x utilities
//#define MOJO_PROFILE_LAYERS  // std::cout layer names and latencies
#define MOJO_INTERNAL_THREADING // try to speed forward pass with internal threading

#ifdef MOJO_OMP
	#include <omp.h>
	#ifndef MOJO_INTERNAL_THREADING
		#define MOJO_THREAD_THIS_LOOP(a)
		#define MOJO_THREAD_THIS_LOOP_DYNAMIC(a)
	#else
		#ifdef _WIN32
			// this macro uses OMP where the loop is split up into equal chunks per thread
			#define MOJO_THREAD_THIS_LOOP(a) __pragma(omp parallel for num_threads(a))
			// this macro uses OMP where the loop is split up dynamically and work is taken by next available thread
			#define MOJO_THREAD_THIS_LOOP_DYNAMIC(a) __pragma(omp parallel for schedule(dynamic) num_threads(a))
		#else
			#define MOJO_THREAD_THIS_LOOP(a)
			#define MOJO_THREAD_THIS_LOOP_DYNAMIC(a)
			//#define MOJO_THREAD_THIS_LOOP(a) _Pragma("omp parallel for schedule(dynamic) num_threads(" #a ")")
			//#define MOJO_THREAD_THIS_LOOP_DYNAMIC(a) _Pragma("omp parallel for schedule(dynamic) num_threads(" #a ")")
	#endif
#endif //_WIN32
#else
	#define MOJO_THREAD_THIS_LOOP(a)
    #define MOJO_THREAD_THIS_LOOP_DYNAMIC(a)
#endif


#ifdef MOJO_AF
#include <arrayfire.h>
#define AF_RELEASE
	#ifdef MOJO_CUDA
		#define AF_CUDA
		#pragma comment(lib, "afcuda")
	#else
		#define AF_CPU
		#pragma comment(lib, "afcpu")
	#endif
#endif


#include "core_math.h"

#include "network.h" // this is the important thing
#include "util.h"


