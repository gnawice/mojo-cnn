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

#define MOJO_SSE3	// turn on SSE3 convolutions
//#define MOJO_OMP	// allow multi-threading through openmp
//#define MOJO_LUTS	// use look up tables, uses more memory
//#define MOJO_CV3	// use OpenCV 3.x utilities
//#define MOJO_CV2	// use OpenCV 2.x utilities
//#define MOJO_AF	// use ArrayFire math (not yet available)
//#define MOJO_CUDA	// use ArrayFire with NVidia CUDA (not yet available)


#ifdef MOJO_OMP
#include <omp.h>
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


