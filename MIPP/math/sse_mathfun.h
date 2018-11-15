/* SIMD (SSE1+MMX or SSE2) implementation of sin, cos, exp and log

   Inspired by Intel Approximate Math library, and based on the
   corresponding algorithms of the cephes math library

   The default is to use the SSE1 version. If you define USE_SSE2 the
   the SSE2 intrinsics will be used in place of the MMX intrinsics. Do
   not expect any significant performance improvement with SSE2.
*/

/* Copyright (C) 2007  Julien Pommier

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

#ifdef __SSE__
#ifndef SSE_MATHFUN_H_
#define SSE_MATHFUN_H_

#include <xmmintrin.h>

typedef __m128 v4sf;  // vector of 4 float (sse1)

// prototypes
inline v4sf log_ps(v4sf x);
inline v4sf exp_ps(v4sf x);
inline v4sf sin_ps(v4sf x);
inline v4sf cos_ps(v4sf x);
inline void sincos_ps(v4sf x, v4sf *s, v4sf *c);

#include "sse_mathfun.hxx"

#endif
#endif