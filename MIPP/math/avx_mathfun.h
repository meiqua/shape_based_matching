/* 
   AVX implementation of sin, cos, sincos, exp and log

   Based on "sse_mathfun.h", by Julien Pommier
   http://gruntthepeon.free.fr/ssemath/

   Copyright (C) 2012 Giovanni Garberoglio
   Interdisciplinary Laboratory for Computational Science (LISC)
   Fondazione Bruno Kessler and University of Trento
   via Sommarive, 18
   I-38123 Trento (Italy)

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
#ifdef __AVX__
#ifndef AVX_MATHFUN_H_
#define AVX_MATHFUN_H_

#include <immintrin.h>

typedef __m256 v8sf; // vector of 8 float (avx)

// prototypes
inline v8sf log256_ps(v8sf x);
inline v8sf exp256_ps(v8sf x);
inline v8sf sin256_ps(v8sf x);
inline v8sf cos256_ps(v8sf x);
inline void sincos256_ps(v8sf x, v8sf *s, v8sf *c);

#include "avx_mathfun.hxx"

#endif
#endif