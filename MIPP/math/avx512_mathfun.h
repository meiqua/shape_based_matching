/* 
   AVX512 implementation of sin, cos, sincos, exp and log

   Based on "sse_mathfun.h", by Julien Pommier
   http://gruntthepeon.free.fr/ssemath/

   Copyright (C) 2017 Adrien Cassagne
   MIT license
*/
#ifdef __AVX512F__
#ifndef AVX512_MATHFUN_H_
#define AVX512_MATHFUN_H_

#include <immintrin.h>

typedef __m512 v16sf; // vector of 8 float (avx)

// prototypes
inline v16sf log512_ps(v16sf x);
inline v16sf exp512_ps(v16sf x);
inline v16sf sin512_ps(v16sf x);
inline v16sf cos512_ps(v16sf x);
inline void sincos512_ps(v16sf x, v16sf *s, v16sf *c);

#include "avx512_mathfun.hxx"

#endif
#endif
