/* NEON implementation of sin, cos, exp and log

   Inspired by Intel Approximate Math library, and based on the
   corresponding algorithms of the cephes math library
*/

/* Copyright (C) 2011  Julien Pommier

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

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#ifndef NEON_MATHFUN_H_
#define NEON_MATHFUN_H_

#include <arm_neon.h>

typedef float32x4_t v4sf; // vector of 4 float

// prototypes
inline v4sf log_ps(v4sf x);
inline v4sf exp_ps(v4sf x);
inline v4sf sin_ps(v4sf x);
inline v4sf cos_ps(v4sf x);
inline void sincos_ps(v4sf x, v4sf *s, v4sf *c);

#include "neon_mathfun.hxx"

#endif
#endif