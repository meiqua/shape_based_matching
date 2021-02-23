/*
The MIT License (MIT)
Copyright (c) 2016 MIPP
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/*
 * mipp.h
 *
 *  Created on: 5 oct. 2014
 *      Author: Adrien Cassagne
 */

#ifndef MY_INTRINSICS_PLUS_PLUS_H_
#define MY_INTRINSICS_PLUS_PLUS_H_

#ifdef __AVX2__
//AVX2
#elif defined ( __AVX__ )
//AVX
#elif (defined(_M_AMD64) || defined(_M_X64))
//SSE2 x64
#define __SSE2__
#define __SSE__
#elif _M_IX86_FP == 2
//SSE2 x32
#define __SSE__
#define __SSE2__
#elif _M_IX86_FP == 1
//SSE x32
#define __SSE__
#else
//nothing
#endif

#define MIPP

#ifndef MIPP_NO_INTRINSICS
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#include "math/neon_mathfun.h"
#elif defined(__SSE__) || defined(__AVX__) || defined(__MIC__) || defined(__KNCNI__) || defined(__AVX512__) || defined(__AVX512F__)
// header for special functions: log, exp, sin, cos
#if !defined(__INTEL_COMPILER) && !defined(__ICL) && !defined(__ICC)
#if defined(__AVX512F__)
#include "math/avx512_mathfun.h"
#elif defined(__AVX__)
#include "math/avx_mathfun.h"
#elif defined(__SSE__)
#include "math/sse_mathfun.h"
#endif
#endif
#include <immintrin.h>
#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __SSE3__
#include <pmmintrin.h>
#endif
#ifdef __SSSE3__
#include <tmmintrin.h>
#endif
#ifdef __SSE4_1__
#include <smmintrin.h>
#endif
#else
#include "mipp_scalar_op.h"
#endif
#else
#include "mipp_scalar_op.h"
#endif

#include <unordered_map>
#include <typeindex>
#include <stdexcept>
#include <typeinfo>
#include <iostream>
#include <iomanip>
#include <cstddef>
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <map>

#if (defined(__GNUC__) || defined(__clang__) || defined(__llvm__)) && (defined(__linux__) || defined(__linux) || defined(__APPLE__))
#include <execinfo.h>
#include <unistd.h>
#include <cstdlib>
#endif

#ifdef _MSC_VER
#ifndef NOMINMAX
#define NOMINMAX
#endif
#undef min
#undef max
#endif

namespace mipp // My Intrinsics Plus Plus => mipp
{
// ------------------------------------------------------------------------------------------ myIntrinsics vector sizes
// --------------------------------------------------------------------------------------------------------------------
#ifndef MIPP_NO_INTRINSICS
// ------------------------------------------------------------------------------------------------------- ARM NEON-128
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
	const std::string InstructionType = "NEON";
	#define MIPP_NEON

	#define MIPP_REQUIRED_ALIGNMENT 16
#ifdef __aarch64__
	const std::string InstructionFullType = InstructionType + "v2";
	const std::string InstructionVersion  = "2";
	#define MIPP_NEONV2
	#define MIPP_INSTR_VERSION 2
	#define MIPP_64BIT
#else
	const std::string InstructionFullType = InstructionType + "v1";
	const std::string InstructionVersion  = "1";
	#define MIPP_NEONV1
	#define MIPP_INSTR_VERSION 1
#endif
	#define MIPP_BW
	#define MIPP_REGISTER_SIZE 128
	#define MIPP_LANES 1

	using msk   = uint32x4_t;
	using reg   = float32x4_t;
	using reg_2 = float32x2_t; // half a full register

	template <int N>
	inline reg toreg(const msk m) {
		return (reg)m;
	}

	inline std::vector<std::string> InstructionExtensions()
	{
		std::vector<std::string> ext;
#ifdef __ARM_FEATURE_FMA
		ext.push_back("FMA");
#endif
		return ext;
	}

// -------------------------------------------------------------------------------------------------------- X86 AVX-512
#elif defined(__MIC__) || defined(__KNCNI__) || defined(__AVX512__) || defined(__AVX512F__)
	const std::string InstructionType = "AVX512";
	#define MIPP_AVX512

	#define MIPP_REQUIRED_ALIGNMENT 64
	#define MIPP_64BIT

#if defined(__MIC__) || defined(__KNCNI__)
	#define MIPP_AVX512KNC
#endif
#ifdef __AVX512F__
	#define MIPP_AVX512F
#endif
#ifdef __AVX512BW__
	#define MIPP_AVX512BW
	#define MIPP_BW
#endif
#ifdef __AVX512CD__
	#define MIPP_AVX512CD
#endif
#ifdef __AVX512ER__
	#define MIPP_AVX512ER
#endif
#ifdef __AVX512PF__
	#define MIPP_AVX512PF
#endif
#ifdef __AVX512DQ__
	#define MIPP_AVX512DQ
#endif
#ifdef __AVX512VL__
	#define MIPP_AVX512VL
#endif
#ifdef __AVX512VBMI__
	#define MIPP_AVX512VBMI
#endif

	const std::string InstructionFullType = InstructionType;
	const std::string InstructionVersion  = "1";

	#define MIPP_INSTR_VERSION 1
	#define MIPP_REGISTER_SIZE 512
	#define MIPP_LANES 4

#ifdef __AVX512BW__
	using msk   = __mmask64;
#else
	using msk   = __mmask16;
#endif
	using reg   = __m512;
	using reg_2 = __m256; // half a full register

	template <int N>
	inline reg toreg(const msk m) {
		throw std::runtime_error("mipp: Invalid mask size 'N' = " + std::to_string(N) + ".");
	}

	inline std::vector<std::string> InstructionExtensions()
	{
		std::vector<std::string> ext;
#if defined(__MIC__) || defined(__KNCNI__)
		ext.push_back("KNC");
#endif
#ifdef __AVX512F__
		ext.push_back("F");
#endif
#ifdef __AVX512BW__
		ext.push_back("BW");
#endif
#ifdef __AVX512CD__
		ext.push_back("CD");
#endif
#ifdef __AVX512ER__
		ext.push_back("ER");
#endif
#ifdef __AVX512PF__
		ext.push_back("PF");
#endif
#ifdef __AVX512DQ__
		ext.push_back("DQ");
#endif
#ifdef __AVX512VL__
		ext.push_back("VL");
#endif
		return ext;
	}

// -------------------------------------------------------------------------------------------------------- X86 AVX-256
#elif defined(__AVX__)
	const std::string InstructionType = "AVX";
	#define MIPP_AVX

	#define MIPP_REQUIRED_ALIGNMENT 32
	#define MIPP_64BIT
#ifdef __AVX2__
	const std::string InstructionFullType = InstructionType + "2";
	const std::string InstructionVersion  = "2";
	#define MIPP_AVX2
	#define MIPP_INSTR_VERSION 2
	#define MIPP_BW
#else
	const std::string InstructionFullType = InstructionType;
	const std::string InstructionVersion  = "1";
	#define MIPP_AVX1
	#define MIPP_INSTR_VERSION 1
#endif
	#define MIPP_REGISTER_SIZE 256
	#define MIPP_LANES 2

	using msk   = __m256i;
	using reg   = __m256;
	using reg_2 = __m128; // half a full register

	template <int N>
	inline reg toreg(const msk m) {
		return _mm256_castsi256_ps(m);
	}

	inline std::vector<std::string> InstructionExtensions()
	{
		std::vector<std::string> ext;
#ifdef __FMA__
		ext.push_back("FMA");
#endif
		return ext;
	}

// -------------------------------------------------------------------------------------------------------- X86 SSE-128
#elif defined(__SSE__)
	const std::string InstructionType = "SSE";
	#define MIPP_SSE

	#define MIPP_REQUIRED_ALIGNMENT 16
#ifdef __SSE2__
	#define MIPP_64BIT
	#define MIPP_BW
#endif
#ifdef __SSE4_2__
	const std::string InstructionFullType = InstructionType + "4.2";
	const std::string InstructionVersion  = "4.2";
	#define MIPP_SSE4_2
	#define MIPP_INSTR_VERSION 42
#elif defined(__SSE4_1__)
	const std::string InstructionFullType = InstructionType + "4.1";
	const std::string InstructionVersion  = "4.1";
	#define MIPP_SSE4_1
	#define MIPP_INSTR_VERSION 41
#elif defined(__SSSE3__)
	const std::string InstructionFullType = "SSSE3";
	const std::string InstructionVersion  = "3";
	#define MIPP_SSSE3
	#define MIPP_INSTR_VERSION 31
#elif defined(__SSE3__)
	const std::string InstructionFullType = InstructionType + "3";
	const std::string InstructionVersion  = "3";
	#define MIPP_SSE3
	#define MIPP_INSTR_VERSION 3
#elif defined(__SSE2__)
	const std::string InstructionFullType = InstructionType + "2";
	const std::string InstructionVersion  = "2";
	#define MIPP_SSE2
	#define MIPP_INSTR_VERSION 2
#else
	const std::string InstructionFullType = InstructionType;
	const std::string InstructionVersion  = "1";
	#define MIPP_SSE1
	#define MIPP_INSTR_VERSION 1
#endif
	#define MIPP_REGISTER_SIZE 128
	#define MIPP_LANES 1

	using msk   = __m128i;
	using reg   = __m128;
	using reg_2 = __m128d; // half a full register (information is in the lower part of the 128 bit register)

	template <int N>
	inline reg toreg(const msk m) {
		return _mm_castsi128_ps(m);
	}

	inline std::vector<std::string> InstructionExtensions()
	{
		std::vector<std::string> ext;
		return ext;
	}

// ------------------------------------------------------------------------------------------------- MIPP_NO_INTRINSICS
#else
	const std::string InstructionType = "NO";
	#define MIPP_NO

	const std::string InstructionFullType = "NO_INTRINSICS";
	const std::string InstructionVersion  = "1";

	#define MIPP_NO_INTRINSICS
	#define MIPP_REQUIRED_ALIGNMENT 1
#if UINTPTR_MAX == 0xffffffffffffffff
#define MIPP_64BIT
#endif
	#define MIPP_BW
	#define MIPP_INSTR_VERSION 1
	#define MIPP_REGISTER_SIZE 0
	#define MIPP_LANES 1

	using msk   = uint8_t;
	using reg   = uint32_t;
	using reg_2 = uint16_t;

	template <int N>
	inline reg toreg(const msk m) {
		return (reg)m;
	}

	inline std::vector<std::string> InstructionExtensions()
	{
		std::vector<std::string> ext;
		return ext;
	}
#endif

// ------------------------------------------------------------------------------------------------- MIPP_NO_INTRINSICS
#else
	const std::string InstructionType     = "NO";
	#define MIPP_NO

	const std::string InstructionFullType = "NO_INTRINSICS";
	const std::string InstructionVersion  = "1";

	#define MIPP_REQUIRED_ALIGNMENT 1
#if UINTPTR_MAX == 0xffffffffffffffff
#define MIPP_64BIT
#endif
	#define MIPP_BW
	#define MIPP_INSTR_VERSION 1
	#define MIPP_REGISTER_SIZE 0
	#define MIPP_LANES 1

	using msk   = uint8_t;
	using reg   = uint32_t;
	using reg_2 = uint16_t;

	template <int N>
	inline reg toreg(const msk m) {
		return (reg)m;
	}

	inline std::vector<std::string> InstructionExtensions()
	{
		std::vector<std::string> ext;
		return ext;
	}
#endif

constexpr uint32_t RequiredAlignment = MIPP_REQUIRED_ALIGNMENT;
constexpr uint32_t RegisterSizeBit   = MIPP_REGISTER_SIZE;
constexpr uint32_t Lanes             = MIPP_LANES;

#ifdef MIPP_64BIT
const bool Support64Bit = true;
#else
const bool Support64Bit = false;
#endif
#ifdef MIPP_BW
const bool SupportByteWord = true;
#else
const bool SupportByteWord = false;
#endif

typedef struct regx2 { reg val[2]; } regx2;

template <typename T>
constexpr int32_t nElmtsPerRegister()
{
#ifdef MIPP_NO_INTRINSICS
	return 1;
#else
	return RegisterSizeBit / (8 * sizeof(T));
#endif
}

template <typename T>
constexpr int32_t nElReg()
{
#ifdef MIPP_NO_INTRINSICS
	return 1;
#else
	return RegisterSizeBit / (8 * sizeof(T));
#endif
}

template <typename T>
constexpr int32_t N()
{
#ifndef MIPP_NO_INTRINSICS
	return mipp::nElReg<T>();
#else
	return 1;
#endif
}

template <typename T>
inline bool isAligned(const T *ptr)
{
#ifdef MIPP_ALIGNED_LOADS
	return (((uintptr_t)ptr) % (RegisterSizeBit / 8)) == 0;
#else
	return true;
#endif
}

// --------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------- memory allocator
template <typename T>
T* malloc(uint32_t nData)
{
	T* ptr = nullptr;

#if !defined(MIPP_NO_INTRINSICS) && (defined(__SSE2__) || defined(__AVX__) || defined(__MIC__) || defined(__KNCNI__) || defined(__AVX512__) || defined(__AVX512F__))
	ptr = (T*)_mm_malloc(nData * sizeof(T), mipp::RequiredAlignment);
#else
	ptr = new T[nData];
#endif

	return ptr;
}

template <typename T>
void free(T* ptr)
{
#if !defined(MIPP_NO_INTRINSICS) && (defined(__SSE2__) || defined(__AVX__) || defined(__MIC__) || defined(__KNCNI__) || defined(__AVX512__) || defined(__AVX512F__))
	_mm_free(ptr);
#else
	delete[] ptr;
#endif
}

template <class T>
struct allocator
{
	typedef T value_type;
	allocator() { }
	template <class C> allocator(const allocator<C>& other) { }
	T* allocate(std::size_t n) { return mipp::malloc<T>((int)n); }
	void deallocate(T* p, std::size_t n) { mipp::free<T>(p); }
};

// returns true if and only if storage allocated from ma1 can be deallocated from ma2, and vice versa.
// always returns true for stateless allocators.
template <class C1, class C2>
bool operator==(const allocator<C1>& ma1, const allocator<C2>& ma2) { return true; }

template <class C1, class C2>
bool operator!=(const allocator<C1>& ma1, const allocator<C2>& ma2) { return !(ma1 == ma2); }

// override vector type
template<class T> using vector = std::vector<T, allocator<T>>;

// --------------------------------------------------------------------------------------------------- memory allocator
// --------------------------------------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------------------- myIntrinsics prototypes
// --------------------------------------------------------------------------------------------------------------------

static inline std::string get_back_trace()
{
	std::string bt_str;
#if defined(MIPP_ENABLE_BACKTRACE) && (defined(__GNUC__) || defined(__clang__) || defined(__llvm__)) && (defined(__linux__) || defined(__linux) || defined(__APPLE__))
	const int bt_max_depth = 32;
	void *bt_array[bt_max_depth];

	size_t size = backtrace(bt_array, bt_max_depth);
	char** bt_symbs = backtrace_symbols(bt_array, size);

	bt_str += "\nBacktrace:";
	for (size_t i = 0; i < size; i++)
		bt_str += "\n" + std::string(bt_symbs[i]);
	free(bt_symbs);
#endif

	return bt_str;
}

template <typename T>
static inline void errorMessage(std::string instr)
{
	// define type names
	std::unordered_map<std::type_index,std::string> type_names;
	type_names[typeid(int8_t)  ] = "int8_t";
	type_names[typeid(uint8_t) ] = "uint8_t";
	type_names[typeid(int16_t) ] = "int16_t";
	type_names[typeid(uint16_t)] = "uint16_t";
	type_names[typeid(int32_t) ] = "int32_t";
	type_names[typeid(uint32_t)] = "uint32_t";
	type_names[typeid(int64_t) ] = "int64_t";
	type_names[typeid(uint64_t)] = "uint64_t";
	type_names[typeid(float)   ] = "float";
	type_names[typeid(double)  ] = "double";

	std::string message;
	if (RegisterSizeBit == 0)
		message = "mipp::" + instr + "<" + type_names[typeid(T)] + "> (" + InstructionFullType + ") is undefined!, "
		          "try to add -mfpu=neon-vfpv4, -msse4.2, -mavx, -march=native... at the compile time.";
	else
		message = "mipp::" + instr + "<" + type_names[typeid(T)] + "> (" + InstructionFullType + ") is undefined!";

	message += get_back_trace();

	throw std::runtime_error(message);
}

template <int N>
static inline void errorMessage(std::string instr)
{
	std::string message;
	if (RegisterSizeBit == 0)
		message = "mipp::" + instr + "<" + std::to_string(N) + "> (" + InstructionFullType + ") is undefined!, "
		          "try to add -mfpu=neon-vfpv4, -msse4.2, -mavx, -march=native... at the compile time.";
	else
		message = "mipp::" + instr + "<" + std::to_string(N) + "> (" + InstructionFullType + ") is undefined!";

	message += get_back_trace();

	throw std::runtime_error(message);
}

template <typename T1, typename T2>
static inline void errorMessage(std::string instr)
{
	// define type names
	std::unordered_map<std::type_index,std::string> type_names;
	type_names[typeid(int8_t  )] = "int8_t";
	type_names[typeid(uint8_t )] = "uint8_t";
	type_names[typeid(int16_t )] = "int16_t";
	type_names[typeid(uint16_t)] = "uint16_t";
	type_names[typeid(int32_t )] = "int32_t";
	type_names[typeid(uint32_t)] = "uint32_t";
	type_names[typeid(int64_t )] = "int64_t";
	type_names[typeid(uint64_t)] = "uint64_t";
	type_names[typeid(float   )] = "float";
	type_names[typeid(double  )] = "double";

	std::string message;
	if (RegisterSizeBit == 0)
		message = "mipp::" + instr + "<" + type_names[typeid(T1)] + "," + type_names[typeid(T2)] + "> (" +
		          InstructionFullType + ") is undefined!, try to add -mfpu=neon, -msse4.2, -mavx, -march=native... "
		          "at the compile time.";
	else
		message = "mipp::" + instr + "<" + type_names[typeid(T1)] + "," + type_names[typeid(T2)] + "> (" +
		          InstructionFullType + ") is undefined!";

	message += get_back_trace();

	throw std::runtime_error(message);
}

template <typename T> inline reg   load         (const T*)                        { errorMessage<T>("load");          exit(-1); }
template <typename T> inline reg   loadu        (const T*)                        { errorMessage<T>("loadu");         exit(-1); }
template <typename T> inline void  store        (T*, const reg)                   { errorMessage<T>("store");         exit(-1); }
template <typename T> inline void  storeu       (T*, const reg)                   { errorMessage<T>("storeu");        exit(-1); }
template <typename T> inline reg   set          (const T[nElReg<T>()])            { errorMessage<T>("set");           exit(-1); }
#ifdef _MSC_VER
template <int      N> inline msk   set          (const bool[])                    { errorMessage<N>("set");           exit(-1); }
#else
template <int      N> inline msk   set          (const bool[N])                   { errorMessage<N>("set");           exit(-1); }
#endif
template <typename T> inline reg   set1         (const T)                         { errorMessage<T>("set1");          exit(-1); }
template <int      N> inline msk   set1         (const bool)                      { errorMessage<N>("set1");          exit(-1); }
template <typename T> inline reg   set0         ()                                { errorMessage<T>("set0");          exit(-1); }
template <int      N> inline msk   set0         ()                                { errorMessage<N>("set0");          exit(-1); }
template <typename T> inline reg_2 low          (const reg)                       { errorMessage<T>("low");           exit(-1); }
template <typename T> inline reg_2 high         (const reg)                       { errorMessage<T>("high");          exit(-1); }
#ifdef MIPP_NO_INTRINSICS // tricks to avoid compiling errors with Clang...
template <typename T> inline reg   cmask        (const uint32_t[1])               { errorMessage<T>("cmask");         exit(-1); }
template <typename T> inline reg   cmask2       (const uint32_t[1])               { errorMessage<T>("cmask2");        exit(-1); }
template <typename T> inline reg   cmask4       (const uint32_t[1])               { errorMessage<T>("cmask4");        exit(-1); }
#else
template <typename T> inline reg   cmask        (const uint32_t[nElReg<T>()])     { errorMessage<T>("cmask");         exit(-1); }
template <typename T> inline reg   cmask2       (const uint32_t[nElReg<T>()/2])   { errorMessage<T>("cmask2");        exit(-1); }
template <typename T> inline reg   cmask4       (const uint32_t[nElReg<T>()/4])   { errorMessage<T>("cmask4");        exit(-1); }
#endif

template <typename T> inline reg   shuff        (const reg, const reg)            { errorMessage<T>("shuff");         exit(-1); }
template <typename T> inline reg   shuff2       (const reg, const reg)            { errorMessage<T>("shuff2");        exit(-1); }
template <typename T> inline reg   shuff4       (const reg, const reg)            { errorMessage<T>("shuff4");        exit(-1); }
template <typename T> inline reg   interleavelo (const reg, const reg)            { errorMessage<T>("interleavelo");  exit(-1); }
template <typename T> inline reg   interleavehi (const reg, const reg)            { errorMessage<T>("interleavehi");  exit(-1); }
template <typename T> inline reg   interleavelo2(const reg, const reg)            { errorMessage<T>("interleavelo2"); exit(-1); }
template <typename T> inline reg   interleavehi2(const reg, const reg)            { errorMessage<T>("interleavehi2"); exit(-1); }
template <typename T> inline reg   interleavelo4(const reg, const reg)            { errorMessage<T>("interleavelo4"); exit(-1); }
template <typename T> inline reg   interleavehi4(const reg, const reg)            { errorMessage<T>("interleavehi4"); exit(-1); }
template <typename T> inline regx2 interleave   (const reg, const reg)            { errorMessage<T>("interleave");    exit(-1); }
template <typename T> inline regx2 interleave2  (const reg, const reg)            { errorMessage<T>("interleave2");   exit(-1); }
template <typename T> inline regx2 interleave4  (const reg, const reg)            { errorMessage<T>("interleave4");   exit(-1); }
template <typename T> inline reg   interleave   (const reg)                       { errorMessage<T>("interleave");    exit(-1); }
template <typename T> inline regx2 interleavex2 (const reg, const reg)            { errorMessage<T>("interleavex2");  exit(-1); }
template <typename T> inline reg   interleavex4 (const reg)                       { errorMessage<T>("interleavex4");  exit(-1); }
template <typename T> inline reg   interleavex16(const reg)                       { errorMessage<T>("interleavex16"); exit(-1); }
template <typename T> inline void  transpose    (      reg[nElReg<T>()])          { errorMessage<T>("transpose");     exit(-1); }
template <typename T> inline void  transpose8x8 (      reg[8])                    { errorMessage<T>("transpose8x8");  exit(-1); }
template <typename T> inline void  transpose2   (      reg[nElReg<T>()/2])        { errorMessage<T>("transpose2");    exit(-1); }
template <typename T> inline void  transpose28x8(      reg[8])                    { errorMessage<T>("transpose28x8"); exit(-1); }
template <typename T> inline void  transpose4   (      reg[nElReg<T>()/2])        { errorMessage<T>("transpose4");    exit(-1); }
template <typename T> inline void  transpose48x8(      reg[8])                    { errorMessage<T>("transpose48x8"); exit(-1); }
template <typename T> inline reg   andb         (const reg, const reg)            { errorMessage<T>("andb");          exit(-1); }
template <int      N> inline msk   andb         (const msk, const msk)            { errorMessage<N>("andb");          exit(-1); }
template <typename T> inline reg   andnb        (const reg, const reg)            { errorMessage<T>("andnb");         exit(-1); }
template <int      N> inline msk   andnb        (const msk, const msk)            { errorMessage<N>("andnb");         exit(-1); }
template <typename T> inline reg   notb         (const reg)                       { errorMessage<T>("notb");          exit(-1); }
template <int      N> inline msk   notb         (const msk)                       { errorMessage<N>("notb");          exit(-1); }
template <typename T> inline reg   orb          (const reg, const reg)            { errorMessage<T>("orb");           exit(-1); }
template <int      N> inline msk   orb          (const msk, const msk)            { errorMessage<N>("orb");           exit(-1); }
template <typename T> inline reg   xorb         (const reg, const reg)            { errorMessage<T>("xorb");          exit(-1); }
template <int      N> inline msk   xorb         (const msk, const msk)            { errorMessage<N>("xorb");          exit(-1); }
template <typename T> inline reg   lshift       (const reg, const uint32_t)       { errorMessage<T>("lshift");        exit(-1); }
template <int      N> inline msk   lshift       (const msk, const uint32_t)       { errorMessage<N>("lshift");        exit(-1); }
template <typename T> inline reg   rshift       (const reg, const uint32_t)       { errorMessage<T>("rshift");        exit(-1); }
template <int      N> inline msk   rshift       (const msk, const uint32_t)       { errorMessage<N>("rshift");        exit(-1); }
template <typename T> inline msk   cmpeq        (const reg, const reg)            { errorMessage<T>("cmpeq");         exit(-1); }
template <typename T> inline msk   cmpneq       (const reg, const reg)            { errorMessage<T>("cmpneq");        exit(-1); }
template <typename T> inline msk   cmplt        (const reg, const reg)            { errorMessage<T>("cmplt");         exit(-1); }
template <typename T> inline msk   cmple        (const reg, const reg)            { errorMessage<T>("cmple");         exit(-1); }
template <typename T> inline msk   cmpgt        (const reg, const reg)            { errorMessage<T>("cmpgt");         exit(-1); }
template <typename T> inline msk   cmpge        (const reg, const reg)            { errorMessage<T>("cmpge");         exit(-1); }
template <typename T> inline reg   add          (const reg, const reg)            { errorMessage<T>("add");           exit(-1); }
template <typename T> inline reg   sub          (const reg, const reg)            { errorMessage<T>("sub");           exit(-1); }
template <typename T> inline reg   mul          (const reg, const reg)            { errorMessage<T>("mul");           exit(-1); }
template <typename T> inline reg   div          (const reg, const reg)            { errorMessage<T>("div");           exit(-1); }
template <typename T> inline reg   min          (const reg, const reg)            { errorMessage<T>("min");           exit(-1); }

template <typename T> inline reg   max          (const reg, const reg)            { errorMessage<T>("max");           exit(-1); }
template <typename T> inline reg   msb          (const reg)                       { errorMessage<T>("msb");           exit(-1); }
template <typename T> inline reg   msb          (const reg, const reg)            { errorMessage<T>("msb");           exit(-1); }
template <typename T> inline msk   sign         (const reg)                       { errorMessage<T>("sign");          exit(-1); }
template <typename T> inline reg   neg          (const reg, const reg)            { errorMessage<T>("neg");           exit(-1); }
template <typename T> inline reg   neg          (const reg, const msk)            { errorMessage<T>("neg");           exit(-1); }
template <typename T> inline reg   abs          (const reg)                       { errorMessage<T>("abs");           exit(-1); }
template <typename T> inline reg   sqrt         (const reg)                       { errorMessage<T>("sqrt");          exit(-1); }
template <typename T> inline reg   rsqrt        (const reg)                       { errorMessage<T>("rsqrt");         exit(-1); }
template <typename T> inline reg   log          (const reg)                       { errorMessage<T>("log");           exit(-1); }
template <typename T> inline reg   exp          (const reg)                       { errorMessage<T>("exp");           exit(-1); }
template <typename T> inline reg   sin          (const reg)                       { errorMessage<T>("sin");           exit(-1); }
template <typename T> inline reg   cos          (const reg)                       { errorMessage<T>("cos");           exit(-1); }
template <typename T> inline void  sincos       (const reg, reg&, reg&)           { errorMessage<T>("sincos");        exit(-1); }
template <typename T> inline reg   fmadd        (const reg, const reg, const reg) { errorMessage<T>("fmadd");         exit(-1); }
template <typename T> inline reg   fnmadd       (const reg, const reg, const reg) { errorMessage<T>("fnmadd");        exit(-1); }
template <typename T> inline reg   fmsub        (const reg, const reg, const reg) { errorMessage<T>("fmsub");         exit(-1); }
template <typename T> inline reg   fnmsub       (const reg, const reg, const reg) { errorMessage<T>("fnmsub");        exit(-1); }
template <typename T> inline reg   blend        (const reg, const reg, const msk) { errorMessage<T>("blend");         exit(-1); }
template <typename T> inline reg   lrot         (const reg)                       { errorMessage<T>("lrot");          exit(-1); }
template <typename T> inline reg   rrot         (const reg)                       { errorMessage<T>("rrot");          exit(-1); }
template <typename T> inline reg   div2         (const reg)                       { errorMessage<T>("div2");          exit(-1); }
template <typename T> inline reg   div4         (const reg)                       { errorMessage<T>("div4");          exit(-1); }
template <typename T> inline reg   sat          (const reg, T, T)                 { errorMessage<T>("sat");           exit(-1); }
template <typename T> inline reg   round        (const reg)                       { errorMessage<T>("round");         exit(-1); }
template <typename T> inline bool  testz        (const reg, const reg)            { errorMessage<T>("testz");         exit(-1); }
template <int      N> inline bool  testz        (const msk, const msk)            { errorMessage<N>("testz");         exit(-1); }
template <typename T> inline bool  testz        (const reg)                       { errorMessage<T>("testz");         exit(-1); }
template <int      N> inline bool  testz        (const msk)                       { errorMessage<N>("testz");         exit(-1); }

template <typename T1, typename T2>
inline reg cvt(const reg) {
	errorMessage<T1,T2>("cvt");
	exit(-1);
}

template <typename T1, typename T2>
inline reg cvt(const reg_2) {
	errorMessage<T1,T2>("cvt");
	exit(-1);
}

template <typename T1, typename T2>
inline reg pack(const reg, const reg) {
	errorMessage<T1,T2>("pack");
	exit(-1);
}

// ------------------------------------------------------------------------------------------------------------ aliases
// --------------------------------------------------------------------------------------------------------------------
template <typename T> inline reg copysign(const reg r1, const reg r2) { return neg<T>(r1, r2); }
template <typename T> inline reg copysign(const reg r1, const msk r2) { return neg<T>(r1, r2); }

// --------------------------------------------------------------------------------- hyperbolic trigonometric functions
// --------------------------------------------------------------------------------------------------------------------
template <typename T>
inline reg tan(const reg r)
{
	mipp::reg sin, cos;
	mipp::sincos<T>(r, sin, cos);
	return mipp::div<T>(sin, cos);
}

template <typename T>
inline reg sinh(const reg r)
{
	mipp::reg zero = mipp::set0<T>();
	mipp::reg half = mipp::set1<T>((T)0.5);
	return mipp::mul<T>(mipp::sub<T>(mipp::exp<T>(r), mipp::exp<T>(mipp::sub<T>(zero,r))), half);
}

template <typename T>
inline reg cosh(const reg r)
{
	mipp::reg zero = mipp::set0<T>();
	mipp::reg half = mipp::set1<T>((T)0.5);
	return mipp::mul<T>(mipp::add<T>(mipp::exp<T>(r), mipp::exp<T>(mipp::sub<T>(zero,r))), half);
}

template <typename T>
inline reg tanh(const reg r)
{
	mipp::reg zero = mipp::set0<T>();
	auto epx = mipp::exp<T>(r);
	auto emx = mipp::exp<T>(mipp::sub<T>(zero,r));
	return mipp::div<T>(mipp::sub<T>(epx, emx), mipp::add<T>(epx, emx));
}

template <typename T>
inline reg asinh(const reg r)
{
	mipp::reg one = mipp::set1<T>((T)1);
	return mipp::log<T>(mipp::add<T>(r, mipp::sqrt<T>(mipp::add<T>(mipp::mul<T>(r, r), one))));
}

template <typename T>
inline reg acosh(const reg r)
{
	mipp::reg one = mipp::set1<T>((T)1);
	return mipp::log<T>(mipp::add<T>(r, mipp::sqrt<T>(mipp::sub<T>(mipp::mul<T>(r, r), one))));
}

template <typename T>
inline reg atanh(const reg r)
{
	mipp::reg one = mipp::set1<T>((T)1);
	mipp::reg half = mipp::set1<T>((T)0.5);
	return mipp::mul<T>(half, mipp::log<T>(mipp::div<T>(mipp::add<T>(one, r), mipp::sub<T>(one, r))));
}

// template <typename T>
// inline reg csch(const reg r)
// {
// 	mipp::reg zero = mipp::set0<T>();
// 	mipp::reg two = mipp::set1<T>((T)2);
// 	return mipp::div<T>(two, mipp::sub<T>(mipp::exp<T>(r), mipp::exp<T>(mipp::sub<T>(zero,r))));
// }

// template <typename T>
// inline reg sech(const reg r)
// {
// 	mipp::reg zero = mipp::set0<T>();
// 	mipp::reg two = mipp::set1<T>((T)2);
// 	return mipp::div<T>(two, mipp::add<T>(mipp::exp<T>(r), mipp::exp<T>(mipp::sub<T>(zero,r))));
// }

// template <typename T>
// inline reg coth(const reg r)
// {
// 	mipp::reg zero = mipp::set0<T>();
// 	auto epx = mipp::exp<T>(r);
// 	auto emx = mipp::exp<T>(mipp::sub<T>(zero,r));
// 	return mipp::div<T>(mipp::add<T>(epx, emx), mipp::sub<T>(epx, emx));
// }

// template <typename T>
// inline reg acsch(const reg r)
// {
// 	mipp::reg one = mipp::set1<T>((T)1);
// 	return mipp::log<T>(mipp::div<T>(mipp::add<T>(one, mipp::sqrt<T>(mipp::add<T>(one, mipp::mul<T>(r, r)))), r));
// }

// template <typename T>
// inline reg asech(const reg r)
// {
// 	mipp::reg one = mipp::set1<T>((T)1);
// 	return mipp::log<T>(mipp::div<T>(mipp::add<T>(one, mipp::sqrt<T>(mipp::sub<T>(one, mipp::mul<T>(r, r)))), r));
// }

// template <typename T>
// inline reg acoth(const reg r)
// {
// 	mipp::reg one = mipp::set1<T>((T)1);
// 	mipp::reg half = mipp::set1<T>((T)0.5);
// 	return mipp::mul<T>(half, mipp::log<T>(mipp::div<T>(mipp::add<T>(r, one), mipp::sub<T>(r, one))));
// }


// ------------------------------------------------------------------------------------------------------------ masking
// --------------------------------------------------------------------------------------------------------------------

template <typename T> using proto_i1 = reg (*)(const reg a);
template <typename T> using proto_i2 = reg (*)(const reg a, const reg b);
template <typename T> using proto_i3 = reg (*)(const reg a, const reg b, const reg c);

template <typename T, proto_i1<T> I1>
inline reg mask(const msk m, const reg src, const reg a)
{
	return blend<T>(I1(a), src, m);
}

template <typename T, proto_i2<T> I2>
inline reg mask(const msk m, const reg src, const reg a, const reg b)
{
	return blend<T>(I2(a, b), src, m);
}

template <typename T, proto_i3<T> I3>
inline reg mask(const msk m, const reg src, const reg a, const reg b, const reg c)
{
	return blend<T>(I3(a, b, c), src, m);
}

template <typename T, proto_i1<T> I1>
inline reg maskz(const msk m, const reg a)
{
	auto m_reg = toreg<N<T>()>(m);
	auto a_modif = I1(a);
	return andb<T>(m_reg, a_modif);
}

template <typename T, proto_i2<T> I2>
inline reg maskz(const msk m, const reg a, const reg b)
{
	auto m_reg = toreg<N<T>()>(m);
	auto a_modif = I2(a, b);
	return andb<T>(m_reg, a_modif);
}

template <typename T, proto_i3<T> I3>
inline reg maskz(const msk m, const reg a, const reg b, const reg c)
{
	auto m_reg = toreg<N<T>()>(m);
	auto a_modif = I3(a, b, c);
	return andb<T>(m_reg, a_modif);
}

// -------------------------------------------------------------------------------------------------------- obj masking

template <typename T>
class Reg;

template <int N>
class Msk;

template <typename T> inline Reg<T> blend(const Reg<T> v1, const Reg<T> v2, const Msk<N<T>()> m);
template <typename T> inline Reg<T> andb (const Reg<T> v1, const Reg<T> v2);

template <typename T> using proto_I1 = Reg<T> (*)(const Reg<T> a);
template <typename T> using proto_I2 = Reg<T> (*)(const Reg<T> a, const Reg<T> b);
template <typename T> using proto_I3 = Reg<T> (*)(const Reg<T> a, const Reg<T> b, const Reg<T> c);

template <typename T, proto_I1<T> I1>
inline Reg<T> mask(const Msk<N<T>()> m, const Reg<T> src, const Reg<T> a)
{
#ifndef MIPP_NO
	return blend<T>(I1(a), src, m);
#else
	return m.m ? I1(a) : src;
#endif
}

template <typename T, proto_I2<T> I2>
inline Reg<T> mask(const Msk<N<T>()> m, const Reg<T> src, const Reg<T> a, const Reg<T> b)
{
#ifndef MIPP_NO
	return blend<T>(I2(a, b), src, m);
#else
	return m.m ? I2(a, b) : src;
#endif
}

template <typename T, proto_I3<T> I3>
inline Reg<T> mask(const Msk<N<T>()> m, const Reg<T> src, const Reg<T> a, const Reg<T> b, const Reg<T> c)
{
#ifndef MIPP_NO
	return blend<T>(I3(a, b, c), src, m);
#else
	return m.m ? I3(a, b, c) : src;
#endif
}

template <typename T, proto_I1<T> I1>
inline Reg<T> maskz(const Msk<N<T>()> m, const Reg<T> a)
{
#ifndef MIPP_NO
	auto m_reg = m.template toReg<T>();
	auto a_modif = I1(a);
	return andb<T>(m_reg, a_modif);
#else
	return m.m ? I1(a) : Reg<T>((T)0);
#endif
}

template <typename T, proto_I2<T> I2>
inline Reg<T> maskz(const Msk<N<T>()> m, const Reg<T> a, const Reg<T> b)
{
#ifndef MIPP_NO
	auto m_reg = m.template toReg<T>();
	auto a_modif = I2(a, b);
	return andb<T>(m_reg, a_modif);
#else
	return m.m ? I2(a, b) : Reg<T>((T)0);
#endif
}

template <typename T, proto_I3<T> I3>
inline Reg<T> maskz(const Msk<N<T>()> m, const Reg<T> a, const Reg<T> b, const Reg<T> c)
{
#ifndef MIPP_NO
	auto m_reg = m.template toReg<T>();
	auto a_modif = I3(a, b, c);
	return andb<T>(m_reg, a_modif);
#else
	return m.m ? I3(a, b, c) : Reg<T>((T)0);
#endif
}

// --------------------------------------------------------------------------------------- myIntrinsics implementations
// --------------------------------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------------------------- dump

template <typename T>
void dump(const mipp::reg r, std::ostream &stream = std::cout, const uint32_t elmtWidth = 6)
{
	constexpr int32_t lane_size = (int32_t)(mipp::N<T>() / mipp::Lanes);

//	const T* data = (T*)&r;
	T data[mipp::nElReg<T>()];
	store<T>(data, r);

	stream << "[";
	for (uint32_t l = 0; l < mipp::Lanes; l++)
	{
		for (auto i = 0; i < lane_size; i++)
			stream << std::setw(elmtWidth) << +data[l * lane_size +i] << ((i < lane_size -1) ? ", " : "");
		stream << (((int)l < (int)mipp::Lanes -1) ? " | " : "");
	}
	stream << "]";
}

template <int N>
void dump(const mipp::msk m, std::ostream &stream = std::cout, const uint32_t elmtWidth = 6)
{
	constexpr int32_t lane_size = (int32_t)(N / mipp::Lanes);
	constexpr int bits = mipp::RegisterSizeBit / N;

	const auto r = toreg<N>(m);

	stream << "[";
	if (bits == 8)
	{
		// const int8_t* data = (int8_t*)&r;
		int8_t data[N];
		store<int8_t>(data, r);

		for (uint32_t l = 0; l < mipp::Lanes; l++)
		{
			for (auto i = 0; i < lane_size; i++)
				stream << std::setw(elmtWidth) << (data[l * lane_size +i] ? 1 : 0) << ((i < lane_size -1) ? ", " : "");
			stream << (((int)l < (int)mipp::Lanes -1) ? " | " : "");
		}
	}
	else if (bits == 16)
	{
		// const int16_t* data = (int16_t*)&r;
		int16_t data[N];
		store<int16_t>(data, r);

		for (uint32_t l = 0; l < (int)mipp::Lanes; l++)
		{
			for (auto i = 0; i < lane_size; i++)
				stream << std::setw(elmtWidth) << (data[l * lane_size +i] ? 1 : 0) << ((i < lane_size -1) ? ", " : "");
			stream << (((int)l < (int)mipp::Lanes -1) ? " | " : "");
		}
	}
	else if (bits == 32)
	{
		// const int32_t* data = (int32_t*)&r;
		int32_t data[N];
		store<int32_t>(data, r);

		for (uint32_t l = 0; l < (int)mipp::Lanes; l++)
		{
			for (auto i = 0; i < lane_size; i++)
				stream << std::setw(elmtWidth) << (data[l * lane_size +i] ? 1 : 0) << ((i < lane_size -1) ? ", " : "");
			stream << (((int)l < (int)mipp::Lanes -1) ? " | " : "");
		}
	}
	else if (bits == 64)
	{
		// const int64_t* data = (int64_t*)&r;
		int64_t data[N];
		store<int64_t>(data, r);

		for (uint32_t l = 0; l < (int)mipp::Lanes; l++)
		{
			for (auto i = 0; i < lane_size; i++)
				stream << std::setw(elmtWidth) << (data[l * lane_size +i] ? 1 : 0) << ((i < lane_size -1) ? ", " : "");
			stream << (((int)l < (int)mipp::Lanes -1) ? " | " : "");
		}
	}

	stream << "]";
}

// ---------------------------------------------------------------------------------------------------------- reduction

template <typename T>
using red_op = reg (*)(const reg, const reg);

template <typename T>
using Red_op = Reg<T> (*)(const Reg<T>, const Reg<T>);

template <typename T>
using ld_op = reg (*)(const T*);

template <typename T, red_op<T> OP>
struct _reduction
{
	static reg apply(const reg) {
		errorMessage<T>("_reduction::apply");
		exit(-1);
	}
};

template <typename T, Red_op<T> OP>
struct _Reduction
{
	static Reg<T> apply(const Reg<T> r) {
#ifndef MIPP_NO_INTRINSICS
		errorMessage<T>("_Reduction::apply");
		exit(-1);
#else
		return r;
#endif
	}
};

template <typename T, red_op<T> OP>
struct reduction
{
	static reg apply(const reg r)
	{
		return _reduction<T,OP>::apply(r);
	}

	static T sapply(const reg r)
	{
		auto red = reduction<T,OP>::apply(r);
#ifdef _MSC_VER
		return *((T*)&red);
#else
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
		return *((T*)&red);
#endif
	}

	template <ld_op<T> LD = mipp::load<T>>
	static T apply(const mipp::vector<T> &data)
	{
		return reduction<T,OP>::template apply<LD>(data.data(), data.size());
	}

	template <ld_op<T> LD = mipp::loadu<T>>
	static T apply(const std::vector<T> &data)
	{
		return reduction<T,OP>::template apply<LD>(data.data(), data.size());
	}

	template <ld_op<T> LD = mipp::loadu<T>>
	static T apply(const T *data, const uint32_t dataSize)
	{
		assert(dataSize > 0);
		assert(dataSize % mipp::nElReg<T>() == 0);

		auto rRed = LD(&data[0]);
		for (auto i = mipp::nElReg<T>(); i < dataSize; i += mipp::nElReg<T>())
			rRed = OP(rRed, LD(&data[i]));
		rRed = reduction<T,OP>::apply(rRed);

		T tRed[mipp::nElReg<T>()];
		mipp::store<T>(tRed, rRed);

		return tRed[0];
	}
};

template <typename T, Red_op<T> OP>
struct Reduction
{
	static Reg<T> apply(const Reg<T> r)
	{
		return _Reduction<T,OP>::apply(r);
	}

	static T sapply(const Reg<T> r)
	{
		auto red = Reduction<T,OP>::apply(r);
		return red[0];
	}

	template <ld_op<T> LD = mipp::load<T>>
	static T apply(const mipp::vector<T> &data)
	{
		return Reduction<T,OP>::template apply<LD>(data.data(), data.size());
	}

	template <ld_op<T> LD = mipp::loadu<T>>
	static T apply(const std::vector<T> &data)
	{
		return Reduction<T,OP>::template apply<LD>(data.data(), data.size());
	}

	template <ld_op<T> LD = mipp::loadu<T>>
	static T apply(const T *data, const uint32_t dataSize)
	{
		assert(dataSize > 0);
		assert(dataSize % mipp::nElReg<T>() == 0);

#ifndef MIPP_NO_INTRINSICS
		auto rRed = Reg<T>(LD(&data[0]));
#else
		auto rRed = Reg<T>(data[0]);
#endif
		for (auto i = mipp::nElReg<T>(); i < dataSize; i += mipp::nElReg<T>())
#ifndef MIPP_NO_INTRINSICS
			rRed = OP(rRed, Reg<T>(LD(&data[i])));
#else
			rRed = OP(rRed, Reg<T>(data[i]));
#endif
		rRed = Reduction<T,OP>::apply(rRed);

		T tRed[mipp::nElReg<T>()];
		rRed.store(tRed);

		return tRed[0];
	}
};

// ------------------------------------------------------------------------- special reduction functions implementation

template <typename T> inline T sum (const reg v) { return reduction<T,mipp::add<T>>::sapply(v); }
template <typename T> inline T hadd(const reg v) { return reduction<T,mipp::add<T>>::sapply(v); }
template <typename T> inline T hmul(const reg v) { return reduction<T,mipp::mul<T>>::sapply(v); }
template <typename T> inline T hmin(const reg v) { return reduction<T,mipp::min<T>>::sapply(v); }
template <typename T> inline T hmax(const reg v) { return reduction<T,mipp::max<T>>::sapply(v); }

// ------------------------------------------------------------------------------------------------- wrapper to objects
#include "mipp_object.hxx"

#ifndef MIPP_NO_INTRINSICS
// ------------------------------------------------------------------------------------------------------- ARM NEON-128
// --------------------------------------------------------------------------------------------------------------------
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include "mipp_impl_NEON.hxx"
// -------------------------------------------------------------------------------------------------------- X86 AVX-512
// --------------------------------------------------------------------------------------------------------------------
#elif defined(__MIC__) || defined(__KNCNI__) || defined(__AVX512__) || defined(__AVX512F__)
#include "mipp_impl_AVX512.hxx"
// -------------------------------------------------------------------------------------------------------- X86 AVX-256
// --------------------------------------------------------------------------------------------------------------------
#elif defined(__AVX__)
#include "mipp_impl_AVX.hxx"
// -------------------------------------------------------------------------------------------------------- X86 SSE-128
// --------------------------------------------------------------------------------------------------------------------
#elif defined(__SSE__)
#include "mipp_impl_SSE.hxx"
#endif
#endif

}

#endif /* MY_INTRINSICS_PLUS_PLUS_H_ */