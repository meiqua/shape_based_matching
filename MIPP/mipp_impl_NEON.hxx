#include "mipp.h"

// ------------------------------------------------------------------------------------------------------- ARM NEON-128
// --------------------------------------------------------------------------------------------------------------------
#if defined(__ARM_NEON__) || defined(__ARM_NEON)

	// ---------------------------------------------------------------------------------------------------------- loadu
#ifdef __aarch64__
	template <>
	inline reg loadu<double>(const double *mem_addr) {
		return (reg) vld1q_f64(mem_addr);
	}
#endif

	template <>
	inline reg loadu<float>(const float *mem_addr) {
		return vld1q_f32(mem_addr);
	}

#ifdef __aarch64__
		template <>
	inline reg loadu<int64_t>(const int64_t *mem_addr) {
		return (reg) vld1q_s64(mem_addr);
	}
#endif

	template <>
	inline reg loadu<int32_t>(const int32_t *mem_addr) {
		return (reg) vld1q_s32(mem_addr);
	}

	template <>
	inline reg loadu<int16_t>(const int16_t *mem_addr) {
		return (reg) vld1q_s16((int16_t*) mem_addr);
	}

	template <>
	inline reg loadu<int8_t>(const int8_t *mem_addr) {
		return (reg) vld1q_s8((int8_t*) mem_addr);
	}

	template <>
	inline reg loadu<uint8_t>(const uint8_t *mem_addr) {
		return (reg) vld1q_u8((uint8_t*) mem_addr);
	}

	// ----------------------------------------------------------------------------------------------------------- load
#ifdef MIPP_ALIGNED_LOADS
#ifdef __aarch64__
	template <>
	inline reg load<double>(const double *mem_addr) {
		return (reg) vld1q_f64(mem_addr);
	}
#endif

	template <>
	inline reg load<float>(const float *mem_addr) {
		return vld1q_f32(mem_addr);
	}

#ifdef __aarch64__
		template <>
	inline reg load<int64_t>(const int64_t *mem_addr) {
		return (reg) vld1q_s64(mem_addr);
	}
#endif

	template <>
	inline reg load<int32_t>(const int32_t *mem_addr) {
		return (reg) vld1q_s32(mem_addr);
	}

	template <>
	inline reg load<int16_t>(const int16_t *mem_addr) {
		return (reg) vld1q_s16((int16_t*) mem_addr);
	}

	template <>
	inline reg load<int8_t>(const int8_t *mem_addr) {
		return (reg) vld1q_s8((int8_t*) mem_addr);
	}

	template <>
	inline reg load<uint8_t>(const uint8_t *mem_addr) {
		return (reg) vld1q_u8((uint8_t*) mem_addr);
	}
#else
	template <>
	inline reg load<double>(const double *mem_addr) {
		return mipp::loadu<double>(mem_addr);
	}

	template <>
	inline reg load<float>(const float *mem_addr) {
		return mipp::loadu<float>(mem_addr);
	}

	template <>
	inline reg load<int64_t>(const int64_t *mem_addr) {
		return mipp::loadu<int64_t>(mem_addr);
	}

	template <>
	inline reg load<int32_t>(const int32_t *mem_addr) {
		return mipp::loadu<int32_t>(mem_addr);
	}

	template <>
	inline reg load<int16_t>(const int16_t *mem_addr) {
		return mipp::loadu<int16_t>(mem_addr);
	}

	template <>
	inline reg load<int8_t>(const int8_t *mem_addr) {
		return mipp::loadu<int8_t>(mem_addr);
	}

	template <>
	inline reg load<uint8_t>(const uint8_t *mem_addr) {
		return mipp::loadu<uint8_t>(mem_addr);
	}
#endif

	// --------------------------------------------------------------------------------------------------------- storeu
#ifdef __aarch64__
	template <>
	inline void storeu<double>(double *mem_addr, const reg v) {
		vst1q_f64(mem_addr, (float64x2_t)v);
	}
#endif

	template <>
	inline void storeu<float>(float *mem_addr, const reg v) {
		vst1q_f32(mem_addr, v);
	}

#ifdef __aarch64__
	template <>
	inline void storeu<int64_t>(int64_t *mem_addr, const reg v) {
		vst1q_s64(mem_addr, (int64x2_t)v);
	}
#endif

	template <>
	inline void storeu<int32_t>(int32_t *mem_addr, const reg v) {
		vst1q_s32(mem_addr, (int32x4_t)v);
	}

	template <>
	inline void storeu<int16_t>(int16_t *mem_addr, const reg v) {
		vst1q_s16((int16_t*) mem_addr, (int16x8_t) v);
	}

	template <>
	inline void storeu<int8_t>(int8_t *mem_addr, const reg v) {
		vst1q_s8((int8_t*) mem_addr, (int8x16_t) v);
	}

	template <>
	inline void storeu<uint8_t>(uint8_t *mem_addr, const reg v) {
		vst1q_u8((uint8_t*) mem_addr, (uint8x16_t) v);
	}
	// ---------------------------------------------------------------------------------------------------------- store
#ifdef MIPP_ALIGNED_LOADS
#ifdef __aarch64__
	template <>
	inline void store<double>(double *mem_addr, const reg v) {
		vst1q_f64(mem_addr, (float64x2_t)v);
	}
#endif

	template <>
	inline void store<float>(float *mem_addr, const reg v) {
		vst1q_f32(mem_addr, v);
	}

#ifdef __aarch64__
	template <>
	inline void store<int64_t>(int64_t *mem_addr, const reg v) {
		vst1q_s64(mem_addr, (int64x2_t)v);
	}
#endif

	template <>
	inline void store<int32_t>(int32_t *mem_addr, const reg v) {
		vst1q_s32(mem_addr, (int32x4_t)v);
	}

	template <>
	inline void store<int16_t>(int16_t *mem_addr, const reg v) {
		vst1q_s16((int16_t*) mem_addr, (int16x8_t) v);
	}

	template <>
	inline void store<int8_t>(int8_t *mem_addr, const reg v) {
		vst1q_s8((int8_t*) mem_addr, (int8x16_t) v);
	}

	template <>
	inline void store<uint8_t>(uint8_t *mem_addr, const reg v) {
		vst1q_u8((uint8_t*) mem_addr, (uint8x16_t) v);
	}
#else
	template <>
	inline void store<double>(double *mem_addr, const reg v) {
		mipp::storeu<double>(mem_addr, v);
	}

	template <>
	inline void store<float>(float *mem_addr, const reg v) {
		mipp::storeu<float>(mem_addr, v);
	}

	template <>
	inline void store<int64_t>(int64_t *mem_addr, const reg v) {
		mipp::storeu<int64_t>(mem_addr, v);
	}

	template <>
	inline void store<int32_t>(int32_t *mem_addr, const reg v) {
		mipp::storeu<int32_t>(mem_addr, v);
	}

	template <>
	inline void store<int16_t>(int16_t *mem_addr, const reg v) {
		mipp::storeu<int16_t>(mem_addr, v);
	}

	template <>
	inline void store<int8_t>(int8_t *mem_addr, const reg v) {
		mipp::storeu<int8_t>(mem_addr, v);
	}

	template <>
	inline void store<uint8_t>(uint8_t *mem_addr, const reg v) {
		mipp::storeu<uint8_t>(mem_addr, v);
	}
#endif

	// ------------------------------------------------------------------------------------------------------------ set
	template <>
	inline reg set<double>(const double vals[nElReg<double>()]) {
		return load<double>(vals);
	}

	template <>
	inline reg set<float>(const float vals[nElReg<float>()]) {
		return load<float>(vals);
	}

	template <>
	inline reg set<int64_t>(const int64_t vals[nElReg<int64_t>()]) {
		return load<int64_t>(vals);
	}

	template <>
	inline reg set<int32_t>(const int32_t vals[nElReg<int32_t>()]) {
		return load<int32_t>(vals);
	}

	template <>
	inline reg set<int16_t>(const int16_t vals[nElReg<int16_t>()]) {
		return load<int16_t>(vals);
	}

	template <>
	inline reg set<int8_t>(const int8_t vals[nElReg<int8_t>()]) {
		return load<int8_t>(vals);
	}

	// ----------------------------------------------------------------------------------------------------- set (mask)
#ifdef __aarch64__
	template <>
	inline msk set<2>(const bool vals[2]) {
		uint64_t v[2] = {vals[0] ? (uint64_t)0xFFFFFFFFFFFFFFFF : (uint64_t)0,
		                 vals[1] ? (uint64_t)0xFFFFFFFFFFFFFFFF : (uint64_t)0};
		return (msk) set<int64_t>((int64_t*)v);
	}
#endif

	template <>
	inline msk set<4>(const bool vals[4]) {
		uint32_t v[4] = {vals[0] ? (uint32_t)0xFFFFFFFF : (uint32_t)0, vals[1] ? (uint32_t)0xFFFFFFFF : (uint32_t)0,
		                 vals[2] ? (uint32_t)0xFFFFFFFF : (uint32_t)0, vals[3] ? (uint32_t)0xFFFFFFFF : (uint32_t)0};
		return (msk) set<int32_t>((int32_t*)v);
	}

	template <>
	inline msk set<8>(const bool vals[8]) {
		uint16_t v[8] = {vals[ 0] ? (uint16_t)0xFFFF : (uint16_t)0, vals[ 1] ? (uint16_t)0xFFFF : (uint16_t)0,
		                 vals[ 2] ? (uint16_t)0xFFFF : (uint16_t)0, vals[ 3] ? (uint16_t)0xFFFF : (uint16_t)0,
		                 vals[ 4] ? (uint16_t)0xFFFF : (uint16_t)0, vals[ 5] ? (uint16_t)0xFFFF : (uint16_t)0,
		                 vals[ 6] ? (uint16_t)0xFFFF : (uint16_t)0, vals[ 7] ? (uint16_t)0xFFFF : (uint16_t)0};
		return (msk) set<int16_t>((int16_t*)v);
	}

	template <>
	inline msk set<16>(const bool vals[16]) {
		uint8_t v[16] = {vals[ 0] ? (uint8_t)0xFF : (uint8_t)0, vals[ 1] ? (uint8_t)0xFF : (uint8_t)0,
		                 vals[ 2] ? (uint8_t)0xFF : (uint8_t)0, vals[ 3] ? (uint8_t)0xFF : (uint8_t)0,
		                 vals[ 4] ? (uint8_t)0xFF : (uint8_t)0, vals[ 5] ? (uint8_t)0xFF : (uint8_t)0,
		                 vals[ 6] ? (uint8_t)0xFF : (uint8_t)0, vals[ 7] ? (uint8_t)0xFF : (uint8_t)0,
		                 vals[ 8] ? (uint8_t)0xFF : (uint8_t)0, vals[ 9] ? (uint8_t)0xFF : (uint8_t)0,
		                 vals[10] ? (uint8_t)0xFF : (uint8_t)0, vals[11] ? (uint8_t)0xFF : (uint8_t)0,
		                 vals[12] ? (uint8_t)0xFF : (uint8_t)0, vals[13] ? (uint8_t)0xFF : (uint8_t)0,
		                 vals[14] ? (uint8_t)0xFF : (uint8_t)0, vals[15] ? (uint8_t)0xFF : (uint8_t)0};
		return (msk) set<int8_t>((int8_t*)v);
	}

	// ----------------------------------------------------------------------------------------------------------- set1
#ifdef __aarch64__
	template <>
	inline reg set1<double>(const double val) {
		return (reg) vdupq_n_f64(val);
	}
#endif

	template <>
	inline reg set1<float>(const float val) {
		return vdupq_n_f32(val);
	}

#ifdef __aarch64__
		template <>
	inline reg set1<int64_t>(const int64_t val) {
		return (reg) vdupq_n_s64(val);
	}
#endif

	template <>
	inline reg set1<int32_t>(const int32_t val) {
		return (reg) vdupq_n_s32(val);
	}

	template <>
	inline reg set1<int16_t>(const int16_t val) {
		return (reg) vdupq_n_s16(val);
	}

	template <>
	inline reg set1<int8_t>(const int8_t val) {
		return (reg) vdupq_n_s8(val);
	}

	template <>
	inline reg set1<uint8_t>(const uint8_t val) {
		return (reg) vdupq_n_u8(val);
	}
	// ---------------------------------------------------------------------------------------------------- set1 (mask)
#ifdef __aarch64__
		template <>
	inline msk set1<2>(const bool val) {
		return (msk) vdupq_n_u64(val ? 0xFFFFFFFFFFFFFFFF : 0);
	}
#endif

	template <>
	inline msk set1<4>(const bool val) {
		return (msk) vdupq_n_u32(val ? 0xFFFFFFFF : 0);
	}

	template <>
	inline msk set1<8>(const bool val) {
		return (msk) vdupq_n_u16(val ? 0xFFFF : 0);
	}

	template <>
	inline msk set1<16>(const bool val) {
		return (msk) vdupq_n_u8(val ? 0xFF : 0);
	}

	// ----------------------------------------------------------------------------------------------------------- set0
#ifdef __aarch64__
	template <>
	inline reg set0<double>() {
		return (reg) vdupq_n_f64(0.0);
	}
#endif

	template <>
	inline reg set0<float>() {
		return vdupq_n_f32(0.f);
	}

#ifdef __aarch64__
	template <>
	inline reg set0<int64_t>() {
		return (reg) vdupq_n_s64(0);
	}
#endif

	template <>
	inline reg set0<int32_t>() {
		return (reg) vdupq_n_s32(0);
	}

	template <>
	inline reg set0<int16_t>() {
		return (reg) vdupq_n_s16(0);
	}

	template <>
	inline reg set0<int8_t>() {
		return (reg) vdupq_n_s8(0);
	}

	// ---------------------------------------------------------------------------------------------------- set0 (mask)
#ifdef __aarch64__
	template <>
	inline msk set0<2>() {
		return (msk) vdupq_n_u64(0);
	}
#endif

	template <>
	inline msk set0<4>() {
		return (msk) vdupq_n_u32(0);
	}

	template <>
	inline msk set0<8>() {
		return (msk) vdupq_n_u16(0);
	}

	template <>
	inline msk set0<16>() {
		return (msk) vdupq_n_u8(0);
	}

	// ------------------------------------------------------------------------------------------------------------ low
#ifdef __aarch64__
	template <>
	inline reg_2 low<double>(const reg v) {
		return (reg_2) vget_low_f64((float64x2_t) v);
	}
#endif

	template <>
	inline reg_2 low<float>(const reg v) {
		return (reg_2) vget_low_f32((float32x4_t) v);
	}

#ifdef __aarch64__
	template <>
	inline reg_2 low<int64_t>(const reg v) {
		return (reg_2) vget_low_f64((float64x2_t) v);
	}
#endif

	template <>
	inline reg_2 low<int32_t>(const reg v) {
		return (reg_2) vget_low_f32((float32x4_t) v);
	}

	template <>
	inline reg_2 low<int16_t>(const reg v) {
		return (reg_2) vget_low_f32((float32x4_t) v);
	}

	template <>
	inline reg_2 low<int8_t>(const reg v) {
		return (reg_2) vget_low_f32((float32x4_t) v);
	}

	// ----------------------------------------------------------------------------------------------------------- high
#ifdef __aarch64__
	template <>
	inline reg_2 high<double>(const reg v) {
		return (reg_2) vget_high_f64((float64x2_t) v);
	}
#endif

	template <>
	inline reg_2 high<float>(const reg v) {
		return (reg_2) vget_high_f32((float32x4_t) v);
	}

#ifdef __aarch64__
	template <>
	inline reg_2 high<int64_t>(const reg v) {
		return (reg_2) vget_high_f64((float64x2_t) v);
	}
#endif

	template <>
	inline reg_2 high<int32_t>(const reg v) {
		return (reg_2) vget_high_f32((float32x4_t) v);
	}

	template <>
	inline reg_2 high<int16_t>(const reg v) {
		return (reg_2) vget_high_f32((float32x4_t) v);
	}

	template <>
	inline reg_2 high<int8_t>(const reg v) {
		return (reg_2) vget_high_f32((float32x4_t) v);
	}

	// ---------------------------------------------------------------------------------------------------------- cmask
#ifdef __aarch64__
	template <>
	inline reg cmask<double>(const uint32_t val[nElReg<double>()]) {
		int8_t val_bis[nElReg<int8_t>()] = {(int8_t)(val[0]*8 + 0), (int8_t)(val[0]*8 + 1),
		                                    (int8_t)(val[0]*8 + 2), (int8_t)(val[0]*8 + 3),
		                                    (int8_t)(val[0]*8 + 4), (int8_t)(val[0]*8 + 5),
		                                    (int8_t)(val[0]*8 + 6), (int8_t)(val[0]*8 + 7),
		                                    (int8_t)(val[1]*8 + 0), (int8_t)(val[1]*8 + 1),
		                                    (int8_t)(val[1]*8 + 2), (int8_t)(val[1]*8 + 3),
		                                    (int8_t)(val[1]*8 + 4), (int8_t)(val[1]*8 + 5),
		                                    (int8_t)(val[1]*8 + 6), (int8_t)(val[1]*8 + 7)};
		return mipp::set<int8_t>(val_bis);
	}
#endif

	template <>
	inline reg cmask<float>(const uint32_t val[nElReg<float>()]) {
		int8_t val_bis[nElReg<int8_t>()] = {(int8_t)(val[0]*4 + 0), (int8_t)(val[0]*4 + 1),
		                                    (int8_t)(val[0]*4 + 2), (int8_t)(val[0]*4 + 3),
		                                    (int8_t)(val[1]*4 + 0), (int8_t)(val[1]*4 + 1),
		                                    (int8_t)(val[1]*4 + 2), (int8_t)(val[1]*4 + 3),
		                                    (int8_t)(val[2]*4 + 0), (int8_t)(val[2]*4 + 1),
		                                    (int8_t)(val[2]*4 + 2), (int8_t)(val[2]*4 + 3),
		                                    (int8_t)(val[3]*4 + 0), (int8_t)(val[3]*4 + 1),
		                                    (int8_t)(val[3]*4 + 2), (int8_t)(val[3]*4 + 3)};
		return mipp::set<int8_t>(val_bis);
	}

#ifdef __aarch64__
	template <>
	inline reg cmask<int64_t>(const uint32_t val[nElReg<int64_t>()]) {
		int8_t val_bis[nElReg<int8_t>()] = {(int8_t)(val[0]*8 + 0), (int8_t)(val[0]*8 + 1),
		                                    (int8_t)(val[0]*8 + 2), (int8_t)(val[0]*8 + 3),
		                                    (int8_t)(val[0]*8 + 4), (int8_t)(val[0]*8 + 5),
		                                    (int8_t)(val[0]*8 + 6), (int8_t)(val[0]*8 + 7),
		                                    (int8_t)(val[1]*8 + 0), (int8_t)(val[1]*8 + 1),
		                                    (int8_t)(val[1]*8 + 2), (int8_t)(val[1]*8 + 3),
		                                    (int8_t)(val[1]*8 + 4), (int8_t)(val[1]*8 + 5),
		                                    (int8_t)(val[1]*8 + 6), (int8_t)(val[1]*8 + 7)};
		return mipp::set<int8_t>(val_bis);
	}
#endif

	template <>
	inline reg cmask<int32_t>(const uint32_t val[nElReg<int32_t>()]) {
		int8_t val_bis[nElReg<int8_t>()] = {(int8_t)(val[0]*4 + 0), (int8_t)(val[0]*4 + 1),
		                                    (int8_t)(val[0]*4 + 2), (int8_t)(val[0]*4 + 3),
		                                    (int8_t)(val[1]*4 + 0), (int8_t)(val[1]*4 + 1),
		                                    (int8_t)(val[1]*4 + 2), (int8_t)(val[1]*4 + 3),
		                                    (int8_t)(val[2]*4 + 0), (int8_t)(val[2]*4 + 1),
		                                    (int8_t)(val[2]*4 + 2), (int8_t)(val[2]*4 + 3),
		                                    (int8_t)(val[3]*4 + 0), (int8_t)(val[3]*4 + 1),
		                                    (int8_t)(val[3]*4 + 2), (int8_t)(val[3]*4 + 3)};
		return mipp::set<int8_t>(val_bis);
	}

	template <>
	inline reg cmask<int16_t>(const uint32_t val[nElReg<int16_t>()]) {
		int8_t val_bis[nElReg<int8_t>()] = {(int8_t)(val[0]*2 + 0), (int8_t)(val[0]*2 + 1),
		                                    (int8_t)(val[1]*2 + 0), (int8_t)(val[1]*2 + 1),
		                                    (int8_t)(val[2]*2 + 0), (int8_t)(val[2]*2 + 1),
		                                    (int8_t)(val[3]*2 + 0), (int8_t)(val[3]*2 + 1),
		                                    (int8_t)(val[4]*2 + 0), (int8_t)(val[4]*2 + 1),
		                                    (int8_t)(val[5]*2 + 0), (int8_t)(val[5]*2 + 1),
		                                    (int8_t)(val[6]*2 + 0), (int8_t)(val[6]*2 + 1),
		                                    (int8_t)(val[7]*2 + 0), (int8_t)(val[7]*2 + 1)};
		return mipp::set<int8_t>(val_bis);
	}

	template <>
	inline reg cmask<int8_t>(const uint32_t val[nElReg<int8_t>()]) {
		int8_t val_bis[nElReg<int8_t>()] = {(int8_t)val[ 0], (int8_t)val[ 1],
		                                    (int8_t)val[ 2], (int8_t)val[ 3],
		                                    (int8_t)val[ 4], (int8_t)val[ 5],
		                                    (int8_t)val[ 6], (int8_t)val[ 7],
		                                    (int8_t)val[ 8], (int8_t)val[ 9],
		                                    (int8_t)val[10], (int8_t)val[11],
		                                    (int8_t)val[12], (int8_t)val[13],
		                                    (int8_t)val[14], (int8_t)val[15]};
		return mipp::set<int8_t>(val_bis);
	}

	// --------------------------------------------------------------------------------------------------------- cmask2
#ifdef __aarch64__
	template <>
	inline reg cmask2<double>(const uint32_t val[nElReg<double>()/2]) {
		int8_t val_bis[nElReg<int8_t>()] = {(int8_t)(val[0]*8 + 0+0), (int8_t)(val[0]*8 + 1+0),
		                                    (int8_t)(val[0]*8 + 2+0), (int8_t)(val[0]*8 + 3+0),
		                                    (int8_t)(val[0]*8 + 4+0), (int8_t)(val[0]*8 + 5+0),
		                                    (int8_t)(val[0]*8 + 6+0), (int8_t)(val[0]*8 + 7+0),
		                                    (int8_t)(val[0]*8 + 0+8), (int8_t)(val[0]*8 + 1+8),
		                                    (int8_t)(val[0]*8 + 2+8), (int8_t)(val[0]*8 + 3+8),
		                                    (int8_t)(val[0]*8 + 4+8), (int8_t)(val[0]*8 + 5+8),
		                                    (int8_t)(val[0]*8 + 6+8), (int8_t)(val[0]*8 + 7+8)};
		return mipp::set<int8_t>(val_bis);
	}
#endif

	template <>
	inline reg cmask2<float>(const uint32_t val[nElReg<float>()/2]) {
		int8_t val_bis[nElReg<int8_t>()] = {(int8_t)(val[0]*4 + 0+0), (int8_t)(val[0]*4 + 1+0),
		                                    (int8_t)(val[0]*4 + 2+0), (int8_t)(val[0]*4 + 3+0),
		                                    (int8_t)(val[1]*4 + 0+0), (int8_t)(val[1]*4 + 1+0),
		                                    (int8_t)(val[1]*4 + 2+0), (int8_t)(val[1]*4 + 3+0),
		                                    (int8_t)(val[0]*4 + 0+8), (int8_t)(val[0]*4 + 1+8),
		                                    (int8_t)(val[0]*4 + 2+8), (int8_t)(val[0]*4 + 3+8),
		                                    (int8_t)(val[1]*4 + 0+8), (int8_t)(val[1]*4 + 1+8),
		                                    (int8_t)(val[1]*4 + 2+8), (int8_t)(val[1]*4 + 3+8)};
		return mipp::set<int8_t>(val_bis);
	}

#ifdef __aarch64__
	template <>
	inline reg cmask2<int64_t>(const uint32_t val[nElReg<int64_t>()/2]) {
		int8_t val_bis[nElReg<int8_t>()] = {(int8_t)(val[0]*8 + 0+0), (int8_t)(val[0]*8 + 1+0),
		                                    (int8_t)(val[0]*8 + 2+0), (int8_t)(val[0]*8 + 3+0),
		                                    (int8_t)(val[0]*8 + 4+0), (int8_t)(val[0]*8 + 5+0),
		                                    (int8_t)(val[0]*8 + 6+0), (int8_t)(val[0]*8 + 7+0),
		                                    (int8_t)(val[0]*8 + 0+8), (int8_t)(val[0]*8 + 1+8),
		                                    (int8_t)(val[0]*8 + 2+8), (int8_t)(val[0]*8 + 3+8),
		                                    (int8_t)(val[0]*8 + 4+8), (int8_t)(val[0]*8 + 5+8),
		                                    (int8_t)(val[0]*8 + 6+8), (int8_t)(val[0]*8 + 7+8)};
		return mipp::set<int8_t>(val_bis);
	}
#endif

	template <>
	inline reg cmask2<int32_t>(const uint32_t val[nElReg<int32_t>()/2]) {
		int8_t val_bis[nElReg<int8_t>()] = {(int8_t)(val[0]*4 + 0+0), (int8_t)(val[0]*4 + 1+0),
		                                    (int8_t)(val[0]*4 + 2+0), (int8_t)(val[0]*4 + 3+0),
		                                    (int8_t)(val[1]*4 + 0+0), (int8_t)(val[1]*4 + 1+0),
		                                    (int8_t)(val[1]*4 + 2+0), (int8_t)(val[1]*4 + 3+0),
		                                    (int8_t)(val[0]*4 + 0+8), (int8_t)(val[0]*4 + 1+8),
		                                    (int8_t)(val[0]*4 + 2+8), (int8_t)(val[0]*4 + 3+8),
		                                    (int8_t)(val[1]*4 + 0+8), (int8_t)(val[1]*4 + 1+8),
		                                    (int8_t)(val[1]*4 + 2+8), (int8_t)(val[1]*4 + 3+8)};
		return mipp::set<int8_t>(val_bis);
	}

	template <>
	inline reg cmask2<int16_t>(const uint32_t val[nElReg<int16_t>()/2]) {
		int8_t val_bis[nElReg<int8_t>()] = {(int8_t)(val[0]*2 + 0+0), (int8_t)(val[0]*2 + 1+0),
		                                    (int8_t)(val[1]*2 + 0+0), (int8_t)(val[1]*2 + 1+0),
		                                    (int8_t)(val[2]*2 + 0+0), (int8_t)(val[2]*2 + 1+0),
		                                    (int8_t)(val[3]*2 + 0+0), (int8_t)(val[3]*2 + 1+0),
		                                    (int8_t)(val[0]*2 + 0+8), (int8_t)(val[0]*2 + 1+8),
		                                    (int8_t)(val[1]*2 + 0+8), (int8_t)(val[1]*2 + 1+8),
		                                    (int8_t)(val[2]*2 + 0+8), (int8_t)(val[2]*2 + 1+8),
		                                    (int8_t)(val[3]*2 + 0+8), (int8_t)(val[3]*2 + 1+8)};
		return mipp::set<int8_t>(val_bis);
	}

	template <>
	inline reg cmask2<int8_t>(const uint32_t val[nElReg<int8_t>()/2]) {
		int8_t val_bis[nElReg<int8_t>()] = {(int8_t)(val[ 0]+0), (int8_t)(val[ 1]+0),
		                                    (int8_t)(val[ 2]+0), (int8_t)(val[ 3]+0),
		                                    (int8_t)(val[ 4]+0), (int8_t)(val[ 5]+0),
		                                    (int8_t)(val[ 6]+0), (int8_t)(val[ 7]+0),
		                                    (int8_t)(val[ 0]+8), (int8_t)(val[ 1]+8),
		                                    (int8_t)(val[ 2]+8), (int8_t)(val[ 3]+8),
		                                    (int8_t)(val[ 4]+8), (int8_t)(val[ 5]+8),
		                                    (int8_t)(val[ 6]+8), (int8_t)(val[ 7]+8)};
		return mipp::set<int8_t>(val_bis);
	}

	// --------------------------------------------------------------------------------------------------------- cmask4
	template <>
	inline reg cmask4<float>(const uint32_t val[nElReg<float>()/4]) {
		int8_t val_bis[nElReg<int8_t>()] = {(int8_t)(val[0]*4 + 0+ 0), (int8_t)(val[0]*4 + 1+ 0),
		                                    (int8_t)(val[0]*4 + 2+ 0), (int8_t)(val[0]*4 + 3+ 0),
		                                    (int8_t)(val[0]*4 + 0+ 4), (int8_t)(val[0]*4 + 1+ 4),
		                                    (int8_t)(val[0]*4 + 2+ 4), (int8_t)(val[0]*4 + 3+ 4),
		                                    (int8_t)(val[0]*4 + 0+ 8), (int8_t)(val[0]*4 + 1+ 8),
		                                    (int8_t)(val[0]*4 + 2+ 8), (int8_t)(val[0]*4 + 3+ 8),
		                                    (int8_t)(val[0]*4 + 0+12), (int8_t)(val[0]*4 + 1+12),
		                                    (int8_t)(val[0]*4 + 2+12), (int8_t)(val[0]*4 + 3+12)};
		return mipp::set<int8_t>(val_bis);
	}

	template <>
	inline reg cmask4<int32_t>(const uint32_t val[nElReg<int32_t>()/4]) {
		int8_t val_bis[nElReg<int8_t>()] = {(int8_t)(val[0]*4 + 0+ 0), (int8_t)(val[0]*4 + 1+ 0),
		                                    (int8_t)(val[0]*4 + 2+ 0), (int8_t)(val[0]*4 + 3+ 0),
		                                    (int8_t)(val[0]*4 + 0+ 4), (int8_t)(val[0]*4 + 1+ 4),
		                                    (int8_t)(val[0]*4 + 2+ 4), (int8_t)(val[0]*4 + 3+ 4),
		                                    (int8_t)(val[0]*4 + 0+ 8), (int8_t)(val[0]*4 + 1+ 8),
		                                    (int8_t)(val[0]*4 + 2+ 8), (int8_t)(val[0]*4 + 3+ 8),
		                                    (int8_t)(val[0]*4 + 0+12), (int8_t)(val[0]*4 + 1+12),
		                                    (int8_t)(val[0]*4 + 2+12), (int8_t)(val[0]*4 + 3+12)};
		return mipp::set<int8_t>(val_bis);
	}

	template <>
	inline reg cmask4<int16_t>(const uint32_t val[nElReg<int16_t>()/4]) {
		int8_t val_bis[nElReg<int8_t>()] = {(int8_t)(val[0]*2 + 0+ 0), (int8_t)(val[0]*2 + 1+ 0),
		                                    (int8_t)(val[1]*2 + 0+ 0), (int8_t)(val[1]*2 + 1+ 0),
		                                    (int8_t)(val[0]*2 + 0+ 4), (int8_t)(val[0]*2 + 1+ 4),
		                                    (int8_t)(val[1]*2 + 0+ 4), (int8_t)(val[1]*2 + 1+ 4),
		                                    (int8_t)(val[0]*2 + 0+ 8), (int8_t)(val[0]*2 + 1+ 8),
		                                    (int8_t)(val[1]*2 + 0+ 8), (int8_t)(val[1]*2 + 1+ 8),
		                                    (int8_t)(val[0]*2 + 0+12), (int8_t)(val[0]*2 + 1+12),
		                                    (int8_t)(val[1]*2 + 0+12), (int8_t)(val[1]*2 + 1+12)};
		return mipp::set<int8_t>(val_bis);
	}

	template <>
	inline reg cmask4<int8_t>(const uint32_t val[nElReg<int8_t>()/4]) {
		int8_t val_bis[nElReg<int8_t>()] = {(int8_t)(val[ 0]+ 0), (int8_t)(val[ 1]+ 0),
		                                    (int8_t)(val[ 2]+ 0), (int8_t)(val[ 3]+ 0),
		                                    (int8_t)(val[ 0]+ 4), (int8_t)(val[ 1]+ 4),
		                                    (int8_t)(val[ 2]+ 4), (int8_t)(val[ 3]+ 4),
		                                    (int8_t)(val[ 0]+ 8), (int8_t)(val[ 1]+ 8),
		                                    (int8_t)(val[ 2]+ 8), (int8_t)(val[ 3]+ 8),
		                                    (int8_t)(val[ 0]+12), (int8_t)(val[ 1]+12),
		                                    (int8_t)(val[ 2]+12), (int8_t)(val[ 3]+12)};
		return mipp::set<int8_t>(val_bis);
	}

	// ---------------------------------------------------------------------------------------------------------- shuff
	template <>
	inline reg shuff<double>(const reg v, const reg cm) {
		uint8x8x2_t v2 = {{vget_low_u8((uint8x16_t)v), vget_high_u8((uint8x16_t)v)}};
		uint8x8_t low  = vtbl2_u8(v2, vget_low_u8 ((uint8x16_t)cm));
		uint8x8_t high = vtbl2_u8(v2, vget_high_u8((uint8x16_t)cm));

		return (reg)vcombine_u8(low, high);
	}

	template <>
	inline reg shuff<float>(const reg v, const reg cm) {
		uint8x8x2_t v2 = {{vget_low_u8((uint8x16_t)v), vget_high_u8((uint8x16_t)v)}};
		uint8x8_t low  = vtbl2_u8(v2, vget_low_u8 ((uint8x16_t)cm));
		uint8x8_t high = vtbl2_u8(v2, vget_high_u8((uint8x16_t)cm));

		return (reg)vcombine_u8(low, high);
	}

	template <>
	inline reg shuff<int64_t>(const reg v, const reg cm) {
		uint8x8x2_t v2 = {{vget_low_u8((uint8x16_t)v), vget_high_u8((uint8x16_t)v)}};
		uint8x8_t low  = vtbl2_u8(v2, vget_low_u8 ((uint8x16_t)cm));
		uint8x8_t high = vtbl2_u8(v2, vget_high_u8((uint8x16_t)cm));

		return (reg)vcombine_u8(low, high);
	}

	template <>
	inline reg shuff<int32_t>(const reg v, const reg cm) {
		uint8x8x2_t v2 = {{vget_low_u8((uint8x16_t)v), vget_high_u8((uint8x16_t)v)}};
		uint8x8_t low  = vtbl2_u8(v2, vget_low_u8 ((uint8x16_t)cm));
		uint8x8_t high = vtbl2_u8(v2, vget_high_u8((uint8x16_t)cm));

		return (reg)vcombine_u8(low, high);
	}

	template <>
	inline reg shuff<int16_t>(const reg v, const reg cm) {
		uint8x8x2_t v2 = {{vget_low_u8((uint8x16_t)v), vget_high_u8((uint8x16_t)v)}};
		uint8x8_t low  = vtbl2_u8(v2, vget_low_u8 ((uint8x16_t)cm));
		uint8x8_t high = vtbl2_u8(v2, vget_high_u8((uint8x16_t)cm));

		return (reg)vcombine_u8(low, high);
	}

#define has_shuff_int8_t
	template <>
	inline reg shuff<int8_t>(const reg v, const reg cm) {
		uint8x8x2_t v2 = {{vget_low_u8((uint8x16_t)v), vget_high_u8((uint8x16_t)v)}};
		uint8x8_t low  = vtbl2_u8(v2, vget_low_u8 ((uint8x16_t)cm));
		uint8x8_t high = vtbl2_u8(v2, vget_high_u8((uint8x16_t)cm));

		return (reg)vcombine_u8(low, high);
	}

	template <>
	inline reg shuff<uint8_t>(const reg v, const reg cm) {
		return shuff<int8_t>(v, cm);
	}

	// --------------------------------------------------------------------------------------------------------- shuff2
	template <>
	inline reg shuff2<double>(const reg v, const reg cm) {
		return mipp::shuff<double>(v, cm);
	}

	template <>
	inline reg shuff2<float>(const reg v, const reg cm) {
		return mipp::shuff<float>(v, cm);
	}

	template <>
	inline reg shuff2<int64_t>(const reg v, const reg cm) {
		return mipp::shuff<int64_t>(v, cm);
	}

	template <>
	inline reg shuff2<int32_t>(const reg v, const reg cm) {
		return mipp::shuff<int32_t>(v, cm);
	}

	template <>
	inline reg shuff2<int16_t>(const reg v, const reg cm) {
		return mipp::shuff<int16_t>(v, cm);
	}

	template <>
	inline reg shuff2<int8_t>(const reg v, const reg cm) {
		return mipp::shuff<int8_t>(v, cm);
	}

	// --------------------------------------------------------------------------------------------------------- shuff4
	template <>
	inline reg shuff4<float>(const reg v, const reg cm) {
		return mipp::shuff<float>(v, cm);
	}

	template <>
	inline reg shuff4<int32_t>(const reg v, const reg cm) {
		return mipp::shuff<int32_t>(v, cm);
	}

	template <>
	inline reg shuff4<int16_t>(const reg v, const reg cm) {
		return mipp::shuff<int16_t>(v, cm);
	}

	template <>
	inline reg shuff4<int8_t>(const reg v, const reg cm) {
		return mipp::shuff<int8_t>(v, cm);
	}

	// --------------------------------------------------------------------------------------------------- interleavelo
	template <>
	inline reg interleavelo<double>(const reg v1, const reg v2) {
		// v1  = [a0, b0], v2 = [a1, b1]
		// res = [a0, a1]
		return (reg)vcombine_u64(vget_low_u64((uint64x2_t)v1), vget_low_u64((uint64x2_t)v2));
	}

	template <>
	inline reg interleavelo<float>(const reg v1, const reg v2) {
		// v1  = [a0, b0, c0, d0], v2 = [a1, b1, c1, d1]
		// res = [a0, a1, b0, b1]
		uint32x2x2_t res = vzip_u32(vget_low_u32((uint32x4_t)v1), vget_low_u32((uint32x4_t)v2));
		return (reg)vcombine_u32(res.val[0], res.val[1]);
	}

	template <>
	inline reg interleavelo<int64_t>(const reg v1, const reg v2) {
		// v1  = [a0, b0], v2 = [a1, b1]
		// res = [a0, a1]
		return (reg)vcombine_u64(vget_low_u64((uint64x2_t)v1), vget_low_u64((uint64x2_t)v2));
	}

	template <>
	inline reg interleavelo<int32_t>(const reg v1, const reg v2) {
		// v1  = [a0, b0, c0, d0], v2 = [a1, b1, c1, d1]
		// res = [a0, a1, b0, b1]
		uint32x2x2_t res = vzip_u32(vget_low_u32((uint32x4_t)v1), vget_low_u32((uint32x4_t)v2));
		return (reg)vcombine_u32(res.val[0], res.val[1]);
	}

	template <>
	inline reg interleavelo<int16_t>(const reg v1, const reg v2) {
		uint16x4x2_t res = vzip_u16(vget_low_u16((uint16x8_t)v1), vget_low_u16((uint16x8_t)v2));
		return (reg)vcombine_u16(res.val[0], res.val[1]);
	}

	template <>
	inline reg interleavelo<int8_t>(const reg v1, const reg v2) {
		uint8x8x2_t res = vzip_u8(vget_low_u8((uint8x16_t)v1), vget_low_u8((uint8x16_t)v2));
		return (reg)vcombine_u8(res.val[0], res.val[1]);
	}

	template <>
	inline reg interleavelo<uint8_t>(const reg v1, const reg v2) {
		return interleavelo<int8_t>(v1, v2);
	}

	// --------------------------------------------------------------------------------------------------- interleavehi
	template <>
	inline reg interleavehi<double>(const reg v1, const reg v2) {
		// v1  = [a0, b0], v2 = [a1, b1]
		// res = [b0, b1]
		return (reg)vcombine_u64(vget_high_u64((uint64x2_t)v1), vget_high_u64((uint64x2_t)v2));
	}

	template <>
	inline reg interleavehi<float>(const reg v1, const reg v2) {
		// v1  = [a0, b0, c0, d0], v2 = [a1, b1, c1, d1]
		// res = [c0, c1, d0, d1]
		uint32x2x2_t res = vzip_u32(vget_high_u32((uint32x4_t)v1), vget_high_u32((uint32x4_t)v2));
		return (reg)vcombine_u32(res.val[0], res.val[1]);
	}

	template <>
	inline reg interleavehi<int64_t>(const reg v1, const reg v2) {
		// v1  = [a0, b0], v2 = [a1, b1]
		// res = [b0, b1]
		return (reg)vcombine_u64(vget_high_u64((uint64x2_t)v1), vget_high_u64((uint64x2_t)v2));
	}

	template <>
	inline reg interleavehi<int32_t>(const reg v1, const reg v2) {
		// v1  = [a0, b0, c0, d0], v2 = [a1, b1, c1, d1]
		// res = [c0, c1, d0, d1]
		uint32x2x2_t res = vzip_u32(vget_high_u32((uint32x4_t)v1), vget_high_u32((uint32x4_t)v2));
		return (reg)vcombine_u32(res.val[0], res.val[1]);
	}

	template <>
	inline reg interleavehi<int16_t>(const reg v1, const reg v2) {
		uint16x4x2_t res = vzip_u16(vget_high_u16((uint16x8_t)v1), vget_high_u16((uint16x8_t)v2));
		return (reg)vcombine_u16(res.val[0], res.val[1]);
	}

	template <>
	inline reg interleavehi<int8_t>(const reg v1, const reg v2) {
		uint8x8x2_t res = vzip_u8(vget_high_u8((uint8x16_t)v1), vget_high_u8((uint8x16_t)v2));
		return (reg)vcombine_u8(res.val[0], res.val[1]);
	}

	// -------------------------------------------------------------------------------------------------- interleavelo2
	template <>
	inline reg interleavelo2<float>(const reg v1, const reg v2) {
		uint32x2x2_t res1 = vzip_u32(vget_low_u32 ((uint32x4_t)v1), vget_low_u32 ((uint32x4_t)v2));
		uint32x2x2_t res2 = vzip_u32(vget_high_u32((uint32x4_t)v1), vget_high_u32((uint32x4_t)v2));

		return (reg) vcombine_u32(res1.val[0], res2.val[0]);
	}

	template <>
	inline reg interleavelo2<int32_t>(const reg v1, const reg v2) {
		uint32x2x2_t res1 = vzip_u32(vget_low_u32 ((uint32x4_t)v1), vget_low_u32 ((uint32x4_t)v2));
		uint32x2x2_t res2 = vzip_u32(vget_high_u32((uint32x4_t)v1), vget_high_u32((uint32x4_t)v2));

		return (reg)vcombine_u32(res1.val[0], res2.val[0]);
	}

	template <>
	inline reg interleavelo2<int16_t>(const reg v1, const reg v2) {
		uint16x4x2_t res1 = vzip_u16(vget_low_u16 ((uint16x8_t)v1), vget_low_u16 ((uint16x8_t)v2));
		uint16x4x2_t res2 = vzip_u16(vget_high_u16((uint16x8_t)v1), vget_high_u16((uint16x8_t)v2));

		return (reg)vcombine_u16(res1.val[0], res2.val[0]);
	}

	template <>
	inline reg interleavelo2<int8_t>(const reg v1, const reg v2) {
		uint8x8x2_t res1 = vzip_u8(vget_low_u8 ((uint8x16_t)v1), vget_low_u8 ((uint8x16_t)v2));
		uint8x8x2_t res2 = vzip_u8(vget_high_u8((uint8x16_t)v1), vget_high_u8((uint8x16_t)v2));

		return (reg)vcombine_u8(res1.val[0], res2.val[0]);
	}

	// -------------------------------------------------------------------------------------------------- interleavehi2
	template <>
	inline reg interleavehi2<float>(const reg v1, const reg v2) {
		uint32x2x2_t res1 = vzip_u32(vget_low_u32 ((uint32x4_t)v1), vget_low_u32 ((uint32x4_t)v2));
		uint32x2x2_t res2 = vzip_u32(vget_high_u32((uint32x4_t)v1), vget_high_u32((uint32x4_t)v2));

		return (reg) vcombine_u32(res1.val[1], res2.val[1]);
	}

	template <>
	inline reg interleavehi2<int32_t>(const reg v1, const reg v2) {
		uint32x2x2_t res1 = vzip_u32(vget_low_u32 ((uint32x4_t)v1), vget_low_u32 ((uint32x4_t)v2));
		uint32x2x2_t res2 = vzip_u32(vget_high_u32((uint32x4_t)v1), vget_high_u32((uint32x4_t)v2));

		return (reg)vcombine_u32(res1.val[1], res2.val[1]);
	}

	template <>
	inline reg interleavehi2<int16_t>(const reg v1, const reg v2) {
		uint16x4x2_t res1 = vzip_u16(vget_low_u16 ((uint16x8_t)v1), vget_low_u16 ((uint16x8_t)v2));
		uint16x4x2_t res2 = vzip_u16(vget_high_u16((uint16x8_t)v1), vget_high_u16((uint16x8_t)v2));

		return (reg)vcombine_u16(res1.val[1], res2.val[1]);
	}

	template <>
	inline reg interleavehi2<int8_t>(const reg v1, const reg v2) {
		uint8x8x2_t res1 = vzip_u8(vget_low_u8 ((uint8x16_t)v1), vget_low_u8 ((uint8x16_t)v2));
		uint8x8x2_t res2 = vzip_u8(vget_high_u8((uint8x16_t)v1), vget_high_u8((uint8x16_t)v2));

		return (reg)vcombine_u8(res1.val[1], res2.val[1]);
	}


	// ----------------------------------------------------------------------------------------------------- interleave
	template <>
	inline regx2 interleave<double>(const reg v1, const reg v2) {
		// v1         = [a0, b0], v2         = [a1, b1]
		// res.val[0] = [a0, a1], res.val[1] = [b0, b1]
		regx2 res = {{(reg)vcombine_u64(vget_low_u64 ((uint64x2_t)v1), vget_low_u64 ((uint64x2_t)v2)),
		              (reg)vcombine_u64(vget_high_u64((uint64x2_t)v1), vget_high_u64((uint64x2_t)v2))}};
		return res;
	}

	template <>
	inline regx2 interleave<float>(const reg v1, const reg v2) {
		// v1         = [a0, b0, c0, d0], v2         = [a1, b1, c1, d1]
		// res.val[0] = [a0, a1, b0, b1], res.val[1] = [c0, c1, d0, d1]
		uint32x2x2_t res0 = vzip_u32(vget_low_u32 ((uint32x4_t)v1), vget_low_u32 ((uint32x4_t)v2));
		uint32x2x2_t res1 = vzip_u32(vget_high_u32((uint32x4_t)v1), vget_high_u32((uint32x4_t)v2));

		regx2 res = {{(reg)vcombine_u32(res0.val[0], res0.val[1]),
		              (reg)vcombine_u32(res1.val[0], res1.val[1])}};

		return res;
	}

	template <>
	inline regx2 interleave<int64_t>(const reg v1, const reg v2) {
		// v1         = [a0, b0], v2         = [a1, b1]
		// res.val[0] = [a0, a1], res.val[1] = [b0, b1]
		regx2 res = {{(reg)vcombine_u64(vget_low_u64 ((uint64x2_t)v1), vget_low_u64 ((uint64x2_t)v2)),
		              (reg)vcombine_u64(vget_high_u64((uint64x2_t)v1), vget_high_u64((uint64x2_t)v2))}};
		return res;
	}

	template <>
	inline regx2 interleave<int32_t>(const reg v1, const reg v2) {
		// v1  = [a0, b0, c0, d0], v2 = [a1, b1, c1, d1]
		// res = [a0, a1, b0, b1]
		uint32x2x2_t res0 = vzip_u32(vget_low_u32 ((uint32x4_t)v1), vget_low_u32 ((uint32x4_t)v2));
		uint32x2x2_t res1 = vzip_u32(vget_high_u32((uint32x4_t)v1), vget_high_u32((uint32x4_t)v2));

		regx2 res = {{(reg)vcombine_u32(res0.val[0], res0.val[1]),
		              (reg)vcombine_u32(res1.val[0], res1.val[1])}};

		return res;
	}

	template <>
	inline regx2 interleave<int16_t>(const reg v1, const reg v2) {
		uint16x4x2_t res0 = vzip_u16(vget_low_u16 ((uint16x8_t)v1), vget_low_u16 ((uint16x8_t)v2));
		uint16x4x2_t res1 = vzip_u16(vget_high_u16((uint16x8_t)v1), vget_high_u16((uint16x8_t)v2));

		regx2 res = {{(reg)vcombine_u16(res0.val[0], res0.val[1]),
		              (reg)vcombine_u16(res1.val[0], res1.val[1])}};

		return res;
	}

	template <>
	inline regx2 interleave<int8_t>(const reg v1, const reg v2) {
		uint8x8x2_t res0 = vzip_u8(vget_low_u8 ((uint8x16_t)v1), vget_low_u8 ((uint8x16_t)v2));
		uint8x8x2_t res1 = vzip_u8(vget_high_u8((uint8x16_t)v1), vget_high_u8((uint8x16_t)v2));

		regx2 res = {{(reg)vcombine_u8(res0.val[0], res0.val[1]),
		              (reg)vcombine_u8(res1.val[0], res1.val[1])}};

		return res;
	}

	// ---------------------------------------------------------------------------------------------------- interleave2
	template <>
	inline regx2 interleave2<float>(const reg v1, const reg v2) {
		uint32x2x2_t res1 = vzip_u32(vget_low_u32 ((uint32x4_t)v1), vget_low_u32 ((uint32x4_t)v2));
		uint32x2x2_t res2 = vzip_u32(vget_high_u32((uint32x4_t)v1), vget_high_u32((uint32x4_t)v2));

		regx2 res = {{(reg) vcombine_u32(res1.val[0], res2.val[0]),
		              (reg) vcombine_u32(res1.val[1], res2.val[1])}};

		return res;
	}

	template <>
	inline regx2 interleave2<int32_t>(const reg v1, const reg v2) {
		uint32x2x2_t res1 = vzip_u32(vget_low_u32 ((uint32x4_t)v1), vget_low_u32 ((uint32x4_t)v2));
		uint32x2x2_t res2 = vzip_u32(vget_high_u32((uint32x4_t)v1), vget_high_u32((uint32x4_t)v2));

		regx2 res = {{(reg) vcombine_u32(res1.val[0], res2.val[0]),
		              (reg) vcombine_u32(res1.val[1], res2.val[1])}};

		return res;
	}

	template <>
	inline regx2 interleave2<int16_t>(const reg v1, const reg v2) {
		uint16x4x2_t res1 = vzip_u16(vget_low_u16 ((uint16x8_t)v1), vget_low_u16 ((uint16x8_t)v2));
		uint16x4x2_t res2 = vzip_u16(vget_high_u16((uint16x8_t)v1), vget_high_u16((uint16x8_t)v2));

		regx2 res = {{(reg) vcombine_u16(res1.val[0], res2.val[0]),
		              (reg) vcombine_u16(res1.val[1], res2.val[1])}};

		return res;
	}

	template <>
	inline regx2 interleave2<int8_t>(const reg v1, const reg v2) {
		uint8x8x2_t res1 = vzip_u8(vget_low_u8 ((uint8x16_t)v1), vget_low_u8 ((uint8x16_t)v2));
		uint8x8x2_t res2 = vzip_u8(vget_high_u8((uint8x16_t)v1), vget_high_u8((uint8x16_t)v2));

		regx2 res = {{(reg) vcombine_u8(res1.val[0], res2.val[0]),
		              (reg) vcombine_u8(res1.val[1], res2.val[1])}};

		return res;
	}

	// ----------------------------------------------------------------------------------------------------------- andb
#ifdef __aarch64__
	template <>
	inline reg andb<double>(const reg v1, const reg v2) {
		return (reg) vandq_u64((uint64x2_t) v1, (uint64x2_t) v2);
	}
#endif

	template <>
	inline reg andb<float>(const reg v1, const reg v2) {
		return (reg) vandq_u32((uint32x4_t) v1, (uint32x4_t) v2);
	}

#ifdef __aarch64__
	template <>
	inline reg andb<int64_t>(const reg v1, const reg v2) {
		return (reg) vandq_u64((uint64x2_t) v1, (uint64x2_t) v2);
	}
#endif

	template <>
	inline reg andb<int32_t>(const reg v1, const reg v2) {
		return (reg) vandq_u32((uint32x4_t) v1, (uint32x4_t) v2);
	}

	template <>
	inline reg andb<int16_t>(const reg v1, const reg v2) {
		return (reg) vandq_u16((uint16x8_t) v1, (uint16x8_t) v2);
	}

	template <>
	inline reg andb<int8_t>(const reg v1, const reg v2) {
		return (reg) vandq_u8((uint8x16_t) v1, (uint8x16_t) v2);
	}

	// ---------------------------------------------------------------------------------------------------- andb (mask)
#ifdef __aarch64__
	template <>
	inline msk andb<2>(const msk v1, const msk v2) {
		return (msk) vandq_u64((uint64x2_t) v1, (uint64x2_t) v2);
	}
#endif

	template <>
	inline msk andb<4>(const msk v1, const msk v2) {
		return (msk) vandq_u32((uint32x4_t) v1, (uint32x4_t) v2);
	}

	template <>
	inline msk andb<8>(const msk v1, const msk v2) {
		return (msk) vandq_u16((uint16x8_t) v1, (uint16x8_t) v2);
	}

	template <>
	inline msk andb<16>(const msk v1, const msk v2) {
		return (msk) vandq_u8((uint8x16_t) v1, (uint8x16_t) v2);
	}

	// ----------------------------------------------------------------------------------------------------------- notb
#ifdef __aarch64__
	template <>
	inline reg notb<double>(const reg v) {
		return (reg) vmvnq_u32((uint32x4_t) v);
	}
#endif

	template <>
	inline reg notb<float>(const reg v) {
		return (reg) vmvnq_u32((uint32x4_t) v);
	}

#ifdef __aarch64__
	template <>
	inline reg notb<int64_t>(const reg v) {
		return (reg) vmvnq_u32((uint32x4_t) v);
	}
#endif

	template <>
	inline reg notb<int32_t>(const reg v) {
		return (reg) vmvnq_u32((uint32x4_t) v);
	}

	template <>
	inline reg notb<int16_t>(const reg v) {
		return (reg) vmvnq_u16((uint16x8_t) v);
	}

	template <>
	inline reg notb<int8_t>(const reg v) {
		return (reg) vmvnq_u8((uint8x16_t) v);
	}

	// ---------------------------------------------------------------------------------------------------- notb (mask)
#ifdef __aarch64__
	template <>
	inline msk notb<2>(const msk v) {
		return (msk) vmvnq_u32((uint32x4_t) v);
	}
#endif

	template <>
	inline msk notb<4>(const msk v) {
		return (msk) vmvnq_u32((uint32x4_t) v);
	}

	template <>
	inline msk notb<8>(const msk v) {
		return (msk) vmvnq_u16((uint16x8_t) v);
	}

	template <>
	inline msk notb<16>(const msk v) {
		return (msk) vmvnq_u8((uint8x16_t) v);
	}

	// ---------------------------------------------------------------------------------------------------------- andnb
#ifdef __aarch64__
	template <>
	inline reg andnb<double>(const reg v1, const reg v2) {
		return (reg) vandq_u32(vmvnq_u32((uint32x4_t) v1), (uint32x4_t) v2);
	}
#endif

	template <>
	inline reg andnb<float>(const reg v1, const reg v2) {
		return (reg) vandq_u32(vmvnq_u32((uint32x4_t) v1), (uint32x4_t) v2);
	}

#ifdef __aarch64__
	template <>
	inline reg andnb<int64_t>(const reg v1, const reg v2) {
		return (reg) vandq_u32(vmvnq_u32((uint32x4_t) v1), (uint32x4_t) v2);
	}
#endif

	template <>
	inline reg andnb<int32_t>(const reg v1, const reg v2) {
		return (reg) vandq_u32(vmvnq_u32((uint32x4_t) v1), (uint32x4_t) v2);
	}

	template <>
	inline reg andnb<int16_t>(const reg v1, const reg v2) {
		return (reg) vandq_u16(vmvnq_u16((uint16x8_t) v1), (uint16x8_t) v2);
	}

	template <>
	inline reg andnb<int8_t>(const reg v1, const reg v2) {
		return (reg) vandq_u8(vmvnq_u8((uint8x16_t) v1), (uint8x16_t) v2);
	}

	// --------------------------------------------------------------------------------------------------- andnb (mask)
#ifdef __aarch64__
	template <>
	inline msk andnb<2>(const msk v1, const msk v2) {
		return (msk) vandq_u32(vmvnq_u32((uint32x4_t) v1), (uint32x4_t) v2);
	}
#endif

	template <>
	inline msk andnb<4>(const msk v1, const msk v2) {
		return (msk) vandq_u32(vmvnq_u32((uint32x4_t) v1), (uint32x4_t) v2);
	}

	template <>
	inline msk andnb<8>(const msk v1, const msk v2) {
		return (msk) vandq_u16(vmvnq_u16((uint16x8_t) v1), (uint16x8_t) v2);
	}

	template <>
	inline msk andnb<16>(const msk v1, const msk v2) {
		return (msk) vandq_u8(vmvnq_u8((uint8x16_t) v1), (uint8x16_t) v2);
	}

	// ------------------------------------------------------------------------------------------------------------ orb
#ifdef __aarch64__
	template <>
	inline reg orb<double>(const reg v1, const reg v2) {
		return (reg) vorrq_u64((uint64x2_t) v1, (uint64x2_t) v2);
	}
#endif

	template <>
	inline reg orb<float>(const reg v1, const reg v2) {
		return (reg) vorrq_u32((uint32x4_t) v1, (uint32x4_t) v2);
	}

#ifdef __aarch64__
	template <>
	inline reg orb<int64_t>(const reg v1, const reg v2) {
		return (reg) vorrq_u64((uint64x2_t) v1, (uint64x2_t) v2);
	}
#endif

	template <>
	inline reg orb<int32_t>(const reg v1, const reg v2) {
		return (reg) vorrq_u32((uint32x4_t) v1, (uint32x4_t) v2);
	}

	template <>
	inline reg orb<int16_t>(const reg v1, const reg v2) {
		return (reg) vorrq_u16((uint16x8_t) v1, (uint16x8_t) v2);
	}

	template <>
	inline reg orb<int8_t>(const reg v1, const reg v2) {
		return (reg) vorrq_u8((uint8x16_t) v1, (uint8x16_t) v2);
	}

	template <>
	inline reg orb<uint8_t>(const reg v1, const reg v2) {
		return orb<int8_t>(v1, v2);
	}

	// ----------------------------------------------------------------------------------------------------- orb (mask)
#ifdef __aarch64__
	template <>
	inline msk orb<2>(const msk v1, const msk v2) {
		return (msk) vorrq_u64((uint64x2_t) v1, (uint64x2_t) v2);
	}
#endif

	template <>
	inline msk orb<4>(const msk v1, const msk v2) {
		return (msk) vorrq_u32((uint32x4_t) v1, (uint32x4_t) v2);
	}

	template <>
	inline msk orb<8>(const msk v1, const msk v2) {
		return (msk) vorrq_u16((uint16x8_t) v1, (uint16x8_t) v2);
	}

	template <>
	inline msk orb<16>(const msk v1, const msk v2) {
		return (msk) vorrq_u8((uint8x16_t) v1, (uint8x16_t) v2);
	}

	// ----------------------------------------------------------------------------------------------------------- xorb
#ifdef __aarch64__
	template <>
	inline reg xorb<double>(const reg v1, const reg v2) {
		return (reg) veorq_u64((uint64x2_t) v1, (uint64x2_t) v2);
	}
#endif

	template <>
	inline reg xorb<float>(const reg v1, const reg v2) {
		return (reg) veorq_u32((uint32x4_t) v1, (uint32x4_t) v2);
	}

#ifdef __aarch64__
	template <>
	inline reg xorb<int64_t>(const reg v1, const reg v2) {
		return (reg) veorq_u64((uint64x2_t) v1, (uint64x2_t) v2);
	}
#endif

	template <>
	inline reg xorb<int32_t>(const reg v1, const reg v2) {
		return (reg) veorq_u32((uint32x4_t) v1, (uint32x4_t) v2);
	}

	template <>
	inline reg xorb<int16_t>(const reg v1, const reg v2) {
		return (reg) veorq_u16((uint16x8_t) v1, (uint16x8_t) v2);
	}

	template <>
	inline reg xorb<int8_t>(const reg v1, const reg v2) {
		return (reg) veorq_u8((uint8x16_t) v1, (uint8x16_t) v2);
	}

	// ---------------------------------------------------------------------------------------------------- xorb (mask)
#ifdef __aarch64__
	template <>
	inline msk xorb<2>(const msk v1, const msk v2) {
		return (msk) veorq_u64((uint64x2_t) v1, (uint64x2_t) v2);
	}
#endif

	template <>
	inline msk xorb<4>(const msk v1, const msk v2) {
		return (msk) veorq_u32((uint32x4_t) v1, (uint32x4_t) v2);
	}

	template <>
	inline msk xorb<8>(const msk v1, const msk v2) {
		return (msk) veorq_u16((uint16x8_t) v1, (uint16x8_t) v2);
	}

	template <>
	inline msk xorb<16>(const msk v1, const msk v2) {
		return (msk) veorq_u8((uint8x16_t) v1, (uint8x16_t) v2);
	}

	// --------------------------------------------------------------------------------------------------------- lshift
#ifdef __aarch64__
	template <>
	inline reg lshift<int64_t>(const reg v1, const uint32_t n) {
		return (reg) vshlq_u64((uint64x2_t) v1, (int64x2_t)mipp::set1<int64_t>(n));
	}
#endif

	template <>
	inline reg lshift<int32_t>(const reg v1, const uint32_t n) {
		return (reg) vshlq_u32((uint32x4_t) v1, (int32x4_t)mipp::set1<int32_t>(n));
	}

	template <>
	inline reg lshift<int16_t>(const reg v1, const uint32_t n) {
		return (reg) vshlq_u16((uint16x8_t) v1, (int16x8_t)mipp::set1<int16_t>((int16_t) n));
	}

	template <>
	inline reg lshift<int8_t>(const reg v1, const uint32_t n) {
		return (reg) vshlq_u8((uint8x16_t) v1, (int8x16_t)mipp::set1<int8_t>((int8_t) n));
	}

	// -------------------------------------------------------------------------------------------------- lshift (mask)
#ifndef __clang__
#ifdef __aarch64__
	template <>
	inline msk lshift<2>(const msk v1, const uint32_t n) {
		const auto s = n * 8;
		     if (s <=  0) return v1;
		else if (s >  15) return set0<2>();
		else              return (msk)vextq_s8(vdupq_n_s8(0), (int8x16_t)v1, 16 - s);
	}
#endif

	template <>
	inline msk lshift<4>(const msk v1, const uint32_t n) {
		const auto s = n * 4;
		     if (s <=  0) return v1;
		else if (s >  15) return set0<4>();
		else              return (msk)vextq_s8(vdupq_n_s8(0), (int8x16_t)v1, 16 - s);
	}

	template <>
	inline msk lshift<8>(const msk v1, const uint32_t n) {
		const auto s = n * 2;
		     if (s <=  0) return v1;
		else if (s >  15) return set0<8>();
		else              return (msk)vextq_s8(vdupq_n_s8(0), (int8x16_t)v1, 16 - s);
	}

	template <>
	inline msk lshift<16>(const msk v1, const uint32_t n) {
		const auto s = n;
		     if (s <=  0) return v1;
		else if (s >  15) return set0<16>();
		else              return (msk)vextq_s8(vdupq_n_s8(0), (int8x16_t)v1, 16 - s);
	}
#endif

	// --------------------------------------------------------------------------------------------------------- rshift
#ifdef __aarch64__
	template <>
	inline reg rshift<int64_t>(const reg v1, const uint32_t n) {
		return (reg) vshlq_u64((uint64x2_t) v1, (int64x2_t)mipp::set1<int64_t>(-n));
	}
#endif

	template <>
	inline reg rshift<int32_t>(const reg v1, const uint32_t n) {
		return (reg) vshlq_u32((uint32x4_t) v1, (int32x4_t)mipp::set1<int32_t>(-n));
	}

	template <>
	inline reg rshift<int16_t>(const reg v1, const uint32_t n) {
		return (reg) vshlq_u16((uint16x8_t) v1, (int16x8_t)mipp::set1<int16_t>((int16_t)-n));
	}

	template <>
	inline reg rshift<int8_t>(const reg v1, const uint32_t n) {
		return (reg) vshlq_u8((uint8x16_t) v1, (int8x16_t)mipp::set1<int8_t>((int8_t)-n));
	}

	// -------------------------------------------------------------------------------------------------- rshift (mask)
#ifndef __clang__
#ifdef __aarch64__
	template <>
	inline msk rshift<2>(const msk v1, const uint32_t n) {
		const auto s = n * 8;
		     if (s <= 0) return v1;
		else if (s > 15) return set0<4>();
		else             return (msk)vextq_s8((int8x16_t)v1, vdupq_n_s8(0), s);
	}
#endif

	template <>
	inline msk rshift<4>(const msk v1, const uint32_t n) {
		const auto s = n * 4;
		     if (s <= 0) return v1;
		else if (s > 15) return set0<4>();
		else             return (msk)vextq_s8((int8x16_t)v1, vdupq_n_s8(0), s);
	}

	template <>
	inline msk rshift<8>(const msk v1, const uint32_t n) {
		const auto s = n * 2;
		     if (s <= 0) return v1;
		else if (s > 15) return set0<8>();
		else             return (msk)vextq_s8((int8x16_t)v1, vdupq_n_s8(0), s);
	}

	template <>
	inline msk rshift<16>(const msk v1, const uint32_t n) {
		const auto s = n;
		     if (s <= 0) return v1;
		else if (s > 15) return set0<16>();
		else             return (msk)vextq_s8((int8x16_t)v1, vdupq_n_s8(0), s);
	}
#endif

	// ---------------------------------------------------------------------------------------------------------- blend
#ifdef __aarch64__
	template <>
	inline reg blend<double>(const reg v1, const reg v2, const msk m) {
		return (reg) vbslq_f64((uint64x2_t)m, (float64x2_t)v1, (float64x2_t)v2);
	}
#endif

	template <>
	inline reg blend<float>(const reg v1, const reg v2, const msk m) {
		return (reg) vbslq_f32((uint32x4_t)m, (float32x4_t)v1, (float32x4_t)v2);
	}

#ifdef __aarch64__
	template <>
	inline reg blend<int64_t>(const reg v1, const reg v2, const msk m) {
		return (reg) vbslq_u64((uint64x2_t)m, (uint64x2_t)v1, (uint64x2_t)v2);
	}
#endif

	template <>
	inline reg blend<int32_t>(const reg v1, const reg v2, const msk m) {
		return (reg) vbslq_u32((uint32x4_t)m, (uint32x4_t)v1, (uint32x4_t)v2);
	}

	template <>
	inline reg blend<int16_t>(const reg v1, const reg v2, const msk m) {
		return (reg) vbslq_u16((uint16x8_t)m, (uint16x8_t)v1, (uint16x8_t)v2);
	}

	template <>
	inline reg blend<int8_t>(const reg v1, const reg v2, const msk m) {
		return (reg) vbslq_u8((uint8x16_t)m, (uint8x16_t)v1, (uint8x16_t)v2);
	}

	// ---------------------------------------------------------------------------------------------------------- cmpeq
#ifdef __aarch64__
	template <>
	inline msk cmpeq<double>(const reg v1, const reg v2) {
		return (msk) vceqq_f64((float64x2_t)v1, (float64x2_t)v2);
	}
#endif

	template <>
	inline msk cmpeq<float>(const reg v1, const reg v2) {
		return (msk) vceqq_f32(v1, v2);
	}

#ifdef __aarch64__
	template <>
	inline msk cmpeq<int64_t>(const reg v1, const reg v2) {
		return (msk) vceqq_s64((int64x2_t) v1, (int64x2_t) v2);
	}
#endif

	template <>
	inline msk cmpeq<int32_t>(const reg v1, const reg v2) {
		return (msk) vceqq_s32((int32x4_t) v1, (int32x4_t) v2);
	}

	template <>
	inline msk cmpeq<int16_t>(const reg v1, const reg v2) {
		return (msk) vceqq_s16((int16x8_t) v1, (int16x8_t) v2);
	}

	template <>
	inline msk cmpeq<int8_t>(const reg v1, const reg v2) {
		return (msk) vceqq_s8((int8x16_t) v1, (int8x16_t) v2);
	}

	// --------------------------------------------------------------------------------------------------------- cmpneq
#ifdef __aarch64__
	template <>
	inline msk cmpneq<double>(const reg v1, const reg v2) {
		return (msk) notb<2>(cmpeq<double>(v1, v2));
	}
#endif

	template <>
	inline msk cmpneq<float>(const reg v1, const reg v2) {
		return (msk) vmvnq_u32((uint32x4_t) vceqq_f32(v1, v2));
	}

#ifdef __aarch64__
	template <>
	inline msk cmpneq<int64_t>(const reg v1, const reg v2) {
		return (msk) notb<2>(cmpeq<int64_t>(v1, v2));
	}
#endif

	template <>
	inline msk cmpneq<int32_t>(const reg v1, const reg v2) {
		return (msk) vmvnq_u32((uint32x4_t) vceqq_s32((int32x4_t) v1, (int32x4_t) v2));
	}

	template <>
	inline msk cmpneq<int16_t>(const reg v1, const reg v2) {
		return (msk) vmvnq_u16((uint16x8_t) vceqq_s16((int16x8_t) v1, (int16x8_t) v2));
	}

	template <>
	inline msk cmpneq<int8_t>(const reg v1, const reg v2) {
		return (msk) vmvnq_u8((uint8x16_t) vceqq_s8((int8x16_t) v1, (int8x16_t) v2));
	}

	// ---------------------------------------------------------------------------------------------------------- cmplt
#ifdef __aarch64__
	template <>
	inline msk cmplt<double>(const reg v1, const reg v2) {
		return (msk) vcltq_f64((float64x2_t)v1, (float64x2_t)v2);
	}
#endif

	template <>
	inline msk cmplt<float>(const reg v1, const reg v2) {
		return (msk) vcltq_f32(v1, v2);
	}

#ifdef __aarch64__
	template <>
	inline msk cmplt<int64_t>(const reg v1, const reg v2) {
		return (msk) vcltq_s64((int64x2_t) v1, (int64x2_t) v2);
	}
#endif

	template <>
	inline msk cmplt<int32_t>(const reg v1, const reg v2) {
		return (msk) vcltq_s32((int32x4_t) v1, (int32x4_t) v2);
	}

	template <>
	inline msk cmplt<int16_t>(const reg v1, const reg v2) {
		return (msk) vcltq_s16((int16x8_t) v1, (int16x8_t) v2);
	}

	template <>
	inline msk cmplt<int8_t>(const reg v1, const reg v2) {
		return (msk) vcltq_s8((int8x16_t) v1, (int8x16_t) v2);
	}

	// ---------------------------------------------------------------------------------------------------------- cmple
#ifdef __aarch64__
	template <>
	inline msk cmple<double>(const reg v1, const reg v2) {
		return (msk) vcleq_f64((float64x2_t)v1, (float64x2_t)v2);
	}
#endif

	template <>
	inline msk cmple<float>(const reg v1, const reg v2) {
		return (msk) vcleq_f32(v1, v2);
	}

#ifdef __aarch64__
	template <>
	inline msk cmple<int64_t>(const reg v1, const reg v2) {
		return (msk) vcleq_s64((int64x2_t) v1, (int64x2_t) v2);
	}
#endif

	template <>
	inline msk cmple<int32_t>(const reg v1, const reg v2) {
		return (msk) vcleq_s32((int32x4_t) v1, (int32x4_t) v2);
	}

	template <>
	inline msk cmple<int16_t>(const reg v1, const reg v2) {
		return (msk) vcleq_s16((int16x8_t) v1, (int16x8_t) v2);
	}

	template <>
	inline msk cmple<int8_t>(const reg v1, const reg v2) {
		return (msk) vcleq_s8((int8x16_t) v1, (int8x16_t) v2);
	}

	// ---------------------------------------------------------------------------------------------------------- cmpgt
#ifdef __aarch64__
	template <>
	inline msk cmpgt<double>(const reg v1, const reg v2) {
		return (msk) vcgtq_f64((float64x2_t)v1, (float64x2_t)v2);
	}
#endif

	template <>
	inline msk cmpgt<float>(const reg v1, const reg v2) {
		return (msk) vcgtq_f32(v1, v2);
	}

#ifdef __aarch64__
	template <>
	inline msk cmpgt<int64_t>(const reg v1, const reg v2) {
		return (msk) vcgtq_s64((int64x2_t) v1, (int64x2_t) v2);
	}
#endif

	template <>
	inline msk cmpgt<int32_t>(const reg v1, const reg v2) {
		return (msk) vcgtq_s32((int32x4_t) v1, (int32x4_t) v2);
	}

	template <>
	inline msk cmpgt<int16_t>(const reg v1, const reg v2) {
		return (msk) vcgtq_s16((int16x8_t) v1, (int16x8_t) v2);
	}

	template <>
	inline msk cmpgt<int8_t>(const reg v1, const reg v2) {
		return (msk) vcgtq_s8((int8x16_t) v1, (int8x16_t) v2);
	}

	// ---------------------------------------------------------------------------------------------------------- cmpge
#ifdef __aarch64__
	template <>
	inline msk cmpge<double>(const reg v1, const reg v2) {
		return (msk) vcgeq_f64((float64x2_t)v1, (float64x2_t)v2);
	}
#endif

	template <>
	inline msk cmpge<float>(const reg v1, const reg v2) {
		return (msk) vcgeq_f32(v1, v2);
	}

#ifdef __aarch64__
	template <>
	inline msk cmpge<int64_t>(const reg v1, const reg v2) {
		return (msk) vcgeq_s64((int64x2_t) v1, (int64x2_t) v2);
	}
#endif

	template <>
	inline msk cmpge<int32_t>(const reg v1, const reg v2) {
		return (msk) vcgeq_s32((int32x4_t) v1, (int32x4_t) v2);
	}

	template <>
	inline msk cmpge<int16_t>(const reg v1, const reg v2) {
		return (msk) vcgeq_s16((int16x8_t) v1, (int16x8_t) v2);
	}

	template <>
	inline msk cmpge<int8_t>(const reg v1, const reg v2) {
		return (msk) vcgeq_s8((int8x16_t) v1, (int8x16_t) v2);
	}

	// ------------------------------------------------------------------------------------------------------------ add
#ifdef __aarch64__
	template <>
	inline reg add<double>(const reg v1, const reg v2) {
		return (reg) vaddq_f64((float64x2_t)v1, (float64x2_t)v2);
	}
#endif

	template <>
	inline reg add<float>(const reg v1, const reg v2) {
		return vaddq_f32(v1, v2);
	}

#ifdef __aarch64__
	template <>
	inline reg add<int64_t>(const reg v1, const reg v2) {
		return (reg) vaddq_s64((int64x2_t) v1, (int64x2_t)v2);
	}
#endif

	template <>
	inline reg add<int32_t>(const reg v1, const reg v2) {
		return (reg) vaddq_s32((int32x4_t) v1, (int32x4_t)v2);
	}

	template <>
	inline reg add<int16_t>(const reg v1, const reg v2) {
		return (reg) vqaddq_s16((int16x8_t) v1, (int16x8_t)v2);
	}

	template <>
	inline reg add<int8_t>(const reg v1, const reg v2) {
		return (reg) vqaddq_s8((int8x16_t) v1, (int8x16_t)v2);
	}

	template <>
	inline reg add<uint8_t>(const reg v1, const reg v2) {
		return (reg) vqaddq_u8((uint8x16_t) v1, (uint8x16_t)v2);
	}

	// ------------------------------------------------------------------------------------------------------------ sub
#ifdef __aarch64__
	template <>
	inline reg sub<double>(const reg v1, const reg v2) {
		return (reg) vsubq_f64((float64x2_t)v1, (float64x2_t)v2);
	}
#endif

	template <>
	inline reg sub<float>(const reg v1, const reg v2) {
		return vsubq_f32(v1, v2);
	}

#ifdef __aarch64__
	template <>
	inline reg sub<int64_t>(const reg v1, const reg v2) {
		return (reg) vsubq_s64((int64x2_t) v1, (int64x2_t) v2);
	}
#endif

	template <>
	inline reg sub<int32_t>(const reg v1, const reg v2) {
		return (reg) vsubq_s32((int32x4_t) v1, (int32x4_t) v2);
	}

	template <>
	inline reg sub<int16_t>(const reg v1, const reg v2) {
		return (reg) vqsubq_s16((int16x8_t) v1, (int16x8_t) v2);
	}

	template <>
	inline reg sub<int8_t>(const reg v1, const reg v2) {
		return (reg) vqsubq_s8((int8x16_t) v1, (int8x16_t) v2);
	}

	// ------------------------------------------------------------------------------------------------------------ mul
#ifdef __aarch64__
	template <>
	inline reg mul<double>(const reg v1, const reg v2) {
		return (reg) vmulq_f64((float64x2_t)v1, (float64x2_t)v2);
	}
#endif

	template <>
	inline reg mul<float>(const reg v1, const reg v2) {
		return vmulq_f32(v1, v2);
	}

	template <>
	inline reg mul<int32_t>(const reg v1, const reg v2) {
		return (reg) vmulq_s32((int32x4_t) v1, (int32x4_t) v2);
	}

	template <>
	inline reg mul<int16_t>(const reg v1, const reg v2) {
		return (reg) vmulq_s16((int16x8_t) v1, (int16x8_t) v2);
	}

	template <>
	inline reg mul<int8_t>(const reg v1, const reg v2) {
		return (reg) vmulq_s8((int8x16_t) v1, (int8x16_t) v2);
	}

	// ------------------------------------------------------------------------------------------------------------ div
#ifdef __aarch64__
	template <>
	inline reg div<double>(const reg v1, const reg v2) {
		return (reg) vdivq_f64((float64x2_t)v1, (float64x2_t)v2);
	}
#endif

#ifdef __aarch64__
	template <>
	inline reg div<float>(const reg v1, const reg v2) {
		return vdivq_f32(v1, v2);
	}
#else
	template <>
	inline reg div<float>(const reg v1, const reg v2) {
		return mul<float>(v1, vrecpeq_f32(v2));
	}
#endif

	// ------------------------------------------------------------------------------------------------------------ min
#ifdef __aarch64__
	template <>
	inline reg min<double>(const reg v1, const reg v2) {
		return (reg) vminq_f64((float64x2_t)v1, (float64x2_t)v2);
	}
#endif

	template <>
	inline reg min<float>(const reg v1, const reg v2) {
		return vminq_f32(v1, v2);
	}

	template <>
	inline reg min<int32_t>(const reg v1, const reg v2) {
		return (reg) vminq_s32((int32x4_t) v1, (int32x4_t) v2);
	}

	template <>
	inline reg min<int16_t>(const reg v1, const reg v2) {
		return (reg) vminq_s16((int16x8_t) v1, (int16x8_t) v2);
	}

	template <>
	inline reg min<int8_t>(const reg v1, const reg v2) {
		return (reg) vminq_s8((int8x16_t) v1, (int8x16_t) v2);
	}

	// ------------------------------------------------------------------------------------------------------------ max
#ifdef __aarch64__
	template <>
	inline reg max<double>(const reg v1, const reg v2) {
		return (reg) vmaxq_f64((float64x2_t)v1, (float64x2_t)v2);
	}
#endif

	template <>
	inline reg max<float>(const reg v1, const reg v2) {
		return vmaxq_f32(v1, v2);
	}

	template <>
	inline reg max<int32_t>(const reg v1, const reg v2) {
		return (reg) vmaxq_s32((int32x4_t) v1, (int32x4_t) v2);
	}

	template <>
	inline reg max<int16_t>(const reg v1, const reg v2) {
		return (reg) vmaxq_s16((int16x8_t) v1, (int16x8_t) v2);
	}

#define has_max_int8_t
	template <>
	inline reg max<int8_t>(const reg v1, const reg v2) {
		return (reg) vmaxq_s8((int8x16_t) v1, (int8x16_t) v2);
	}

	template <>
	inline reg max<uint8_t>(const reg v1, const reg v2) {
		return (reg) vmaxq_u8((uint8x16_t) v1, (uint8x16_t) v2);
	}

	// ----------------------------------------------------------------------------------------------------------- msb
#ifdef __aarch64__
	template <>
	inline reg msb<double>(const reg v1) {
		const reg msb_mask = set1<int64_t>(0x8000000000000000);
		return andb<double>(v1, msb_mask);
	}

	template <>
	inline reg msb<double>(const reg v1, const reg v2) {
		reg msb_v1_v2 = xorb<double>(v1, v2);
		    msb_v1_v2 = msb<double>(msb_v1_v2);
		return msb_v1_v2;
	}
#endif

	template <>
	inline reg msb<float>(const reg v1) {
		// msb_mask = 10000000000000000000000000000000 // 32 bits
		const reg msb_mask = set1<int32_t>(0x80000000);

		// indices  = 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
		// msb_mask =  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
		// v1       =  ù  €  è  é  à  &  z  y  x  w  v  u  t  s  r  q  p  o  n  m  l  k  j  i  h  g  f  e  d  c  b  a
		// res      =  ù  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
		return andb<float>(v1, msb_mask);
	}

	template <>
	inline reg msb<float>(const reg v1, const reg v2) {
		reg msb_v1_v2 = xorb<float>(v1, v2);
		    msb_v1_v2 = msb<float>(msb_v1_v2);
		return msb_v1_v2;
	}

#ifdef __aarch64__
	template <>
	inline reg msb<int64_t>(const reg v1) {
		const reg msb_mask = set1<int64_t>(0x8000000000000000);
		return andb<int64_t>(v1, msb_mask);
	}

	template <>
	inline reg msb<int64_t>(const reg v1, const reg v2) {
		reg msb_v1_v2 = xorb<int64_t>(v1, v2);
		    msb_v1_v2 = msb<int64_t>(msb_v1_v2);
		return msb_v1_v2;
	}
#endif

	template <>
	inline reg msb<int32_t>(const reg v1) {
		const reg msb_mask = set1<int32_t>(0x80000000);
		return andb<int32_t>(v1, msb_mask);
	}

	template <>
	inline reg msb<int32_t>(const reg v1, const reg v2) {
		reg msb_v1_v2 = xorb<int32_t>(v1, v2);
		    msb_v1_v2 = msb<int32_t>(msb_v1_v2);
		return msb_v1_v2;
	}

	template <>
	inline reg msb<int16_t>(const reg v1) {
		const reg msb_mask = set1<int16_t>(0x8000);
		return andb<int16_t>(v1, msb_mask);
	}

	template <>
	inline reg msb<int16_t>(const reg v1, const reg v2) {
		reg msb_v1_v2 = xorb<int16_t>(v1, v2);
		    msb_v1_v2 = msb<int16_t>(msb_v1_v2);
		return msb_v1_v2;
	}

	template <>
	inline reg msb<int8_t>(const reg v1) {
		const reg msb_mask = set1<int8_t>(0x80);
		return andb<int8_t>(v1, msb_mask);
	}

	template <>
	inline reg msb<int8_t>(const reg v1, const reg v2) {
		reg msb_v1_v2 = xorb<int8_t>(v1, v2);
		    msb_v1_v2 = msb<int8_t>(msb_v1_v2);
		return msb_v1_v2;
	}

	// ----------------------------------------------------------------------------------------------------------- sign
	template <>
	inline msk sign<double>(const reg v1) {
		return cmplt<double>(v1, set0<double>());
	}

	template <>
	inline msk sign<float>(const reg v1) {
		return cmplt<float>(v1, set0<float>());
	}

	template <>
	inline msk sign<int64_t>(const reg v1) {
		return cmplt<int64_t>(v1, set0<int64_t>());
	}

	template <>
	inline msk sign<int32_t>(const reg v1) {
		return cmplt<int32_t>(v1, set0<int32_t>());
	}

	template <>
	inline msk sign<int16_t>(const reg v1) {
		return cmplt<int16_t>(v1, set0<int16_t>());
	}

	template <>
	inline msk sign<int8_t>(const reg v1) {
		return cmplt<int8_t>(v1, set0<int8_t>());
	}

	// ------------------------------------------------------------------------------------------------------------ neg
#ifdef __aarch64__
	template <>
	inline reg neg<double>(const reg v1, const reg v2) {
		return xorb<double>(v1, msb<double>(v2));
	}

	template <>
	inline reg neg<double>(const reg v1, const msk v2) {
		return neg<double>(v1, toreg<2>(v2));
	}
#endif

	template <>
	inline reg neg<float>(const reg v1, const reg v2) {
		return xorb<float>(v1, msb<float>(v2));
	}

	template <>
	inline reg neg<float>(const reg v1, const msk v2) {
		return neg<float>(v1, toreg<4>(v2));
	}

#ifdef __aarch64__
	template <>
	inline reg neg<int64_t>(const reg v1, const reg v2) {
		reg neg_v1 = (reg) vqnegq_s64((int64x2_t) v1);
		reg v2_2   = orb  <int64_t>(v2, set1<int64_t>(1)); // hack to avoid -0 case
		reg mask   = toreg<2>(cmplt<int64_t>(v2_2, set0<int64_t>()));
		reg res1   = andb <int64_t>(mask, neg_v1);
		reg res2   = andb <int64_t>(notb<int64_t>(mask), v1);
		reg res    = orb  <int64_t>(res1, res2);
		return res;
	}

	template <>
	inline reg neg<int64_t>(const reg v1, const msk v2) {
		return neg<int64_t>(v1, toreg<2>(v2));
	}
#endif

	template <>
	inline reg neg<int32_t>(const reg v1, const reg v2) {
		reg neg_v1 = (reg) vqnegq_s32((int32x4_t) v1);
		reg v2_2   = orb  <int32_t>(v2, set1<int32_t>(1)); // hack to avoid -0 case
		reg mask   = toreg<4>(cmplt<int32_t>(v2_2, set0<int32_t>()));
		reg res1   = andb <int32_t>(mask, neg_v1);
		reg res2   = andb <int32_t>(notb<int32_t>(mask), v1);
		reg res    = orb  <int32_t>(res1, res2);
		return res;
	}

	template <>
	inline reg neg<int32_t>(const reg v1, const msk v2) {
		return neg<int32_t>(v1, toreg<4>(v2));
	}

	template <>
	inline reg neg<int16_t>(const reg v1, const reg v2) {
		reg neg_v1 = (reg) vqnegq_s16((int16x8_t) v1);
		reg v2_2   = orb  <int16_t>(v2, set1<int16_t>(1)); // hack to avoid -0 case
		reg mask   = toreg<8>(cmplt<int16_t>(v2_2, set0<int16_t>()));
		reg res1   = andb <int16_t>(mask, neg_v1);
		reg res2   = andb <int16_t>(notb<int16_t>(mask), v1);
		reg res    = orb  <int16_t>(res1, res2);
		return res;
	}

	template <>
	inline reg neg<int16_t>(const reg v1, const msk v2) {
		return neg<int16_t>(v1, toreg<8>(v2));
	}

	template <>
	inline reg neg<int8_t>(const reg v1, const reg v2) {
		reg neg_v1 = (reg) vqnegq_s8((int8x16_t) v1);
		reg v2_2   = orb  <int8_t>(v2, set1<int8_t>(1)); // hack to avoid -0 case
		reg mask   = toreg<16>(cmplt<int8_t>(v2_2, set0<int8_t>()));
		reg res1   = andb <int8_t>(mask, neg_v1);
		reg res2   = andb <int8_t>(notb<int8_t>(mask), v1);
		reg res    = orb  <int8_t>(res1, res2);
		return res;
	}

	template <>
	inline reg neg<int8_t>(const reg v1, const msk v2) {
		return neg<int8_t>(v1, toreg<16>(v2));
	}

	// ------------------------------------------------------------------------------------------------------------ abs
#ifdef __aarch64__
	template <>
	inline reg abs<double>(const reg v1) {
		return (reg) vabsq_f64((float64x2_t)v1);
	}
#endif

	template <>
	inline reg abs<float>(const reg v1) {
		return vabsq_f32(v1);
	}

#ifdef __aarch64__
	template <>
	inline reg abs<int64_t>(const reg v1) {
		return (reg) vabsq_s64((int64x2_t)v1);
	}
#endif

	template <>
	inline reg abs<int32_t>(const reg v1) {
		return (reg) vabsq_s32((int32x4_t)v1);
	}

	template <>
	inline reg abs<int16_t>(const reg v1) {
		return (reg) vabsq_s16((int16x8_t)v1);
	}

	template <>
	inline reg abs<int8_t>(const reg v1) {
		return (reg) vabsq_s8((int8x16_t)v1);
	}

	// ---------------------------------------------------------------------------------------------------------- rsqrt
#ifdef __aarch64__
	template <>
	inline reg rsqrt<double>(const reg v1) {
		return (reg) vrsqrteq_f64((float64x2_t)v1);
	}
#endif

	template <>
	inline reg rsqrt<float>(const reg v1) {
		return vrsqrteq_f32(v1);
	}

	// ----------------------------------------------------------------------------------------------------------- sqrt
#ifdef __aarch64__
	template <>
	inline reg sqrt<double>(const reg v1) {
		return (reg) vrecpeq_f64((float64x2_t) rsqrt<double>(v1));
	}
#endif

	template <>
	inline reg sqrt<float>(const reg v1) {
		return vrecpeq_f32(rsqrt<float>(v1));
	}

	// ------------------------------------------------------------------------------------------------------------ log
	template <>
	inline reg log<float>(const reg v) {
		auto v_bis = v;
		return (reg) log_ps(v_bis);
	}

	// ------------------------------------------------------------------------------------------------------------ exp
	template <>
	inline reg exp<float>(const reg v) {
		auto v_bis = v;
		return (reg) exp_ps(v_bis);
	}

	// ------------------------------------------------------------------------------------------------------------ sin
	template <>
	inline reg sin<float>(const reg v) {
		auto v_bis = v;
		return (reg) sin_ps(v_bis);
	}

	// ------------------------------------------------------------------------------------------------------------ cos
	template <>
	inline reg cos<float>(const reg v) {
		auto v_bis = v;
		return (reg) cos_ps(v_bis);
	}

	// --------------------------------------------------------------------------------------------------------- sincos
	template <>
	inline void sincos<float>(const reg x, reg &s, reg &c) {
		sincos_ps(x, &s, &c);
	}

	// ---------------------------------------------------------------------------------------------------------- fmadd
#ifdef __aarch64__
	template <>
	inline reg fmadd<double>(const reg v1, const reg v2, const reg v3) {
#ifdef __ARM_FEATURE_FMA
		return (reg) vfmaq_f64((float64x2_t)v3, (float64x2_t)v1, (float64x2_t)v2);
#else
 		return add<double>(mul<double>(v1, v2), v3);
 #endif
	}
#endif

	template <>
	inline reg fmadd<float>(const reg v1, const reg v2, const reg v3) {
#ifdef __ARM_FEATURE_FMA
		return (reg) vfmaq_f32((float32x4_t)v3, (float32x4_t)v1, (float32x4_t)v2);
#else
 		return add<float>(mul<float>(v1, v2), v3);
#endif
	}

	// --------------------------------------------------------------------------------------------------------- fnmadd
#ifdef __aarch64__
	template <>
	inline reg fnmadd<double>(const reg v1, const reg v2, const reg v3) {
#if defined(__ARM_FEATURE_FMA) && !defined(__clang__)
		return (reg) vfmsq_f64((float64x2_t)v3, (float64x2_t)v1, (float64x2_t)v2);
#else
 		return sub<double>(v3, mul<double>(v1, v2));
#endif
	}
#endif

	template <>
	inline reg fnmadd<float>(const reg v1, const reg v2, const reg v3) {
#if defined(__ARM_FEATURE_FMA) && !defined(__clang__)
		return (reg) vfmsq_f32((float32x4_t)v3, (float32x4_t)v1, (float32x4_t)v2);
#else
 		return sub<float>(v3, mul<float>(v1, v2));
#endif
	}

	// ---------------------------------------------------------------------------------------------------------- fmsub
	template <>
	inline reg fmsub<double>(const reg v1, const reg v2, const reg v3) {
		return fmadd<double>(v1, v2, xorb<double>(v3, set1<int64_t>(0x8000000000000000)));
	}

	template <>
	inline reg fmsub<float>(const reg v1, const reg v2, const reg v3) {
		return fmadd<float>(v1, v2, xorb<float>(v3, set1<int32_t>(0x80000000)));
	}

	// --------------------------------------------------------------------------------------------------------- fnmsub
	template <>
	inline reg fnmsub<double>(const reg v1, const reg v2, const reg v3) {
		return xorb<double>(fmadd<double>(v1, v2, v3), set1<int64_t>(0x8000000000000000));
	}

	template <>
	inline reg fnmsub<float>(const reg v1, const reg v2, const reg v3) {
		return xorb<float>(fmadd<float>(v1, v2, v3), set1<int32_t>(0x80000000));
	}

	// ----------------------------------------------------------------------------------------------------------- lrot
#ifdef __aarch64__
	template <>
	inline reg lrot<double>(const reg v1) {
		return (reg) vextq_f64((float64x2_t)v1, (float64x2_t)v1, 1);
	}
#endif

	template <>
	inline reg lrot<float>(const reg v1) {
		return vextq_f32(v1, v1, 1);
	}

	template <>
	inline reg lrot<int64_t>(const reg v1) {
		return (reg) vextq_u64((uint64x2_t)v1, (uint64x2_t)v1, 1);
	}

	template <>
	inline reg lrot<int32_t>(const reg v1) {
		return (reg) vextq_u32((uint32x4_t)v1, (uint32x4_t)v1, 1);
	}

	template <>
	inline reg lrot<int16_t>(const reg v1) {
		return (reg) vextq_u16((uint16x8_t)v1, (uint16x8_t)v1, 1);
	}

	template <>
	inline reg lrot<int8_t>(const reg v1) {
		return (reg) vextq_u8((uint8x16_t)v1, (uint8x16_t)v1, 1);
	}

	// ----------------------------------------------------------------------------------------------------------- rrot
#ifdef __aarch64__
	template <>
	inline reg rrot<double>(const reg v1) {
		return (reg) vextq_f64((float64x2_t)v1, (float64x2_t)v1, 1);
	}
#endif

	template <>
	inline reg rrot<float>(const reg v1) {
		return vextq_f32(v1, v1, 3);
	}

	template <>
	inline reg rrot<int64_t>(const reg v1) {
		return (reg) vextq_u64((uint64x2_t)v1, (uint64x2_t)v1, 1);
	}

	template <>
	inline reg rrot<int32_t>(const reg v1) {
		return (reg) vextq_u32((uint32x4_t)v1, (uint32x4_t)v1, 3);
	}

	template <>
	inline reg rrot<int16_t>(const reg v1) {
		return (reg) vextq_u16((uint16x8_t)v1, (uint16x8_t)v1, 7);
	}

	template <>
	inline reg rrot<int8_t>(const reg v1) {
		return (reg) vextq_u8((uint8x16_t)v1, (uint8x16_t)v1, 15);
	}

	// ----------------------------------------------------------------------------------------------------------- div2
#ifdef __aarch64__
	template <>
	inline reg div2<double>(const reg v1) {
		return mul<double>(v1, set1<double>(0.5));
	}
#endif

	template <>
	inline reg div2<float>(const reg v1) {
		return mul<float>(v1, set1<float>(0.5f));
	}

#ifdef __aarch64__
	template <>
	inline reg div2<int64_t>(const reg v1) {
		reg abs_v1 = abs<int64_t>(v1);
		reg sh = rshift<int64_t>(abs_v1, 1);
		return neg<int64_t>(sh, v1);
	}
#endif

	template <>
	inline reg div2<int32_t>(const reg v1) {
		reg abs_v1 = abs<int32_t>(v1);
		reg sh = rshift<int32_t>(abs_v1, 1);
		return neg<int32_t>(sh, v1);
	}

	template <>
	inline reg div2<int16_t>(const reg v1) {
		reg abs_v1 = abs<int16_t>(v1);
		reg sh = rshift<int16_t>(abs_v1, 1);
		return neg<int16_t>(sh, v1);
	}

	template <>
	inline reg div2<int8_t>(const reg v1) {
		reg abs_v1 = abs<int8_t>(v1);
		reg sh = rshift<int8_t>(abs_v1, 1);
		return neg<int8_t>(sh, v1);
	}

	// ----------------------------------------------------------------------------------------------------------- div4
#ifdef __aarch64__
	template <>
	inline reg div4<double>(const reg v1) {
		return mul<double>(v1, set1<double>(0.25));
	}
#endif

	template <>
	inline reg div4<float>(const reg v1) {
		return mul<float>(v1, set1<float>(0.25f));
	}

#ifdef __aarch64__
	template <>
	inline reg div4<int64_t>(const reg v1) {
		reg abs_v1 = abs<int64_t>(v1);
		reg sh = rshift<int64_t>(abs_v1, 2);
		return neg<int64_t>(sh, v1);
	}
#endif

	template <>
	inline reg div4<int32_t>(const reg v1) {
		reg abs_v1 = abs<int32_t>(v1);
		reg sh = rshift<int32_t>(abs_v1, 2);
		return neg<int32_t>(sh, v1);
	}

	template <>
	inline reg div4<int16_t>(const reg v1) {
		reg abs_v1 = abs<int16_t>(v1);
		reg sh = rshift<int16_t>(abs_v1, 2);
		return neg<int16_t>(sh, v1);
	}

	template <>
	inline reg div4<int8_t>(const reg v1) {
		reg abs_v1 = abs<int8_t>(v1);
		reg sh = rshift<int8_t>(abs_v1, 2);
		return neg<int8_t>(sh, v1);
	}

	// ------------------------------------------------------------------------------------------------------------ sat
#ifdef __aarch64__
	template <>
	inline reg sat<double>(const reg v1, double min, double max) {
		return mipp::min<double>(mipp::max<double>(v1, set1<double>(min)), set1<double>(max));
	}
#endif

	template <>
	inline reg sat<float>(const reg v1, float min, float max) {
		return mipp::min<float>(mipp::max<float>(v1, set1<float>(min)), set1<float>(max));
	}

#ifdef __aarch64__
	template <>
	inline reg sat<int64_t>(const reg v1, int64_t min, int64_t max) {
		return mipp::min<int64_t>(mipp::max<int64_t>(v1, set1<int64_t>(min)), set1<int64_t>(max));
	}
#endif

	template <>
	inline reg sat<int32_t>(const reg v1, int32_t min, int32_t max) {
		return mipp::min<int32_t>(mipp::max<int32_t>(v1, set1<int32_t>(min)), set1<int32_t>(max));
	}

	template <>
	inline reg sat<int16_t>(const reg v1, int16_t min, int16_t max) {
		return mipp::min<int16_t>(mipp::max<int16_t>(v1, set1<int16_t>(min)), set1<int16_t>(max));
	}

	template <>
	inline reg sat<int8_t>(const reg v1, int8_t min, int8_t max) {
		return mipp::min<int8_t>(mipp::max<int8_t>(v1, set1<int8_t>(min)), set1<int8_t>(max));
	}

	// ---------------------------------------------------------------------------------------------------------- round
#ifdef __aarch64__
	template <>
	inline reg round<double>(const reg v) {
		return (reg) vrndnq_f64((float64x2_t) v);
	}

	template <>
	inline reg round<float>(const reg v) {
		return (reg) vrndnq_f32((float32x4_t) v);
	}
#else
	template <>
	inline reg round<float>(const reg v) {
		auto half = mipp::orb<float>(mipp::msb<float>(v), mipp::set1<float>(0.5f));
		auto tmp = mipp::add<float>(v, half);
		return vcvtq_f32_s32(vcvtq_s32_f32(tmp));
	}

#endif

	// ------------------------------------------------------------------------------------------------------------ cvt
#ifdef __aarch64__
	template <>
	inline reg cvt<double,int64_t>(const reg v) {
		return (reg) vcvtq_s64_f64((float64x2_t) round<double>(v));
	}

	template <>
	inline reg cvt<int64_t,double>(const reg v) {
		return (reg) vcvtq_f64_s64((int64x2_t) v);
	}
#endif

	template <>
	inline reg cvt<float,int32_t>(const reg v) {
		return (reg) vcvtq_s32_f32((float32x4_t) round<float>(v));
	}

	template <>
	inline reg cvt<int32_t,float>(const reg v) {
		return (reg) vcvtq_f32_s32((int32x4_t) v);
	}

	template <>
	inline reg cvt<int8_t,int16_t>(const reg_2 v) {
		return (reg) vmovl_s8((int8x8_t) v);
	}

	template <>
	inline reg cvt<int16_t,int32_t>(const reg_2 v) {
		return (reg) vmovl_s16((int16x4_t) v);
	}

	template <>
	inline reg cvt<int32_t,int64_t>(const reg_2 v) {
		return (reg) vmovl_s32((int32x2_t) v);
	}

	// ----------------------------------------------------------------------------------------------------------- pack
	template <>
	inline reg pack<int64_t,int32_t>(const reg v1, const reg v2) {
		return (reg) vcombine_s32(vqmovn_s64((int64x2_t) v1), vqmovn_s64((int64x2_t) v2));
	}

	template <>
	inline reg pack<int32_t,int16_t>(const reg v1, const reg v2) {
		return (reg) vcombine_s16(vqmovn_s32((int32x4_t) v1), vqmovn_s32((int32x4_t) v2));
	}

	template <>
	inline reg pack<int16_t,int8_t>(const reg v1, const reg v2) {
		return (reg) vcombine_s8(vqmovn_s16((int16x8_t) v1), vqmovn_s16((int16x8_t) v2));
	}

	// ------------------------------------------------------------------------------------------------------ reduction
#ifdef __aarch64__
	template <red_op<double> OP>
	struct _reduction<double,OP>
	{
		static reg apply(const reg v1) {
			auto val = v1;

			val = OP(val, (reg) vextq_f64((float64x2_t)val, (float64x2_t)val, 1));

			return val;
		}
	};

	template <Red_op<double> OP>
	struct _Reduction<double,OP>
	{
		static Reg<double> apply(const Reg<double> v1) {
			auto val = v1;

			val = OP(val, Reg<double>((reg) vextq_f64((float64x2_t)val.r, (float64x2_t)val.r, 1)));

			return val;
		}
	};
#endif

	template <red_op<float> OP>
	struct _reduction<float,OP>
	{
		static reg apply(const reg v1) {
			auto val = v1;

			val = OP(val, (reg) vextq_f32(val, val, 2));

			float32x2_t low1  = vrev64_f32(vget_low_f32 (val));
			float32x2_t high1 = vrev64_f32(vget_high_f32(val));
			val = OP(val, vcombine_f32(low1, high1));

			return val;
		}
	};

	template <Red_op<float> OP>
	struct _Reduction<float,OP>
	{
		static Reg<float> apply(const Reg<float> v1) {
			auto val = v1;

			val = OP(val, Reg<float>((reg) vextq_f32(val.r, val.r, 2)));

			float32x2_t low1  = vrev64_f32(vget_low_f32 (val.r));
			float32x2_t high1 = vrev64_f32(vget_high_f32(val.r));
			val = OP(val, Reg<float>(vcombine_f32(low1, high1)));

			return val;
		}
	};

#ifdef __aarch64__
	template <red_op<int64_t> OP>
	struct _reduction<int64_t,OP>
	{
		static reg apply(const reg v1) {
			auto val = v1;

			val = OP(val, (reg) vextq_s64((int64x2_t)val, (int64x2_t)val, 1));

			return val;
		}
	};

	template <Red_op<int64_t> OP>
	struct _Reduction<int64_t,OP>
	{
		static Reg<int64_t> apply(const Reg<int64_t> v1) {
			auto val = v1;

			val = OP(val, Reg<int64_t>((reg) vextq_s64((int64x2_t)val.r, (int64x2_t)val.r, 1)));

			return val;
		}
	};
#endif

	template <red_op<int32_t> OP>
	struct _reduction<int32_t,OP>
	{
		static reg apply(const reg v1) {
			auto val = v1;

			val = OP(val, (reg) vextq_s32((int32x4_t) val, (int32x4_t) val, 2));

			int32x2_t low1  = vrev64_s32((int32x2_t) vget_low_s32 ((int32x4_t) val));
			int32x2_t high1 = vrev64_s32((int32x2_t) vget_high_s32((int32x4_t) val));
			val = OP(val, (reg) vcombine_s32((int32x2_t) low1, (int32x2_t) high1));

			return val;
		}
	};

	template <Red_op<int32_t> OP>
	struct _Reduction<int32_t,OP>
	{
		static Reg<int32_t> apply(const Reg<int32_t> v1) {
			auto val = v1;

			val = OP(val, Reg<int32_t>((reg) vextq_s32((int32x4_t) val.r, (int32x4_t) val.r, 2)));

			int32x2_t low1  = vrev64_s32((int32x2_t) vget_low_s32 ((int32x4_t) val.r));
			int32x2_t high1 = vrev64_s32((int32x2_t) vget_high_s32((int32x4_t) val.r));
			val = OP(val, Reg<int32_t>((reg) vcombine_s32((int32x2_t) low1, (int32x2_t) high1)));

			return val;
		}
	};

	template <red_op<int16_t> OP>
	struct _reduction<int16_t,OP>
	{
		static reg apply(const reg v1) {
			auto val = v1;

			val = OP(val, (reg) vextq_s32((int32x4_t) val, (int32x4_t) val, 2));

			int32x2_t low1  = vrev64_s32((int32x2_t) vget_low_s32 ((int32x4_t) val));
			int32x2_t high1 = vrev64_s32((int32x2_t) vget_high_s32((int32x4_t) val));
			val = OP(val, (reg) vcombine_s32((int32x2_t) low1, (int32x2_t) high1));

			int16x4_t low2  = vrev32_s16((int16x4_t) vget_low_s32((int32x4_t) val));
			int16x4_t high2 = vrev32_s16((int16x4_t) vget_high_s32((int32x4_t) val));
			val = OP(val, (reg) vcombine_s32((int32x2_t) low2, (int32x2_t) high2));

			return val;
		}
	};

	template <Red_op<int16_t> OP>
	struct _Reduction<int16_t,OP>
	{
		static Reg<int16_t> apply(const Reg<int16_t> v1) {
			auto val = v1;

			val = OP(val, Reg<int16_t>((reg) vextq_s32((int32x4_t) val.r, (int32x4_t) val.r, 2)));

			int32x2_t low1  = vrev64_s32((int32x2_t) vget_low_s32 ((int32x4_t) val.r));
			int32x2_t high1 = vrev64_s32((int32x2_t) vget_high_s32((int32x4_t) val.r));
			val = OP(val, Reg<int16_t>((reg) vcombine_s32((int32x2_t) low1, (int32x2_t) high1)));

			int16x4_t low2  = vrev32_s16((int16x4_t) vget_low_s32((int32x4_t) val.r));
			int16x4_t high2 = vrev32_s16((int16x4_t) vget_high_s32((int32x4_t) val.r));
			val = OP(val, Reg<int16_t>((reg) vcombine_s32((int32x2_t) low2, (int32x2_t) high2)));

			return val;
		}
	};

	template <red_op<int8_t> OP>
	struct _reduction<int8_t,OP>
	{
		static reg apply(const reg v1) {
			auto val = v1;

			val = OP(val, (reg) vextq_s32((int32x4_t) val, (int32x4_t) val, 2));

			int32x2_t low1  = vrev64_s32((int32x2_t) vget_low_s32 ((int32x4_t) val));
			int32x2_t high1 = vrev64_s32((int32x2_t) vget_high_s32((int32x4_t) val));
			val = OP(val, (reg) vcombine_s32((int32x2_t) low1, (int32x2_t) high1));

			int16x4_t low2  = vrev32_s16((int16x4_t) vget_low_s32((int32x4_t) val));
			int16x4_t high2 = vrev32_s16((int16x4_t) vget_high_s32((int32x4_t) val));
			val = OP(val, (reg) vcombine_s32((int32x2_t) low2, (int32x2_t) high2));

			int8x8_t low3  = vrev16_s8((int8x8_t) vget_low_s32((int32x4_t) val));
			int8x8_t high3 = vrev16_s8((int8x8_t) vget_high_s32((int32x4_t) val));
			val = OP(val, (reg) vcombine_s32((int32x2_t) low3, (int32x2_t) high3));

			return val;
		}
	};

	template <Red_op<int8_t> OP>
	struct _Reduction<int8_t,OP>
	{
		static Reg<int8_t> apply(const Reg<int8_t> v1) {
			auto val = v1;

			val = OP(val, Reg<int8_t>((reg) vextq_s32((int32x4_t) val.r, (int32x4_t) val.r, 2)));

			int32x2_t low1  = vrev64_s32((int32x2_t) vget_low_s32 ((int32x4_t) val.r));
			int32x2_t high1 = vrev64_s32((int32x2_t) vget_high_s32((int32x4_t) val.r));
			val = OP(val, Reg<int8_t>((reg) vcombine_s32((int32x2_t) low1, (int32x2_t) high1)));

			int16x4_t low2  = vrev32_s16((int16x4_t) vget_low_s32((int32x4_t) val.r));
			int16x4_t high2 = vrev32_s16((int16x4_t) vget_high_s32((int32x4_t) val.r));
			val = OP(val, Reg<int8_t>((reg) vcombine_s32((int32x2_t) low2, (int32x2_t) high2)));

			int8x8_t low3  = vrev16_s8((int8x8_t) vget_low_s32((int32x4_t) val.r));
			int8x8_t high3 = vrev16_s8((int8x8_t) vget_high_s32((int32x4_t) val.r));
			val = OP(val, Reg<int8_t>((reg) vcombine_s32((int32x2_t) low3, (int32x2_t) high3)));

			return val;
		}
	};

	// ---------------------------------------------------------------------------------------------------------- testz
#ifdef __aarch64__
	template <>
	inline bool testz<int64_t>(const reg v1, const reg v2) {
		auto andvec = mipp::andb<int64_t>(v1, v2);
		return mipp::reduction<int64_t, mipp::orb<int64_t>>::sapply(andvec) == 0;
	}
#endif

	template <>
	inline bool testz<int32_t>(const reg v1, const reg v2) {
		auto andvec = mipp::andb<int32_t>(v1, v2);
		return mipp::reduction<int32_t, mipp::orb<int32_t>>::sapply(andvec) == 0;
	}

	template <>
	inline bool testz<int16_t>(const reg v1, const reg v2) {
		auto andvec = mipp::andb<int16_t>(v1, v2);
		return mipp::reduction<int16_t, mipp::orb<int16_t>>::sapply(andvec) == 0;
	}

	template <>
	inline bool testz<int8_t>(const reg v1, const reg v2) {
		auto andvec = mipp::andb<int8_t>(v1, v2);
		return mipp::reduction<int8_t, mipp::orb<int8_t>>::sapply(andvec) == 0;
	}

#ifdef __aarch64__
	template <>
	inline bool testz<int64_t>(const reg v1) {
		auto tmp = vorr_u64(vget_low_u64((uint64x2_t)v1), vget_high_u64((uint64x2_t)v1));
		return !vget_lane_u64(tmp, 0);
	}

	template <>
	inline bool testz<int32_t>(const reg v1) {
		auto tmp = vorr_u64(vget_low_u64((uint64x2_t)v1), vget_high_u64((uint64x2_t)v1));
		return !vget_lane_u64(tmp, 0);
	}

	template <>
	inline bool testz<int16_t>(const reg v1) {
		auto tmp = vorr_u64(vget_low_u64((uint64x2_t)v1), vget_high_u64((uint64x2_t)v1));
		return !vget_lane_u64(tmp, 0);
	}

	template <>
	inline bool testz<int8_t>(const reg v1) {
		auto tmp = vorr_u64(vget_low_u64((uint64x2_t)v1), vget_high_u64((uint64x2_t)v1));
		return !vget_lane_u64(tmp, 0);
	}
#else
	template <>
	inline bool testz<int32_t>(const reg v1) {
		uint32x2_t v2 = vorr_u32(vget_low_u32((uint32x4_t)v1), vget_high_u32((uint32x4_t)v1));
		return !(vget_lane_u32(vpmax_u32(v2, v2), 0));
	}

	template <>
	inline bool testz<int16_t>(const reg v1) {
		uint32x2_t v2 = vorr_u32(vget_low_u32((uint32x4_t)v1), vget_high_u32((uint32x4_t)v1));
		return !(vget_lane_u32(vpmax_u32(v2, v2), 0));
	}

	template <>
	inline bool testz<int8_t>(const reg v1) {
		uint32x2_t v2 = vorr_u32(vget_low_u32((uint32x4_t)v1), vget_high_u32((uint32x4_t)v1));
		return !(vget_lane_u32(vpmax_u32(v2, v2), 0));
	}
#endif

	// --------------------------------------------------------------------------------------------------- testz (mask)
#ifdef __aarch64__
	template <>
	inline bool testz<2>(const msk v1, const msk v2) {
		auto andvec = mipp::andb<2>(v1, v2);
		return mipp::reduction<int64_t, mipp::orb<int64_t>>::sapply(mipp::toreg<2>(andvec)) == 0;
	}
#endif

	template <>
	inline bool testz<4>(const msk v1, const msk v2) {
		auto andvec = mipp::andb<4>(v1, v2);
		return mipp::reduction<int32_t, mipp::orb<int32_t>>::sapply(mipp::toreg<4>(andvec)) == 0;
	}

	template <>
	inline bool testz<8>(const msk v1, const msk v2) {
		auto andvec = mipp::andb<8>(v1, v2);
		return mipp::reduction<int16_t, mipp::orb<int16_t>>::sapply(mipp::toreg<8>(andvec)) == 0;
	}

	template <>
	inline bool testz<16>(const msk v1, const msk v2) {
		auto andvec = mipp::andb<16>(v1, v2);
		return mipp::reduction<int8_t, mipp::orb<int8_t>>::sapply(mipp::toreg<16>(andvec)) == 0;
	}

#ifdef __aarch64__
	template <>
	inline bool testz<2>(const msk v1) {
		auto tmp = vorr_u64(vget_low_u64((uint64x2_t)v1), vget_high_u64((uint64x2_t)v1));
		return !vget_lane_u64(tmp, 0);
	}

	template <>
	inline bool testz<4>(const msk v1) {
		auto tmp = vorr_u64(vget_low_u64((uint64x2_t)v1), vget_high_u64((uint64x2_t)v1));
		return !vget_lane_u64(tmp, 0);
	}

	template <>
	inline bool testz<8>(const msk v1) {
		auto tmp = vorr_u64(vget_low_u64((uint64x2_t)v1), vget_high_u64((uint64x2_t)v1));
		return !vget_lane_u64(tmp, 0);
	}

	template <>
	inline bool testz<16>(const msk v1) {
		auto tmp = vorr_u64(vget_low_u64((uint64x2_t)v1), vget_high_u64((uint64x2_t)v1));
		return !vget_lane_u64(tmp, 0);
	}
#else
	template <>
	inline bool testz<4>(const msk v1) {
		uint32x2_t v2 = vorr_u32(vget_low_u32((uint32x4_t)v1), vget_high_u32((uint32x4_t)v1));
		return !(vget_lane_u32(vpmax_u32(v2, v2), 0));
	}

	template <>
	inline bool testz<8>(const msk v1) {
		uint32x2_t v2 = vorr_u32(vget_low_u32((uint32x4_t)v1), vget_high_u32((uint32x4_t)v1));
		return !(vget_lane_u32(vpmax_u32(v2, v2), 0));
	}

	template <>
	inline bool testz<16>(const msk v1) {
		uint32x2_t v2 = vorr_u32(vget_low_u32((uint32x4_t)v1), vget_high_u32((uint32x4_t)v1));
		return !(vget_lane_u32(vpmax_u32(v2, v2), 0));
	}
#endif

	// ------------------------------------------------------------------------------------------------------ transpose
	template <>
	inline void transpose<int16_t>(reg tab[nElReg<int16_t>()]) {
		// /!\ this implementation can be further improved by using the dedicated VTRN instructions.
		//
		// Transpose the 8x8 matrix:
		// -------------------------
		// tab[0] = [a0, a1, a2, a3, a4, a5, a6, a7]        tab[0] = [a0, b0, c0, d0, e0, f0, g0, h0]
		// tab[1] = [b0, b1, b2, b3, b4, b5, b6, b7]        tab[1] = [a1, b1, c1, d1, e1, f1, g1, h1]
		// tab[2] = [c0, c1, c2, c3, c4, c5, c6, c7]        tab[2] = [a2, b2, c2, d2, e2, f2, g2, h2]
		// tab[3] = [d0, d1, d2, d3, d4, d5, d6, d7]        tab[3] = [a3, b3, c3, d3, e3, f3, g3, h3]
		// tab[4] = [e0, e1, e2, e3, e4, e5, e6, e7]   =>   tab[4] = [a4, b4, c4, d4, e4, f4, g4, h4]
		// tab[5] = [f0, f1, f2, f3, f4, f5, f6, f7]        tab[5] = [a5, b5, c5, d5, e5, f5, g5, h5]
		// tab[6] = [g0, g1, g2, g3, g4, g5, g6, g7]        tab[6] = [a6, b6, c6, d6, e6, f6, g6, h6]
		// tab[7] = [h0, h1, h2, h3, h4, h5, h6, h7]        tab[7] = [a7, b7, c7, d7, e7, f7, g7, h7]

		auto ab = mipp::interleave<int16_t>(tab[0], tab[1]);
		auto cd = mipp::interleave<int16_t>(tab[2], tab[3]);
		auto ef = mipp::interleave<int16_t>(tab[4], tab[5]);
		auto gh = mipp::interleave<int16_t>(tab[6], tab[7]);

		auto a03b03 = ab.val[0];
		auto c03d03 = cd.val[0];
		auto e03f03 = ef.val[0];
		auto g03h03 = gh.val[0];
		auto a47b47 = ab.val[1];
		auto c47d47 = cd.val[1];
		auto e47f47 = ef.val[1];
		auto g47h47 = gh.val[1];

		auto a03b03c03d03 = mipp::interleave<int32_t>(a03b03, c03d03);
		auto e03f03g03h03 = mipp::interleave<int32_t>(e03f03, g03h03);
		auto a47b47c47d47 = mipp::interleave<int32_t>(a47b47, c47d47);
		auto e47f47g47h47 = mipp::interleave<int32_t>(e47f47, g47h47);

		auto a01b01c01d01 = a03b03c03d03.val[0];
		auto a23b23c23d23 = a03b03c03d03.val[1];
		auto e01f01g01h01 = e03f03g03h03.val[0];
		auto e23f23g23h23 = e03f03g03h03.val[1];
		auto a45b45c45d45 = a47b47c47d47.val[0];
		auto a67b67c67d67 = a47b47c47d47.val[1];
		auto e45f45g45h45 = e47f47g47h47.val[0];
		auto e67f67g67h67 = e47f47g47h47.val[1];

		auto a01b01c01d01e01f01g01h01 = mipp::interleave<int64_t>(a01b01c01d01, e01f01g01h01);
		auto a23b23c23d23e23f23g23h23 = mipp::interleave<int64_t>(a23b23c23d23, e23f23g23h23);
		auto a45b45c45d45e45f45g45h45 = mipp::interleave<int64_t>(a45b45c45d45, e45f45g45h45);
		auto a67b67c67d67e67f67g67h67 = mipp::interleave<int64_t>(a67b67c67d67, e67f67g67h67);

		auto a0b0c0d0e0f0g0h0 = a01b01c01d01e01f01g01h01.val[0];
		auto a1b1c1d1e1f1g1h1 = a01b01c01d01e01f01g01h01.val[1];
		auto a2b2c2d2e2f2g2h2 = a23b23c23d23e23f23g23h23.val[0];
		auto a3b3c3d3e3f3g3h3 = a23b23c23d23e23f23g23h23.val[1];
		auto a4b4c4d4e4f4g4h4 = a45b45c45d45e45f45g45h45.val[0];
		auto a5b5c5d5e5f5g5h5 = a45b45c45d45e45f45g45h45.val[1];
		auto a6b6c6d6e6f6g6h6 = a67b67c67d67e67f67g67h67.val[0];
		auto a7b7c7d7e7f7g7h7 = a67b67c67d67e67f67g67h67.val[1];

		tab[0] = (reg)a0b0c0d0e0f0g0h0;
		tab[1] = (reg)a1b1c1d1e1f1g1h1;
		tab[2] = (reg)a2b2c2d2e2f2g2h2;
		tab[3] = (reg)a3b3c3d3e3f3g3h3;
		tab[4] = (reg)a4b4c4d4e4f4g4h4;
		tab[5] = (reg)a5b5c5d5e5f5g5h5;
		tab[6] = (reg)a6b6c6d6e6f6g6h6;
		tab[7] = (reg)a7b7c7d7e7f7g7h7;
	}

	// --------------------------------------------------------------------------------------------------- transpose8x8
	template <>
	inline void transpose8x8<int8_t>(reg tab[8]) {
		mipp::transpose<int16_t>(tab);
	}
#endif
