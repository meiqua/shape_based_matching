#ifndef MIPP_SCALAR_OP_H_
#define MIPP_SCALAR_OP_H_

namespace mipp_scop // My Intrinsics Plus Plus SCalar OPerations
{
	template <typename T>
	inline T add(const T val1, const T val2);

	template <typename T>
	inline T sub(const T val1, const T val2);

	template <typename T>
	inline T andb(const T val1, const T val2);

	template <typename T>
	inline T xorb(const T val1, const T val2);

	template <typename T>
	inline T msb(const T val);

	template <typename T>
	inline T div2(const T val);

	template <typename T>
	inline T div4(const T val);

	template <typename T>
	inline T rshift(const T val, const int n);

	template <typename T>
	inline T lshift(const T val, const int n);
}

#include "mipp_scalar_op.hxx"

#endif /* MIPP_SCALAR_OP_H_ */
