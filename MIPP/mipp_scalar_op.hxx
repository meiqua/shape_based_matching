#include <limits>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <algorithm>

#include "mipp_scalar_op.h"

namespace mipp_scop
{
template <typename T> inline T       add(const T       val1, const T       val2) { return val1 + val2; }
template <          > inline int16_t add(const int16_t val1, const int16_t val2) { return (int16_t)std::min(std::max((int32_t)((int32_t)val1 + (int32_t)val2),(int32_t)std::numeric_limits<int16_t>::min()),(int32_t)std::numeric_limits<int16_t>::max()); }
template <          > inline int8_t  add(const int8_t  val1, const int8_t  val2) { return (int8_t )std::min(std::max((int16_t)((int16_t)val1 + (int16_t)val2),(int16_t)std::numeric_limits<int8_t >::min()),(int16_t)std::numeric_limits<int8_t >::max()); }

template <typename T> inline T       sub(const T       val1, const T       val2) { return val1 - val2; }
template <          > inline int16_t sub(const int16_t val1, const int16_t val2) { return (int16_t)std::min(std::max((int32_t)((int32_t)val1 - (int32_t)val2),(int32_t)std::numeric_limits<int16_t>::min()),(int32_t)std::numeric_limits<int16_t>::max()); }
template <          > inline int8_t  sub(const int8_t  val1, const int8_t  val2) { return (int8_t )std::min(std::max((int16_t)((int16_t)val1 - (int16_t)val2),(int16_t)std::numeric_limits<int8_t >::min()),(int16_t)std::numeric_limits<int8_t >::max()); }

template <typename T> inline T      andb(const T      val1, const T      val2) { return                                          val1  &                      val2;   }
template <          > inline double andb(const double val1, const double val2) { return static_cast<double>(static_cast<int64_t>(val1) & static_cast<int64_t>(val2)); }
template <          > inline float  andb(const float  val1, const float  val2) { return static_cast<float >(static_cast<int32_t>(val1) & static_cast<int32_t>(val2)); }

template <typename T> inline T      xorb(const T      val1, const T      val2) { return                                          val1  ^                      val2;   }
template <          > inline double xorb(const double val1, const double val2) { return static_cast<double>(static_cast<int64_t>(val1) ^ static_cast<int64_t>(val2)); }
template <          > inline float  xorb(const float  val1, const float  val2) { return static_cast<float >(static_cast<int32_t>(val1) ^ static_cast<int32_t>(val2)); }

template <typename T> inline T       msb(const T       val) { return (val >> (sizeof(T) * 8 -1)) << (sizeof(T) * 8 -1);              }
template <          > inline double  msb(const double  val) { return static_cast<double >((static_cast<uint64_t>(val) >> 63) << 63); }
template <          > inline float   msb(const float   val) { return static_cast<float  >((static_cast<uint32_t>(val) >> 31) << 31); }
template <          > inline int64_t msb(const int64_t val) { return static_cast<int64_t>((static_cast<uint64_t>(val) >> 63) << 63); }
template <          > inline int32_t msb(const int32_t val) { return static_cast<int32_t>((static_cast<uint32_t>(val) >> 31) << 31); }
template <          > inline int16_t msb(const int16_t val) { return static_cast<int16_t>((static_cast<uint16_t>(val) >> 15) << 15); }
template <          > inline int8_t  msb(const int8_t  val) { return static_cast<int8_t >((static_cast<uint8_t >(val) >>  7) <<  7); }

template <typename T> inline T       div2(const T       val) { return val * (T)0.5; }
template <          > inline int64_t div2(const int64_t val) { return val >> 1;     }
template <          > inline int32_t div2(const int32_t val) { return val >> 1;     }
template <          > inline int16_t div2(const int16_t val) { return val >> 1;     }
template <          > inline int8_t  div2(const int8_t  val) { return val >> 1;     }

template <typename T> inline T       div4(const T       val) { return val * (T)0.25; }
template <          > inline int64_t div4(const int64_t val) { return val >> 2;      }
template <          > inline int32_t div4(const int32_t val) { return val >> 2;      }
template <          > inline int16_t div4(const int16_t val) { return val >> 2;      }
template <          > inline int8_t  div4(const int8_t  val) { return val >> 2;      }

template <typename T> inline T       lshift(const T       val, const int n) { return                                            val  << n;  }
template <          > inline double  lshift(const double  val, const int n) { return static_cast<double >(static_cast<uint64_t>(val) << n); }
template <          > inline float   lshift(const float   val, const int n) { return static_cast<float  >(static_cast<uint32_t>(val) << n); }
template <          > inline int64_t lshift(const int64_t val, const int n) { return static_cast<int64_t>(static_cast<uint64_t>(val) << n); }
template <          > inline int32_t lshift(const int32_t val, const int n) { return static_cast<int32_t>(static_cast<uint32_t>(val) << n); }
template <          > inline int16_t lshift(const int16_t val, const int n) { return static_cast<int16_t>(static_cast<uint16_t>(val) << n); }
template <          > inline int8_t  lshift(const int8_t  val, const int n) { return static_cast<int8_t >(static_cast<uint8_t >(val) << n); }

template <typename T> inline T       rshift(const T       val, const int n) { return                                            val  >> n;  }
template <          > inline double  rshift(const double  val, const int n) { return static_cast<double >(static_cast<uint64_t>(val) >> n); }
template <          > inline float   rshift(const float   val, const int n) { return static_cast<float  >(static_cast<uint32_t>(val) >> n); }
template <          > inline int64_t rshift(const int64_t val, const int n) { return static_cast<int64_t>(static_cast<uint64_t>(val) >> n); }
template <          > inline int32_t rshift(const int32_t val, const int n) { return static_cast<int32_t>(static_cast<uint32_t>(val) >> n); }
template <          > inline int16_t rshift(const int16_t val, const int n) { return static_cast<int16_t>(static_cast<uint16_t>(val) >> n); }
template <          > inline int8_t  rshift(const int8_t  val, const int n) { return static_cast<int8_t >(static_cast<uint8_t >(val) >> n); }
}
