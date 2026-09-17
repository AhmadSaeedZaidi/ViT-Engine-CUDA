#pragma once

#include <cuda_runtime.h>

// Vec4: A layout-compatible replacement for float4 with descriptive field names
// and arithmetic operators. Maintains the same 16-byte alignment as float4,
// so it can be used with reinterpret_cast<const float4*> for vectorized loads.
//
// Usage:
//   Vec4 acc;                          // zero-initialized
//   Vec4 v(1.0f, 2.0f, 3.0f, 4.0f);    // explicit construction
//   Vec4 w = acc + v;                  // element-wise addition
//   Vec4 s = v * 0.5f;                 // scalar multiplication
//   float4 raw = v;                    // implicit conversion to float4
//   Vec4* ptr = reinterpret_cast<Vec4*>(float4_ptr);  // safe cast

struct Vec4 {
    float x, y, z, w;

    __device__ __forceinline__ Vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
    __device__ __forceinline__ Vec4(float v) : x(v), y(v), z(v), w(v) {}
    __device__ __forceinline__ Vec4(float x, float y, float z, float w)
        : x(x), y(y), z(z), w(w) {}

    // Implicit conversion to float4 (for reinterpret_cast compatibility)
    __device__ __forceinline__ operator float4&() {
        return *reinterpret_cast<float4*>(this);
    }
    __device__ __forceinline__ operator const float4&() const {
        return *reinterpret_cast<const float4*>(this);
    }

    // Element-wise addition
    __device__ __forceinline__ Vec4 operator+(const Vec4& o) const {
        return Vec4(x + o.x, y + o.y, z + o.z, w + o.w);
    }
    __device__ __forceinline__ Vec4& operator+=(const Vec4& o) {
        x += o.x; y += o.y; z += o.z; w += o.w;
        return *this;
    }

    // Element-wise subtraction
    __device__ __forceinline__ Vec4 operator-(const Vec4& o) const {
        return Vec4(x - o.x, y - o.y, z - o.z, w - o.w);
    }

    // Scalar multiplication
    __device__ __forceinline__ Vec4 operator*(float s) const {
        return Vec4(x * s, y * s, z * s, w * s);
    }
    __device__ __forceinline__ Vec4& operator*=(float s) {
        x *= s; y *= s; z *= s; w *= s;
        return *this;
    }

    // Scalar division
    __device__ __forceinline__ Vec4 operator/(float s) const {
        return Vec4(x / s, y / s, z / s, w / s);
    }
    __device__ __forceinline__ Vec4& operator/=(float s) {
        x /= s; y /= s; z /= s; w /= s;
        return *this;
    }

    // Fused multiply-add (element-wise)
    __device__ __forceinline__ Vec4 fma(const Vec4& a, const Vec4& b) const {
        return Vec4(
            __fmaf_rn(a.x, b.x, x),
            __fmaf_rn(a.y, b.y, y),
            __fmaf_rn(a.z, b.z, z),
            __fmaf_rn(a.w, b.w, w)
        );
    }

    // Dot product (sum of element-wise products)
    __device__ __forceinline__ float dot(const Vec4& o) const {
        return __fmaf_rn(x, o.x,
               __fmaf_rn(y, o.y,
               __fmaf_rn(z, o.z, w * o.w)));
    }

    // Broadcast: create Vec4 from a single value
    __device__ __forceinline__ static Vec4 splat(float v) {
        return Vec4(v, v, v, v);
    }

    // Load from a float4 (for compatibility with existing code)
    __device__ __forceinline__ static Vec4 from_float4(const float4& f) {
        return Vec4(f.x, f.y, f.z, f.w);
    }
};

// Free-function FMA helper (matches the pattern used in kernels)
__device__ __forceinline__ Vec4 fma4(const Vec4& acc, const Vec4& a, const Vec4& b) {
    return Vec4(
        __fmaf_rn(a.x, b.x, acc.x),
        __fmaf_rn(a.y, b.y, acc.y),
        __fmaf_rn(a.z, b.z, acc.z),
        __fmaf_rn(a.w, b.w, acc.w)
    );
}
