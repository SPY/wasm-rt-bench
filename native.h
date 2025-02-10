#include <arm_neon.h>
#include <chrono>

#include "common.h"

#define NOW duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()
// #define NOW 42

#ifdef F32X4
#define VERSION "f32x4"
#define SIMD_WIDTH 4
union v128;

struct f32x4 {
    f32 Value[4];

    inline f32x4() { }
    inline f32x4(f32 V) {
        for (int i = 0; i < 4; ++i) {
            Value[i] = V;
        }
    }

    inline f32 &operator[](uint32_t Index) {
        return Value[Index];
    }

    inline const f32 &operator[](uint32_t Index) const {
        return Value[Index];
    }

    static inline f32x4 SquareRoot(const f32x4 &A);
    static inline f32x4 InverseSquareRoot(const f32x4 &A);
    static inline f32x4 Min(const f32x4 &A, const f32x4 &B);
    static inline f32x4 Max(const f32x4 &A, const f32x4 &B);
    static inline void ConditionalMove(f32x4 *A, const f32x4 &B, const f32x4 &MoveMask);
    static inline f32 HorizontalMin(const f32x4 &A);
    static inline uint32_t HorizontalMinIndex(const f32x4 &A);

} __attribute__((__vector_size__(16), __aligned__(16)));

union v128 {
    f32 Float;
    v3 Vector3;
    f32x4 Float4;
    float32x4_t Register;

    inline v128() { Register = vdupq_n_f32(0.0f); };
    inline v128(f32 Value) : Float(Value) { };
    inline v128(const v3 &Value) : Vector3(Value) { };
    inline v128(const f32x4 &Value) : Float4(Value) { };
    inline v128(const float32x4_t &SIMDLane) : Register(SIMDLane) { };

    explicit operator f32() const { return Float; }
    explicit operator v3() const { return Vector3; }
    explicit operator f32x4() const { return Float4; }
    operator float32x4_t() const { return Register; }

    MATHCALL v128 CreateMask(bool Value) {
        v128 Result = vdupq_n_s32((uint32_t)Value * -1);
        return Result;
    }
};

inline f32x4 f32x4::SquareRoot(const f32x4 &A) {
    v128 Result = vrsqrteq_f32(v128(A));
    return (f32x4)Result;
}
inline f32x4 f32x4::Min(const f32x4 &A, const f32x4 &B) {
    v128 Result = vminq_f32(v128(A), v128(B));
    return (f32x4)Result;
}
inline f32x4 f32x4::Max(const f32x4 &A, const f32x4 &B) {
    v128 Result = vmaxq_f32(v128(A), v128(B));
    return (f32x4)Result;
}
inline void f32x4::ConditionalMove(f32x4 *A, const f32x4 &B, const f32x4 &MoveMask) {
    v128 Result = vbslq_f32(v128(B), v128(*A), v128(MoveMask));
    *A = (f32x4)Result;
}

inline f32 f32x4::HorizontalMin(const f32x4 &Value) {
    v128 UpperHalf = float32x4_t{Value[2], Value[3], 0, 0};
    v128 Result = vminq_f32(v128(Value), UpperHalf);
    Result = vminq_f32(Result, float32x4_t{Result.Float4[1], 0, 0, 0});
    return (f32)Result;
}
MATHCALL f32x4 operator==(const f32x4 &A, const f32x4 &B) {
    v128 Result = vceqq_f32(v128(A), v128(B));
    return (f32x4)Result;
}
inline uint32_t f32x4::HorizontalMinIndex(const f32x4 &Value) {
    f32 MinValue = f32x4::HorizontalMin(Value);
    v128 Comparison = Value == f32x4(MinValue);
    uint32_t MoveMask = 1; //wasm_i32x4_bitmask(Comparison);
    uint32_t Result = __builtin_ctz(MoveMask);
    return Result;
}

MATHCALL f32x4 operator+(const f32x4 &A, const f32x4 &B) {
    v128 Result = vaddq_f32(v128(A), v128(B));
    return (f32x4)Result;
}
MATHCALL f32x4 operator-(const f32x4 &A, const f32x4 &B) {
    v128 Result = vsubq_f32(v128(A), v128(B));
    return (f32x4)Result;
}
MATHCALL f32x4 operator*(const f32x4 &A, const f32x4 &B) {
    v128 Result = vmulq_f32(v128(A), v128(B));
    return (f32x4)Result;
}
MATHCALL f32x4 operator/(const f32x4 &A, const f32x4 &B) {
    v128 Result = vdivq_f32(v128(A), v128(B));
    return (f32x4)Result;
}
MATHCALL f32x4 operator!=(const f32x4 &A, const f32x4 &B) {
    v128 Result = vmvnq_u32(vceqq_f32(v128(A), v128(B)));
    return (f32x4)Result;
}
MATHCALL f32x4 operator>(const f32x4 &A, const f32x4 &B) {
    v128 Result = vcgtq_f32(v128(A), v128(B));
    return (f32x4)Result;
}
MATHCALL f32x4 operator<(const f32x4 &A, const f32x4 &B) {
    v128 Result = vcltq_f32(v128(A), v128(B));
    return (f32x4)Result;
}
MATHCALL f32x4 operator&(const f32x4 &A, const f32x4 &B) {
    v128 Result = vandq_u32(v128(A), v128(B));
    return (f32x4)Result;
}
MATHCALL f32x4 operator|(const f32x4 &A, const f32x4 &B) {
    v128 Result = vorrq_u32(v128(A), v128(B));
    return (f32x4)Result;
}
MATHCALL f32x4 operator^(const f32x4 &A, const f32x4 &B) {
    v128 Result = veorq_u32(v128(A), v128(B));
    return (f32x4)Result;
}
MATHCALL bool IsZero(f32x4 &Value) {
    float32x4_t v{Value.Value[0], Value.Value[1], Value.Value[2], Value.Value[3]};
    return vmaxvq_u32(v) == 0;
}

struct v3_reference {
    f32 &x, &y, &z;
    constexpr inline void operator=(const v3 &Value) {
        this->x = Value.x;
        this->y = Value.y;
        this->z = Value.z;
    }
};

struct v3x4 {
    f32x4 x, y, z;

    inline v3x4() { }
    inline v3x4(const f32 &&Value) : x(Value), y(Value), z(Value) { }
    inline v3x4(const f32x4 &Value) : x(Value), y(Value), z(Value) { }
    inline v3x4(const v3 &Value) : x(Value.x), y(Value.y), z(Value.z) { }

    inline v3_reference operator[](uint32_t Index) {
        v3_reference Result = {
            .x = x[Index],
            .y = y[Index],
            .z = z[Index],
        };
        return Result;
    }

    static inline f32x4 Dot(const v3x4 &A, const v3x4 &B);
    static inline f32x4 Length(const v3x4 &A);
    static inline f32x4 LengthSquared(const v3x4 &A);
    static inline v3x4 Normalize(const v3x4 &A);
    static inline v3x4 NormalizeFast(const v3x4 &A);
    static inline void ConditionalMove(v3x4 *A, const v3x4 &B, const f32x4 &MoveMask);
};

MATHCALL v3x4 operator+(const v3x4 &A, const v3x4 &B) {
    v3x4 Result;
    Result.x = A.x + B.x;
    Result.y = A.y + B.y;
    Result.z = A.z + B.z;
    return Result;
}
MATHCALL v3x4 operator-(const v3x4 &A, const v3x4 &B) {
    v3x4 Result;
    Result.x = A.x - B.x;
    Result.y = A.y - B.y;
    Result.z = A.z - B.z;
    return Result;
}
MATHCALL v3x4 operator*(const v3x4 &A, const v3x4 &B) {
    v3x4 Result;
    Result.x = A.x * B.x;
    Result.y = A.y * B.y;
    Result.z = A.z * B.z;
    return Result;
}
MATHCALL v3x4 operator/(const v3x4 &A, const v3x4 &B) {
    v3x4 Result;
    Result.x = A.x / B.x;
    Result.y = A.y / B.y;
    Result.z = A.z / B.z;
    return Result;
}
MATHCALL v3x4 operator&(const v3x4 &A, const v3x4 &B) {
    v3x4 Result;
    Result.x = A.x & B.x;
    Result.y = A.y & B.y;
    Result.z = A.z & B.z;
    return Result;
}

inline f32x4 v3x4::Dot(const v3x4 &A, const v3x4 &B) {
    v3x4 C = A * B;
    f32x4 Result = C.x + C.y + C.z;
    return Result;
}

inline f32x4 v3x4::Length(const v3x4 &A) {
    f32x4 LengthSquared = v3x4::Dot(A, A);
    f32x4 Length = f32x4::SquareRoot(LengthSquared);
    return Length;
}

inline f32x4 v3x4::LengthSquared(const v3x4 &A) {
    v3x4 C = A * A;
    f32x4 Result = C.x + C.y + C.z;
    return Result;
}

constexpr static f32 F32Epsilon = 1e-5f;

inline v3x4 v3x4::Normalize(const v3x4 &Value) {
    f32x4 LengthSquared = v3x4::LengthSquared(Value);

    f32x4 LengthGreaterThanZeroMask = LengthSquared > F32Epsilon;

    f32x4 Length = f32x4::SquareRoot(LengthSquared);
    v3x4 Result = Value / Length;

    v3x4 MaskedResult = Result & LengthGreaterThanZeroMask;
    return MaskedResult;
}

inline void v3x4::ConditionalMove(v3x4 *A, const v3x4 &B, const f32x4 &MoveMask) {
    f32x4::ConditionalMove(&A->x, B.x, MoveMask);
    f32x4::ConditionalMove(&A->y, B.y, MoveMask);
    f32x4::ConditionalMove(&A->z, B.z, MoveMask);
}

inline v3::v3(const f32 &X) {
    v128 Value = vdupq_n_f32(X);
    *this = (v3)Value;
}

inline f32 v3::Dot(const v3 &A, const v3 &B) {
    v3 Mul = A * B;
    return Mul.x + Mul.y + Mul.z;
}
inline f32 v3::LengthSquared(const v3 &Value) {
    return v3::Dot(Value, Value);
}

inline v3 v3::Normalize(const v3 &Value) {
    f32 LengthSquared = v3::LengthSquared(Value);

    bool LengthGreaterThanZero = LengthSquared > F32Epsilon;
    v128 Mask = v128::CreateMask(LengthGreaterThanZero);

    f32 Length = __builtin_sqrt(LengthSquared);
    v3 Result = Value / Length;

    v128 MaskedResult = vandq_u32(v128(Result), Mask);
    return (v3)MaskedResult;
}

MATHCALL v3 operator+(const v3 &A, const v3 &B) {
    v128 Result = vaddq_f32(v128(A), v128(B));
    return (v3)Result;
}
MATHCALL v3 operator-(const v3 &A, const v3 &B) {
    v128 Result = vsubq_f32(v128(A), v128(B));
    return (v3)Result;
}
MATHCALL v3 operator*(const v3 &A, const v3 &B) {
    v128 Result = vmulq_f32(v128(A), v128(B));
    return (v3)Result;
}
MATHCALL v3 operator/(const v3 &A, const v3 &B) {
    v128 Result = vdivq_f32(v128(A), v128(B));
    return (v3)Result;
}
MATHCALL v3 operator-(const v3 &A) {
    v128 Result = vnegq_f32(v128(A));
    return (v3)Result;
}

struct sphere_group {
    v3x4 Positions;
    f32x4 Radii;
    v3x4 Color;
    f32x4 Specular;
};
#endif


#ifdef F16X8
#define VERSION "f16x8"
#define SIMD_WIDTH 8
union v128;

using f16 = __fp16;

struct f16x8 {
    f16 Value[8];

    inline f16x8() { }
    inline f16x8(f32 V) {
        float16x8_t v = vdupq_n_f16(static_cast<f16>(V));
        memcpy(&Value, &v, 16);
    }

    inline f16 &operator[](uint32_t Index) {
        return Value[Index];
    }

    inline const f16 &operator[](uint32_t Index) const {
        return Value[Index];
    }

    static inline f16x8 SquareRoot(const f16x8 &A);
    static inline f16x8 InverseSquareRoot(const f16x8 &A);
    static inline f16x8 Min(const f16x8 &A, const f16x8 &B);
    static inline f16x8 Max(const f16x8 &A, const f16x8 &B);
    static inline void ConditionalMove(f16x8 *A, const f16x8 &B, const f16x8 &MoveMask);
    static inline f32 HorizontalMin(const f16x8 &A);
    static inline uint32_t HorizontalMinIndex(const f16x8 &A);

} __attribute__((__vector_size__(16), __aligned__(16)));

union v128 {
    f32 Float;
    v3 Vector3;
    f16x8 Half8;
    float16x8_t Register;

    inline v128() { Register = vdupq_n_f16(0.0f16); };
    inline v128(f32 Value) : Float(Value) { };
    inline v128(const v3 &Value) : Vector3(Value) { };
    inline v128(const f16x8 &Value) : Half8(Value) { };
    inline v128(const float16x8_t &SIMDLane) : Register(SIMDLane) { };

    explicit operator f32() const { return Float; }
    explicit operator v3() const { return Vector3; }
    explicit operator f16x8() const { return Half8; }
    operator float16x8_t() const { return Register; }

    MATHCALL v128 CreateMask(bool Value) {
        v128 Result = vdupq_n_u16((uint32_t)Value * -1);
        return Result;
    }
};

inline f16x8 f16x8::SquareRoot(const f16x8 &A) {
    v128 Result = vrsqrteq_f16(v128(A));
    return (f16x8)Result;
}
inline f16x8 f16x8::Min(const f16x8 &A, const f16x8 &B) {
    v128 Result = vminq_f16(v128(A), v128(B));
    return (f16x8)Result;
}
inline f16x8 f16x8::Max(const f16x8 &A, const f16x8 &B) {
    v128 Result = vmaxq_f16(v128(A), v128(B));
    return (f16x8)Result;
}
inline void f16x8::ConditionalMove(f16x8 *A, const f16x8 &B, const f16x8 &MoveMask) {
    v128 Result = vbslq_f16(v128(B), v128(*A), v128(MoveMask));
    *A = (f16x8)Result;
}

inline f32 f16x8::HorizontalMin(const f16x8 &Value) {
    float16x8_t UpperHalf{Value[4], Value[5], Value[6], Value[7], 0, 0, 0, 0};
    v128 Result = vminq_f16(v128(Value), UpperHalf);
    float16x8_t UpperQuad{Result.Half8[2], Result.Half8[3], 0, 0, 0, 0, 0, 0};
    Result = vminq_f16(Result, UpperQuad);
    Result = vminq_f16(Result, float16x8_t{Result.Half8[1], 0, 0, 0, 0, 0, 0, 0});
    return static_cast<f32>(Result.Half8[0]);
}
MATHCALL f16x8 operator==(const f16x8 &A, const f16x8 &B) {
    v128 Result = vceqq_f16(v128(A), v128(B));
    return (f16x8)Result;
}
inline uint32_t f16x8::HorizontalMinIndex(const f16x8 &Value) {
    f32 MinValue = f16x8::HorizontalMin(Value);
    v128 Comparison = Value == f16x8(MinValue);
    uint32_t MoveMask = 1; //wasm_i16x8_bitmask(Comparison);
    uint32_t Result = __builtin_ctz(MoveMask);
    return Result;
}

MATHCALL f16x8 operator+(const f16x8 &A, const f16x8 &B) {
    v128 Result = vaddq_f16(v128(A), v128(B));
    return (f16x8)Result;
}
MATHCALL f16x8 operator-(const f16x8 &A, const f16x8 &B) {
    v128 Result = vsubq_f16(v128(A), v128(B));
    return (f16x8)Result;
}
MATHCALL f16x8 operator*(const f16x8 &A, const f16x8 &B) {
    v128 Result = vmulq_f16(v128(A), v128(B));
    return (f16x8)Result;
}
MATHCALL f16x8 operator/(const f16x8 &A, const f16x8 &B) {
    v128 Result = vdivq_f16(v128(A), v128(B));
    return (f16x8)Result;
}
MATHCALL f16x8 operator!=(const f16x8 &A, const f16x8 &B) {
    v128 Result = vmvnq_u16(vceqq_f16(v128(A), v128(B)));
    return (f16x8)Result;
}
MATHCALL f16x8 operator>(const f16x8 &A, const f16x8 &B) {
    v128 Result = vcgtq_f16(v128(A), v128(B));
    return (f16x8)Result;
}
MATHCALL f16x8 operator<(const f16x8 &A, const f16x8 &B) {
    v128 Result = vcltq_f16(v128(A), v128(B));
    return (f16x8)Result;
}
MATHCALL f16x8 operator&(const f16x8 &A, const f16x8 &B) {
    v128 Result = vandq_u16(v128(A), v128(B));
    return (f16x8)Result;
}
MATHCALL f16x8 operator|(const f16x8 &A, const f16x8 &B) {
    v128 Result = vorrq_u16(v128(A), v128(B));
    return (f16x8)Result;
}
MATHCALL f16x8 operator^(const f16x8 &A, const f16x8 &B) {
    v128 Result = veorq_u16(v128(A), v128(B));
    return (f16x8)Result;
}
MATHCALL bool IsZero(const f16x8 &Value) {
    return vmaxvq_u16(*(float16x8_t*)((void*)&Value)) == 0;
}

struct v3_reference {
    f16 &x, &y, &z;
    constexpr inline void operator=(const v3 &Value) {
        this->x = Value.x;
        this->y = Value.y;
        this->z = Value.z;
    }
};

struct v3x8 {
    f16x8 x, y, z;

    inline v3x8() { }
    inline v3x8(const f32 &&Value) : x(Value), y(Value), z(Value) { }
    inline v3x8(const f16x8 &Value) : x(Value), y(Value), z(Value) { }
    inline v3x8(const v3 &Value) : x(Value.x), y(Value.y), z(Value.z) { }

    inline v3_reference operator[](uint32_t Index) {
        v3_reference Result = {
            .x = x[Index],
            .y = y[Index],
            .z = z[Index],
        };
        return Result;
    }

    static inline f16x8 Dot(const v3x8 &A, const v3x8 &B);
    static inline f16x8 Length(const v3x8 &A);
    static inline f16x8 LengthSquared(const v3x8 &A);
    static inline v3x8 Normalize(const v3x8 &A);
    static inline v3x8 NormalizeFast(const v3x8 &A);
    static inline void ConditionalMove(v3x8 *A, const v3x8 &B, const f16x8 &MoveMask);
};

MATHCALL v3x8 operator+(const v3x8 &A, const v3x8 &B) {
    v3x8 Result;
    Result.x = A.x + B.x;
    Result.y = A.y + B.y;
    Result.z = A.z + B.z;
    return Result;
}
MATHCALL v3x8 operator-(const v3x8 &A, const v3x8 &B) {
    v3x8 Result;
    Result.x = A.x - B.x;
    Result.y = A.y - B.y;
    Result.z = A.z - B.z;
    return Result;
}
MATHCALL v3x8 operator*(const v3x8 &A, const v3x8 &B) {
    v3x8 Result;
    Result.x = A.x * B.x;
    Result.y = A.y * B.y;
    Result.z = A.z * B.z;
    return Result;
}
MATHCALL v3x8 operator/(const v3x8 &A, const v3x8 &B) {
    v3x8 Result;
    Result.x = A.x / B.x;
    Result.y = A.y / B.y;
    Result.z = A.z / B.z;
    return Result;
}
MATHCALL v3x8 operator&(const v3x8 &A, const v3x8 &B) {
    v3x8 Result;
    Result.x = A.x & B.x;
    Result.y = A.y & B.y;
    Result.z = A.z & B.z;
    return Result;
}

inline f16x8 v3x8::Dot(const v3x8 &A, const v3x8 &B) {
    v3x8 C = A * B;
    f16x8 Result = C.x + C.y + C.z;
    return Result;
}

inline f16x8 v3x8::Length(const v3x8 &A) {
    f16x8 LengthSquared = v3x8::Dot(A, A);
    f16x8 Length = f16x8::SquareRoot(LengthSquared);
    return Length;
}

inline f16x8 v3x8::LengthSquared(const v3x8 &A) {
    v3x8 C = A * A;
    f16x8 Result = C.x + C.y + C.z;
    return Result;
}

constexpr static f32 F32Epsilon = 1e-5f;

inline v3x8 v3x8::Normalize(const v3x8 &Value) {
    f16x8 LengthSquared = v3x8::LengthSquared(Value);

    f16x8 LengthGreaterThanZeroMask = LengthSquared > F32Epsilon;

    f16x8 Length = f16x8::SquareRoot(LengthSquared);
    v3x8 Result = Value / Length;

    v3x8 MaskedResult = Result & LengthGreaterThanZeroMask;
    return MaskedResult;
}

inline void v3x8::ConditionalMove(v3x8 *A, const v3x8 &B, const f16x8 &MoveMask) {
    f16x8::ConditionalMove(&A->x, B.x, MoveMask);
    f16x8::ConditionalMove(&A->y, B.y, MoveMask);
    f16x8::ConditionalMove(&A->z, B.z, MoveMask);
}

inline v3::v3(const f32 &X) {
    v128 Value = vdupq_n_f16(static_cast<__fp16>(X));
    *this = (v3)Value;
}

inline f32 v3::Dot(const v3 &A, const v3 &B) {
    v3 Mul = A * B;
    return Mul.x + Mul.y + Mul.z;
}
inline f32 v3::LengthSquared(const v3 &Value) {
    return v3::Dot(Value, Value);
}

inline v3 v3::Normalize(const v3 &Value) {
    f32 LengthSquared = v3::LengthSquared(Value);

    bool LengthGreaterThanZero = LengthSquared > F32Epsilon;
    v128 Mask = v128::CreateMask(LengthGreaterThanZero);

    f32 Length = __builtin_sqrt(LengthSquared);
    v3 Result = Value / Length;

    v128 MaskedResult = vandq_u16(v128(Result), Mask);
    return (v3)MaskedResult;
}

MATHCALL v3 operator+(const v3 &A, const v3 &B) {
    v128 Result = vaddq_f16(v128(A), v128(B));
    return (v3)Result;
}
MATHCALL v3 operator-(const v3 &A, const v3 &B) {
    v128 Result = vsubq_f16(v128(A), v128(B));
    return (v3)Result;
}
MATHCALL v3 operator*(const v3 &A, const v3 &B) {
    v128 Result = vmulq_f16(v128(A), v128(B));
    return (v3)Result;
}
MATHCALL v3 operator/(const v3 &A, const v3 &B) {
    v128 Result = vdivq_f16(v128(A), v128(B));
    return (v3)Result;
}
MATHCALL v3 operator-(const v3 &A) {
    v128 Result = vnegq_f16(v128(A));
    return (v3)Result;
}

struct sphere_group {
    v3x8 Positions;
    f16x8 Radii;
    v3x8 Color;
    f16x8 Specular;
};
#endif