#include <wasm_simd128.h>
#include <emscripten.h>

#include "common.h"

#define NOW emscripten_get_now()

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
    v128_t Register;

    inline v128() { Register = wasm_f32x4_const_splat(0.0f); };
    inline v128(f32 Value) : Float(Value) { };
    inline v128(const v3 &Value) : Vector3(Value) { };
    inline v128(const f32x4 &Value) : Float4(Value) { };
    inline v128(const v128_t &SIMDLane) : Register(SIMDLane) { };

    explicit operator f32() const { return Float; }
    explicit operator v3() const { return Vector3; }
    explicit operator f32x4() const { return Float4; }
    operator v128_t() const { return Register; }

    MATHCALL v128 CreateMask(bool Value) {
        v128 Result = wasm_u32x4_splat((uint32_t)Value * -1);
        return Result;
    }
};

inline f32x4 f32x4::SquareRoot(const f32x4 &A) {
    v128 Result = wasm_f32x4_sqrt(v128(A));
    return (f32x4)Result;
}
inline f32x4 f32x4::Min(const f32x4 &A, const f32x4 &B) {
    v128 Result = wasm_f32x4_min(v128(A), v128(B));
    return (f32x4)Result;
}
inline f32x4 f32x4::Max(const f32x4 &A, const f32x4 &B) {
    v128 Result = wasm_f32x4_max(v128(A), v128(B));
    return (f32x4)Result;
}
inline void f32x4::ConditionalMove(f32x4 *A, const f32x4 &B, const f32x4 &MoveMask) {
    v128 Result = wasm_v128_bitselect(v128(B), v128(*A), v128(MoveMask));
    *A = (f32x4)Result;
}

inline f32 f32x4::HorizontalMin(const f32x4 &Value) {
    v128 UpperHalf = wasm_f32x4_make(Value[2], Value[3], 0, 0);
    v128 Result = wasm_f32x4_min(v128(Value), UpperHalf);
    Result = wasm_f32x4_min(Result, wasm_f32x4_make(Result.Float4[1], 0, 0, 0));
    return (f32)Result;
}
MATHCALL f32x4 operator==(const f32x4 &A, const f32x4 &B) {
    v128 Result = wasm_f32x4_eq(v128(A), v128(B));
    return (f32x4)Result;
}
inline uint32_t f32x4::HorizontalMinIndex(const f32x4 &Value) {
    f32 MinValue = f32x4::HorizontalMin(Value);
    v128 Comparison = Value == f32x4(MinValue);
    uint32_t MoveMask = wasm_i32x4_bitmask(Comparison);
    uint32_t Result = __builtin_ctz(MoveMask);
    return Result;
}

MATHCALL f32x4 operator+(const f32x4 &A, const f32x4 &B) {
    v128 Result = wasm_f32x4_add(v128(A), v128(B));
    return (f32x4)Result;
}
MATHCALL f32x4 operator-(const f32x4 &A, const f32x4 &B) {
    v128 Result = wasm_f32x4_sub(v128(A), v128(B));
    return (f32x4)Result;
}
MATHCALL f32x4 operator*(const f32x4 &A, const f32x4 &B) {
    v128 Result = wasm_f32x4_mul(v128(A), v128(B));
    return (f32x4)Result;
}
MATHCALL f32x4 operator/(const f32x4 &A, const f32x4 &B) {
    v128 Result = wasm_f32x4_div(v128(A), v128(B));
    return (f32x4)Result;
}
MATHCALL f32x4 operator!=(const f32x4 &A, const f32x4 &B) {
    v128 Result = wasm_f32x4_ne(v128(A), v128(B));
    return (f32x4)Result;
}
MATHCALL f32x4 operator>(const f32x4 &A, const f32x4 &B) {
    v128 Result = wasm_f32x4_gt(v128(A), v128(B));
    return (f32x4)Result;
}
MATHCALL f32x4 operator<(const f32x4 &A, const f32x4 &B) {
    v128 Result = wasm_f32x4_lt(v128(A), v128(B));
    return (f32x4)Result;
}
MATHCALL f32x4 operator&(const f32x4 &A, const f32x4 &B) {
    v128 Result = wasm_v128_and(v128(A), v128(B));
    return (f32x4)Result;
}
MATHCALL f32x4 operator|(const f32x4 &A, const f32x4 &B) {
    v128 Result = wasm_v128_or(v128(A), v128(B));
    return (f32x4)Result;
}
MATHCALL f32x4 operator^(const f32x4 &A, const f32x4 &B) {
    v128 Result = wasm_v128_xor(v128(A), v128(B));
    return (f32x4)Result;
}
MATHCALL bool IsZero(const f32x4 &Value) {
    v128 Zero = wasm_f32x4_const_splat(0.0f);
    v128 ComparisonResult = wasm_f32x4_ne(v128(Value), Zero);
    bool Result = wasm_v128_any_true(ComparisonResult);
    return Result == 0;
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
    v128 Value = wasm_f32x4_splat(X);
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

    v128 MaskedResult = wasm_v128_and(v128(Result), Mask);
    return (v3)MaskedResult;
}

MATHCALL v3 operator+(const v3 &A, const v3 &B) {
    v128 Result = wasm_f32x4_add(v128(A), v128(B));
    return (v3)Result;
}
MATHCALL v3 operator-(const v3 &A, const v3 &B) {
    v128 Result = wasm_f32x4_sub(v128(A), v128(B));
    return (v3)Result;
}
MATHCALL v3 operator*(const v3 &A, const v3 &B) {
    v128 Result = wasm_f32x4_mul(v128(A), v128(B));
    return (v3)Result;
}
MATHCALL v3 operator/(const v3 &A, const v3 &B) {
    v128 Result = wasm_f32x4_div(v128(A), v128(B));
    return (v3)Result;
}
MATHCALL v3 operator-(const v3 &A) {
    v128 Result = wasm_f32x4_neg(v128(A));
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

using f16 = uint16_t;

struct f16x8 {
    f16 Value[8];

    inline f16x8() { }
    inline f16x8(f32 V) {
        v128_t v = wasm_f16x8_splat(V);
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
    v128_t Register;

    inline v128() { Register = wasm_f16x8_splat(0.0f); };
    inline v128(f32 Value) : Float(Value) { };
    inline v128(const v3 &Value) : Vector3(Value) { };
    inline v128(const f16x8 &Value) : Half8(Value) { };
    inline v128(const v128_t &SIMDLane) : Register(SIMDLane) { };

    explicit operator f32() const { return Float; }
    explicit operator v3() const { return Vector3; }
    explicit operator f16x8() const { return Half8; }
    operator v128_t() const { return Register; }

    MATHCALL v128 CreateMask(bool Value) {
        v128 Result = wasm_u16x8_splat((uint32_t)Value * -1);
        return Result;
    }
};

inline f16x8 f16x8::SquareRoot(const f16x8 &A) {
    v128 Result = wasm_f16x8_sqrt(v128(0));
    return (f16x8)Result;
}
inline f16x8 f16x8::Min(const f16x8 &A, const f16x8 &B) {
    v128 Result = wasm_f16x8_min(v128(A), v128(B));
    return (f16x8)Result;
}
inline f16x8 f16x8::Max(const f16x8 &A, const f16x8 &B) {
    v128 Result = wasm_f16x8_max(v128(A), v128(B));
    return (f16x8)Result;
}
inline void f16x8::ConditionalMove(f16x8 *A, const f16x8 &B, const f16x8 &MoveMask) {
    v128 Result = wasm_v128_bitselect(v128(B), v128(*A), v128(MoveMask));
    *A = (f16x8)Result;
}

inline f32 f16x8::HorizontalMin(const f16x8 &Value) {
    v128 UpperHalf = wasm_u16x8_make(Value[4], Value[5], Value[6], Value[7], 0, 0, 0, 0);
    v128 Result = wasm_f16x8_pmin(v128(Value), UpperHalf);
    v128 UpperQuad = wasm_u16x8_make(Result.Half8[2], Result.Half8[3], 0, 0, 0, 0, 0, 0);
    // Result = wasm_f16x8_pmin(Result, UpperQuad);
    // Result = wasm_f16x8_pmin(Result, wasm_i16x8_make(Result.Half8[1], 0, 0, 0, 0, 0, 0, 0));
    return wasm_f16x8_extract_lane(Result, 0);
}
MATHCALL f16x8 operator==(const f16x8 &A, const f16x8 &B) {
    v128 Result = wasm_f16x8_eq(v128(A), v128(B));
    return (f16x8)Result;
}
inline uint32_t f16x8::HorizontalMinIndex(const f16x8 &Value) {
    f32 MinValue = f16x8::HorizontalMin(Value);
    v128 Comparison = Value == f16x8(MinValue);
    uint32_t MoveMask = wasm_i16x8_bitmask(Comparison);
    uint32_t Result = __builtin_ctz(MoveMask);
    return Result;
}

MATHCALL f16x8 operator+(const f16x8 &A, const f16x8 &B) {
    v128 Result = wasm_f16x8_add(v128(A), v128(B));
    return (f16x8)Result;
}
MATHCALL f16x8 operator-(const f16x8 &A, const f16x8 &B) {
    v128 Result = wasm_f16x8_sub(v128(A), v128(B));
    return (f16x8)Result;
}
MATHCALL f16x8 operator*(const f16x8 &A, const f16x8 &B) {
    v128 Result = wasm_f16x8_mul(v128(A), v128(B));
    return (f16x8)Result;
}
MATHCALL f16x8 operator/(const f16x8 &A, const f16x8 &B) {
    v128 Result = wasm_f16x8_div(v128(A), v128(B));
    return (f16x8)Result;
}
MATHCALL f16x8 operator!=(const f16x8 &A, const f16x8 &B) {
    v128 Result = wasm_f16x8_ne(v128(A), v128(B));
    return (f16x8)Result;
}
MATHCALL f16x8 operator>(const f16x8 &A, const f16x8 &B) {
    v128 Result = wasm_f16x8_gt(v128(A), v128(B));
    return (f16x8)Result;
}
MATHCALL f16x8 operator<(const f16x8 &A, const f16x8 &B) {
    v128 Result = wasm_f16x8_lt(v128(A), v128(B));
    return (f16x8)Result;
}
MATHCALL f16x8 operator&(const f16x8 &A, const f16x8 &B) {
    v128 Result = wasm_v128_and(v128(A), v128(B));
    return (f16x8)Result;
}
MATHCALL f16x8 operator|(const f16x8 &A, const f16x8 &B) {
    v128 Result = wasm_v128_or(v128(A), v128(B));
    return (f16x8)Result;
}
MATHCALL f16x8 operator^(const f16x8 &A, const f16x8 &B) {
    v128 Result = wasm_v128_xor(v128(A), v128(B));
    return (f16x8)Result;
}
MATHCALL bool IsZero(const f16x8 &Value) {
    v128 Zero = wasm_f16x8_splat(0.0f);
    v128 ComparisonResult = wasm_f16x8_ne(v128(Value), Zero);
    bool Result = wasm_v128_any_true(ComparisonResult);
    return Result == 0;
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

constexpr static f32 F16Epsilon = 1e-3f;

inline v3x8 v3x8::Normalize(const v3x8 &Value) {
    f16x8 LengthSquared = v3x8::LengthSquared(Value);

    f16x8 LengthGreaterThanZeroMask = LengthSquared > F16Epsilon;

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
    v128 Value = wasm_f16x8_splat(X);
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

    bool LengthGreaterThanZero = LengthSquared > F16Epsilon;
    v128 Mask = v128::CreateMask(LengthGreaterThanZero);

    f32 Length = __builtin_sqrt(LengthSquared);
    v3 Result = Value / Length;

    v128 MaskedResult = wasm_v128_and(v128(Result), Mask);
    return (v3)MaskedResult;
}

MATHCALL v3 operator+(const v3 &A, const v3 &B) {
    v128 Result = wasm_f16x8_add(v128(A), v128(B));
    return (v3)Result;
}
MATHCALL v3 operator-(const v3 &A, const v3 &B) {
    v128 Result = wasm_f16x8_sub(v128(A), v128(B));
    return (v3)Result;
}
MATHCALL v3 operator*(const v3 &A, const v3 &B) {
    v128 Result = wasm_f16x8_mul(v128(A), v128(B));
    return (v3)Result;
}
MATHCALL v3 operator/(const v3 &A, const v3 &B) {
    v128 Result = wasm_f16x8_div(v128(A), v128(B));
    return (v3)Result;
}
MATHCALL v3 operator-(const v3 &A) {
    v128 Result = wasm_f16x8_neg(v128(A));
    return (v3)Result;
}

struct sphere_group {
    v3x8 Positions;
    f16x8 Radii;
    v3x8 Color;
    f16x8 Specular;
};
#endif