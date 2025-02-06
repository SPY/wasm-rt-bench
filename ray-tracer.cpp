#include <stdint.h>
#include <iostream>
#include <wasm_simd128.h>
#include <emscripten.h>

#define F32X4
// #define F16X8

#define MATHCALL static __attribute__((noinline))

typedef float f32;
typedef double f64;

struct v3 {
    f32 x, y, z;
    f32 _w;
    inline v3() { };

    constexpr inline v3(const f32 &&X) : x(X), y(X), z(X), _w(0) { };
    constexpr inline v3(const f32 &&X, const f32 &&Y, const f32 &&Z) : x(X), y(Y), z(Z), _w(0) { };
    inline v3(const f32 &X);
    inline v3(const f32 &X, const f32 &Y, const f32 &Z);

    static inline f32 Dot(const v3 &A, const v3 &B);
    static inline f32 LengthSquared(const v3 &Value);
    static inline f32 Length(const v3 &Value);
    static inline v3 Normalize(const v3 &Value);
    static inline v3 NormalizeFast(const v3 &Value);
    static inline v3 Cross(const v3 &A, const v3 &B);
} __attribute__((__vector_size__(12), __aligned__(16)));
MATHCALL v3 operator+(const v3 &A, const v3 &B);
MATHCALL v3 operator-(const v3 &A, const v3 &B);
MATHCALL v3 operator*(const v3 &A, const v3 &B);
MATHCALL v3 operator/(const v3 &A, const v3 &B);

MATHCALL void operator+=(v3 &A, const v3 &B) {
    A = A + B;
}
MATHCALL void operator-=(v3 &A, const v3 &B) {
    A = A - B;
}
MATHCALL void operator*=(v3 &A, const v3 &B) {
    A = A * B;
}
MATHCALL void operator/=(v3 &A, const v3 &B) {
    A = A / B;
}

MATHCALL v3 operator-(const v3 &A);

struct image {
    void *Data;
    uint32_t Width, Height;
};

static image CreateImage(uint32_t Width, uint32_t Height) {
    image Result = {0};
    Result.Data = new uint32_t[Width * Height];
    Result.Width = Width;
    Result.Height = Height;
    return Result;
}

struct scalar_sphere {
    v3 Position;
    f32 Radius;
    v3 Color;
    f32 Specular;
};

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
        v128_t v = __builtin_wasm_splat_f16x8(0.0f);
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
    v128 Result = wasm_f16x8_sqrt(v128(A));
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
    Result = wasm_f16x8_pmin(Result, UpperQuad);
    Result = wasm_f16x8_pmin(Result, wasm_i16x8_make(Result.Half8[1], 0, 0, 0, 0, 0, 0, 0));
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

    bool LengthGreaterThanZero = LengthSquared > F32Epsilon;
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


static v3 CameraPosition;

static scalar_sphere ScalarSpheres[8];
static sphere_group SphereGroups[8 / SIMD_WIDTH];

static constexpr inline void InitScalarSpheres(scalar_sphere *ScalarSpheres) {
    ScalarSpheres[0].Position.x = 0.0f;
    ScalarSpheres[0].Position.y = 0.0f;
    ScalarSpheres[0].Position.z = -15.0f;
    ScalarSpheres[0].Radius = 2.0f;
    ScalarSpheres[0].Color.x = 1.0f;
    ScalarSpheres[0].Color.y = 0.125f;
    ScalarSpheres[0].Color.z = 0.0f;
    ScalarSpheres[0].Specular = 1.0f;

    ScalarSpheres[1].Position.x = 0.0f;
    ScalarSpheres[1].Position.y = -130.0f;
    ScalarSpheres[1].Position.z = -15.0f;
    ScalarSpheres[1].Radius = 128.0f;
    ScalarSpheres[1].Color.x = 0.2f;
    ScalarSpheres[1].Color.y = 0.2f;
    ScalarSpheres[1].Color.z = 0.2f;

    ScalarSpheres[2].Position.x = 5.0f;
    ScalarSpheres[2].Position.y = 2.0f;
    ScalarSpheres[2].Position.z = -25.0f;
    ScalarSpheres[2].Radius = 2.0f;
    ScalarSpheres[2].Color.x = 0.0f;
    ScalarSpheres[2].Color.y = 0.0f;
    ScalarSpheres[2].Color.z = 1.0f;

    ScalarSpheres[3].Position.x = 6.0f;
    ScalarSpheres[3].Position.y = 6.0f;
    ScalarSpheres[3].Position.z = -18.0f;
    ScalarSpheres[3].Radius = 2.0f;
    ScalarSpheres[3].Color.x = 0.75f;
    ScalarSpheres[3].Color.y = 0.85f;
    ScalarSpheres[3].Color.z = 0.125f;

    ScalarSpheres[4].Position.x = -7.0f;
    ScalarSpheres[4].Position.y = -0.5f;
    ScalarSpheres[4].Position.z = -25.0f;
    ScalarSpheres[4].Radius = 1.25f;
    ScalarSpheres[4].Color.x = 1.0f;
    ScalarSpheres[4].Color.y = 0.5f;
    ScalarSpheres[4].Color.z = 0.0f;
    ScalarSpheres[4].Specular = 0.95f;
    
    ScalarSpheres[5].Position.x = 7.0f;
    ScalarSpheres[5].Position.y = 6.0f;
    ScalarSpheres[5].Position.z = -30.0f;
    ScalarSpheres[5].Radius = 3.0f;
    ScalarSpheres[5].Color.x = 0.125f;
    ScalarSpheres[5].Color.y = 0.5f;
    ScalarSpheres[5].Color.z = 0.2f;

    ScalarSpheres[6].Position.x = -3.0f;
    ScalarSpheres[6].Position.y = 3.0f;
    ScalarSpheres[6].Position.z = -30.0f;
    ScalarSpheres[6].Radius = 2.5f;
    ScalarSpheres[6].Color.x = 0.25f;
    ScalarSpheres[6].Color.y = 0.15f;
    ScalarSpheres[6].Color.z = 0.12f;

    ScalarSpheres[7].Position.x = -12.0f;
    ScalarSpheres[7].Position.y = 3.0f;
    ScalarSpheres[7].Position.z = -45.0f;
    ScalarSpheres[7].Radius = 1.0f;
    ScalarSpheres[7].Color.x = 0.65f;
    ScalarSpheres[7].Color.y = 0.25f;
    ScalarSpheres[7].Color.z = 0.42f;
}

static void constexpr ConvertScalarSpheresToSIMDSpheres(const scalar_sphere * const Spheres, uint32_t ScalarLength, sphere_group *SIMDSpheres) {
    for (uint32_t i = 0; i < ScalarLength; i += SIMD_WIDTH) {
        sphere_group &SphereGroup = SIMDSpheres[i / SIMD_WIDTH];
        for (uint32_t j = 0; j < SIMD_WIDTH; ++j) {
            f32 R = Spheres[i + j].Radius;
            SphereGroup.Radii[j] = R;
        }
        for (uint32_t j = 0; j < SIMD_WIDTH; ++j) {
            const v3 &Position = Spheres[i + j].Position;
            SphereGroup.Positions[j] = Position;
        }
        for (uint32_t j = 0; j < SIMD_WIDTH; ++j) {
            const v3 &Color = Spheres[i + j].Color;
            SphereGroup.Color[j] = Color;
        }
        for (uint32_t j = 0; j < SIMD_WIDTH; ++j) {
            const f32 &Specular = Spheres[i + j].Specular;
            SphereGroup.Specular[j] = Specular;
        }
    }
}

void Init() {
    InitScalarSpheres(ScalarSpheres);
    ConvertScalarSpheresToSIMDSpheres(ScalarSpheres, 8, SphereGroups);
}

MATHCALL f32 Saturate(f32 Value) {
    if (Value < 0.0f) return 0.0f;
    if (Value > 1.0f) return 1.0f;
    return Value;
}

static inline uint32_t& GetPixel(const image &Image, uint32_t X, uint32_t Y) {
    uint32_t *ImageData = static_cast<uint32_t*>(Image.Data);
    return ImageData[Y * Image.Width + X];
}

__attribute__((noinline)) void Render(const image &Image) {
    v3 CameraZ = v3(0.0f, 0.0f, 1.0f);
    v3 CameraX = v3(1.0f, 0.0f, 0.0f);
    v3 CameraY = v3(0.0f, 1.0f, 0.0f);
    v3 FilmCenter = CameraPosition - CameraZ;

    f32 FilmW = 1.0f;
    f32 FilmH = 1.0f;
    if (Image.Width > Image.Height) {
        FilmH = (f32)Image.Height / (f32)Image.Width;
    } else {
        FilmW = (f32)Image.Width / (f32)Image.Height;
    }

    for (uint32_t y = 0; y < Image.Height; ++y) {
        for (uint32_t x = 0; x < Image.Width; ++x) {
            f32 FilmX = -1.0f + (x * 2.0f) / (f32)Image.Width;
            f32 FilmY = -1.0f + (y * 2.0f) / (f32)Image.Height;

            v3 FilmP = FilmCenter + (FilmX * FilmW * 0.5f * CameraX) + (FilmY * FilmH * 0.5f * CameraY);
            v3 RayOrigin = CameraPosition;
            v3 RayDirection = v3::Normalize(FilmP - RayOrigin);

            v3 DefaultColor = v3(0.0);

#ifdef F32X4
            v3x4 Color = v3x4(DefaultColor);
            constexpr static f32 F32Max = 1e30f;
            f32x4 MinT = F32Max;

            for (const sphere_group &SphereGroup : SphereGroups) {
                v3x4 SphereCenter = SphereGroup.Positions - RayOrigin;
                f32x4 T = v3x4::Dot(SphereCenter, RayDirection);
                v3x4 ProjectedPoint = RayDirection * T;

                f32x4 Radius = SphereGroup.Radii;
                f32x4 DistanceFromCenter = v3x4::Length(SphereCenter - ProjectedPoint);
                f32x4 HitMask = DistanceFromCenter < Radius;
                if (IsZero(HitMask)) continue;
                
                f32x4 X = f32x4::SquareRoot(Radius*Radius - DistanceFromCenter*DistanceFromCenter);
                T = T - X;
                
                f32x4 MinMask = (T < MinT) & (T > 0);
                f32x4 MoveMask = MinMask & HitMask;
                if (IsZero(MoveMask)) continue;

                v3x4 IntersectionPoint = RayDirection * T;
                v3x4 Normal = v3x4::Normalize(IntersectionPoint - SphereCenter);

                f32x4::ConditionalMove(&MinT, T, MoveMask);
                v3x4::ConditionalMove(&Color, (Normal + 1.0f) * 0.5f, MoveMask);
            }

            uint32_t Index = f32x4::HorizontalMinIndex(MinT);
#endif
#ifdef F16X8
            v3x8 Color = v3x8(DefaultColor);
            constexpr static f16 F16Max = 0x7bff;
            f16x8 MinT = F16Max;

            for (const sphere_group &SphereGroup : SphereGroups) {
                v3x8 SphereCenter = SphereGroup.Positions - RayOrigin;
                f16x8 T = v3x8::Dot(SphereCenter, RayDirection);
                v3x8 ProjectedPoint = RayDirection * T;

                f16x8 Radius = SphereGroup.Radii;
                f16x8 DistanceFromCenter = v3x8::Length(SphereCenter - ProjectedPoint);
                f16x8 HitMask = DistanceFromCenter < Radius;
                if (IsZero(HitMask)) continue;
                
                f16x8 X = f16x8::SquareRoot(Radius*Radius - DistanceFromCenter*DistanceFromCenter);
                T = T - X;
                
                f16x8 MinMask = (T < MinT) & (T > 0);
                f16x8 MoveMask = MinMask & HitMask;
                if (IsZero(MoveMask)) continue;

                v3x8 IntersectionPoint = RayDirection * T;
                v3x8 Normal = v3x8::Normalize(IntersectionPoint - SphereCenter);

                f16x8::ConditionalMove(&MinT, T, MoveMask);
                v3x8::ConditionalMove(&Color, (Normal + 1.0f) * 0.5f, MoveMask);
            }

            uint32_t Index = f16x8::HorizontalMinIndex(MinT);
#endif

            v3 OutputColor;
            OutputColor.x = Color.x[Index];
            OutputColor.y = Color.y[Index];
            OutputColor.z = Color.z[Index];

            uint32_t &Pixel = GetPixel(Image, x, y);
            uint32_t r = Saturate(OutputColor.x) * 255.0f;
            uint32_t g = Saturate(OutputColor.y) * 255.0f;
            uint32_t b = Saturate(OutputColor.z) * 255.0f;
            uint32_t a = 255;
            Pixel = (r) | (g << 8) | (b << 16) | (a << 24);
        }
    }
}

int main() {
    image img = CreateImage(640, 480);
    Init();
    Render(img);
    auto start = emscripten_get_now();
    auto iterations = 10;
    for (int i = 0; i < iterations; i++) {
        Render(img);
    }
    auto end = emscripten_get_now();
    volatile uint32_t i = 0;
    for (uint32_t y = 0; y < img.Height; ++y) {
        for (uint32_t x = 0; x < img.Width; ++x) {
            i += GetPixel(img, x, y);
        }
    }
    std::cout << VERSION << ": It took to run "
        << (end - start)/iterations << "ms on average" << i << std::endl;
}