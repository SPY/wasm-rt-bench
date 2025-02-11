#ifndef COMMON_H_
#define COMMON_H_

#define MATHCALL static inline

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

#endif