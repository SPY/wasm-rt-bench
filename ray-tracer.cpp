#include <stdint.h>
#include <iostream>

#ifdef NATIVE
#include "./native.h"
#endif
#ifdef EMSCRIPTEN
#include "./wasm.h"
#endif

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
    const int WIDTH = 2048;
    image img = CreateImage(WIDTH, WIDTH);
    Init();
    Render(img);
    volatile auto iterations = 10;
    volatile uint32_t acc = 0;
    auto start = NOW;
    for (int i = 0; i < iterations; i++) {
        Render(img);
        acc += GetPixel(img, iterations, iterations);
    }
    auto end = NOW;
    std::cout << VERSION << ": It took to run "
        << (end - start)/iterations << "ms on average" << acc << std::endl;
}