# FP16/FP32 SIMD Ray Tracer Wasm Benchmark

## Building
Uncomment line 6 or 7 in ray-tracer.cpp to build FP16 or FP32 version.
```
emcc -msimd128 -mfp16 ray-tracer.cpp -o builds/ray-tracer-fp32.js
```
To build D8 compatible version:
```
emcc -msimd128 -mfp16 ray-tracer.cpp -o ray-tracer.js -sENVIRONMENT=shell
```
Prebuilt versions available in builds directory.

## Running
To run the benchmark you will need NodeJS with V8 supporting experimental FP16 proposal or d8. NodeJS from version 23 supports it out-of-the-box.

```
node --experimental-wasm-fp16 --no-liftoff ray-tracer.js
```

```
d8 --experimental-wasm-fp16 --no-liftoff ray-tracer.js
```

## Results

|                     | F32x4     | F16x8     |
| ------------------- | --------- | --------- |
| Arm64(M1 Pro, 2021) | 701.363ms | 622.229ms |
| X64(iMac, i7, 2020) | 1274.7ms  | 1580.25ms |