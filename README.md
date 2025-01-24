# FP16/FP32 SIMD Ray Tracer Wasm Benchmark

## Building
Uncomment line 6 or 7 in ray-tracer.cpp to build FP16 or FP32 version.
```
emcc -msimd128 -mfp16 ray-tracer.cpp -o builds/ray-tracer-fp32.js
```
Prebuilt versions available in builds directory.

## Running
To run the benchmark you will need NodeJS with V8 supporting experimental FP16 proposal. NodeJS from version 23 supports it out-of-the-box.

```
node --experimental-wasm-fp16 --turboshaft-wasm --no-liftoff ray-tracer.js
```
