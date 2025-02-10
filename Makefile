.PHONY : all
all:
	emcc -msimd128 -mrelaxed-simd -O2 -sENVIRONMENT=shell -DF32X4 ./ray-tracer.cpp -o builds/ray-tracer-fp32.js
	emcc -msimd128 -mfp16 -mrelaxed-simd -O2 -sENVIRONMENT=shell -DF16X8 ./ray-tracer.cpp -o builds/ray-tracer-fp16.js
	clang -std=c++20 -lstdc++ -O2 -D F32X4 -D NATIVE ./ray-tracer.cpp -o builds/ray-tracer-fp32
	clang -std=c++20 -lstdc++ -O2 -D F16X8 -D NATIVE ./ray-tracer.cpp -o builds/ray-tracer-fp16

bench:
	./builds/ray-tracer-fp32
	./builds/ray-tracer-fp16
	cd builds && ../../v8/out/arm64.debug/d8 --experimental-wasm-fp16 --no-liftoff ray-tracer-fp32.js
	cd builds && ../../v8/out/arm64.debug/d8 --experimental-wasm-fp16 --no-liftoff ray-tracer-fp16.js