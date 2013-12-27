INCLUDES  := -I/Developer/NVIDIA/CUDA-5.5/samples/common/inc

all: build

build: rotatingMask

rotatingMask:
	/Developer/NVIDIA/CUDA-5.5/bin/nvcc -I/Developer/NVIDIA/CUDA-5.5/samples/common/inc -arch=sm_20 rotatingMask.cu -o rotatingMask

run: build
	./rotatingMask

clean:
	rm -f rotatingMask rotatingMask.o