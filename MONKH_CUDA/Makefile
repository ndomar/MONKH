INCLUDES  := -I/Developer/NVIDIA/CUDA-5.5/samples/common/inc

all: clean build

build: rotatingMask
	   
rotatingMask:
	/Developer/NVIDIA/CUDA-5.5/bin/nvcc -arch=sm_20 -o rotatingMask rotatingMask_Granularity_16.cu -I /Developer/NVIDIA/CUDA-5.5/samples/common/inc -I /Users/KareemYousrii/cuda-workspace/MONKH/include -I ../precompile -I ../include

run:
	./rotatingMask

clean:
	rm -f rotatingMask rotatingMask.o