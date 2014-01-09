// System includes
#include <stdio.h>
#include <assert.h>
#include <float.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <helper_functions.h>

#include <EasyBMP.h>
#include <EasyBMP.cpp>

#define BLOCK_DIM 16

typedef struct {
	float avgerages[9];
	float dispersions[9];
} Pair;

/* The device kernel, takes as input the noisy image
 * and outputs the filtered image
 */
template<int BLOCK_SIZE> __global__ void rotatingMaskCUDA(Pair * filtered,
		unsigned char * img, int rows, int cols) {
	__shared__ unsigned char input_img[BLOCK_SIZE][BLOCK_SIZE];

	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * BLOCK_SIZE + ty;
	int col = bx * BLOCK_SIZE + tx;

	/* Overlapping the tiles */
	row -= 2 * by;
	col -= 2 * bx;

	if (row < rows && col < cols)
		input_img[ty][tx] = img[cols * row + col];

	__syncthreads();

	if (row < rows && col < cols) {
		float tmp_c = cols;
		float tmp_r = rows;
		int numberOfBlocksx = (int) ceil(tmp_c / (BLOCK_SIZE - 2));
		int numberOfBlocksy = (int) ceil(tmp_r / (BLOCK_SIZE - 2));

		// Check if this pixel should compute the average and the dispersion
		if ((bx < numberOfBlocksx - 1
				|| (bx == numberOfBlocksx - 1
						&& (tx < cols - bx * (BLOCK_SIZE - 2) - 2)))
				&& (by < numberOfBlocksy - 1
						|| (by == numberOfBlocksy - 1
								&& (ty < rows - by * (BLOCK_SIZE - 2) - 2)))) {
			if (tx < BLOCK_SIZE - 2 && ty < BLOCK_SIZE - 2) {

				/* Calculate the average for the mask
				 * with the current pixel positioned at
				 * the upper-left corner */

				float sum = 0;
				for (int i = 0; i < 3; i++) {
					for (int j = 0; j < 3; j++) {

						int tmp_col = tx + j;
						int tmp_row = ty + i;

						sum += input_img[tmp_row][tmp_col];
					}
				}

				float average = sum / 9;

				/* Calculate the dispersion for the mask
				 * with the current pixel positioned at
				 * the upper-left corner */
				float dispersion = 0;
				for (int i = 0; i < 3; i++) {
					for (int j = 0; j < 3; j++) {
						int tmp_col = tx + j;
						int tmp_row = ty + i;
						dispersion += (input_img[tmp_row][tmp_col] - average)
								* (input_img[tmp_row][tmp_col] - average);
					}
				}

				/* Assign the value of the calculated mask to each pixel
				 * i.e. the current mask will be added to index 0
				 * of the Upper left pixel, and index 1 of the
				 * Upper left-but-one pixel, and so on.
				 */
				int index = 0;
				for (int i = 0; i < 3; i++) {
					for (int j = 0; j < 3; j++) {
						int tmp_col = col + j;
						int tmp_row = row + i;

						filtered[tmp_col + tmp_row * cols].avgerages[index] =
								average;
						filtered[tmp_col + tmp_row * cols].dispersions[index] =
								dispersion;
						index++;
					}
				}
			}
		}
	}
}

template<int BLOCK_SIZE> __global__ void getArrayMin(unsigned char * output_img,
		Pair * input_img, int rows, int cols) {
	
	/* Calculate the index of the 2d array */
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * BLOCK_SIZE + ty;
	int col = bx * BLOCK_SIZE + tx;

	float min = FLT_MAX;
	int min_index = 0;
	float  *dispersions = input_img[col + row * cols].dispersions;
	for (int i = 0; i < 9; i++) {
		float tmp = dispersions[i];

		if (tmp < min && tmp >= 0) {
			min = tmp;
			min_index = i;
		}
	}
	output_img[col + row * cols] = input_img[col + row * cols].avgerages[min_index];
}

unsigned char * init(unsigned char * img, int rows, int cols) {

	// Device input image and filtered image
	unsigned char *d_img, *filtered_img, *d_filtered;

	filtered_img = (unsigned char *) malloc(sizeof(unsigned char) * cols * rows);

	// The temporary matrix holding the averages
	// and dispersions for all 9 mask positions
	Pair * d_tmp;

	// Allocate and copy input image to device
	int size = rows * cols * sizeof(unsigned char);

	cudaMalloc((void**) &d_img, size);
	cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);

	// Allocate memory for output image
	cudaMalloc((void**) &d_filtered, size);

	// Allocate memory for tmp matrix
	int size_pair = rows * cols * sizeof(Pair);
	cudaMalloc((void**) &d_tmp, size_pair);

	// Define grid and block dimensions
	dim3 block(BLOCK_DIM, BLOCK_DIM, 1);

	dim3 grid((int) ceil((cols * 1.0) / (BLOCK_DIM - 2)),
			(int) ceil((rows * 1.0) / (BLOCK_DIM - 2)), 1);

	// Kernel invocations
	rotatingMaskCUDA<BLOCK_DIM> <<<grid, block>>>(d_tmp, d_img, rows, cols);

	dim3 grid2((int) ceil((cols * 1.0) / BLOCK_DIM),
			(int) ceil((rows * 1.0) / BLOCK_DIM), 1);

	getArrayMin<BLOCK_DIM> <<<grid2, block>>>(d_filtered, d_tmp, rows, cols);

	// Copy the filtered image to the host memory
	cudaMemcpy(filtered_img, d_filtered, size, cudaMemcpyDeviceToHost);

	// Free allocated memory
	cudaFree(d_img);
	cudaFree(d_tmp);
	cudaFree(d_filtered);

	return filtered_img;
}

void checkCuda(int argc, char **argv) {
	printf(
			"[Rotating mask technique for image filtering Using CUDA] - Starting...\n");

	// By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
	int devID = 0;

	if (checkCmdLineFlag(argc, (const char **) argv, "device")) {
		devID = getCmdLineArgumentInt(argc, (const char **) argv, "device");
		cudaSetDevice(devID);
	}

	cudaError_t error;
	cudaDeviceProp deviceProp;
	error = cudaGetDevice(&devID);

	if (error != cudaSuccess) {
		printf("cudaGetDevice returned error code %d, line(%d)\n", error,
				__LINE__);
	}

	error = cudaGetDeviceProperties(&deviceProp, devID);

	if (deviceProp.computeMode == cudaComputeModeProhibited) {
		fprintf(stderr,
				"Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
		exit(EXIT_SUCCESS);
	}

	if (error != cudaSuccess) {
		printf("cudaGetDeviceProperties returned error code %d, line(%d)\n",
				error, __LINE__);
	} else {
		printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID,
				deviceProp.name, deviceProp.major, deviceProp.minor);
	}
}

int main(int argc, char **argv) {

	/* Check if CUDA is available */
	checkCuda(argc, argv);

	BMP imgIn, imgOut;

	imgIn.ReadFromFile("../../test images/lena_noise.bmp");

	int width = imgIn.TellWidth();
	int height = imgIn.TellHeight();

	unsigned char *pixelsIn_r, *pixelsIn_g, *pixelsIn_b, *pixelsIn_a;
	unsigned char *pixelsOut_r, *pixelsOut_g, *pixelsOut_b, *pixelsOut_a;

	// read the 4 channels R, G, B and A from the BMP object
	pixelsIn_r = imgIn.getPixelArray(Red);
	pixelsIn_g = imgIn.getPixelArray(Green);
	pixelsIn_b = imgIn.getPixelArray(Blue);
	pixelsIn_a = imgIn.getPixelArray(Alpha);

	/************************************** Timing **************************************/
	// cudaError_t error;

 // 	// Allocate CUDA events that we'll use for timing
 //    cudaEvent_t start;
 //    error = cudaEventCreate(&start);

 //    if (error != cudaSuccess)
 //    {
 //        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
 //        exit(EXIT_FAILURE);
 //    }

 //    cudaEvent_t stop;
 //    error = cudaEventCreate(&stop);

 //    if (error != cudaSuccess)
 //    {
 //        fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
 //        exit(EXIT_FAILURE);
 //    }

 //    // Record the start event
 //    error = cudaEventRecord(start, NULL);

 //    if (error != cudaSuccess)
 //    {
 //        fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
 //        exit(EXIT_FAILURE);
 //    }
    /************************************** Timing **************************************/

	// compute the corresponding 4 channels after performing filtering
	pixelsOut_r = init(pixelsIn_r, height, width);
	pixelsOut_g = init(pixelsIn_g, height, width);
	pixelsOut_b = init(pixelsIn_b, height, width);
	pixelsOut_a = init(pixelsIn_a, height, width);

	/************************************** Timing **************************************/
	// // Record the stop event
 //    error = cudaEventRecord(stop, NULL);

 //    if (error != cudaSuccess)
 //    {
 //        fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
 //        exit(EXIT_FAILURE);
 //    }

 //    // Wait for the stop event to complete
 //    error = cudaEventSynchronize(stop);

 //    if (error != cudaSuccess)
 //    {
 //        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
 //        exit(EXIT_FAILURE);
 //    }

 //    float msecTotal = 0.0f;
 //    error = cudaEventElapsedTime(&msecTotal, start, stop);

 //    if (error != cudaSuccess)
 //    {
 //        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
 //        exit(EXIT_FAILURE);
 //    }

    /************************************** Timing **************************************/

	// write the computed channels to a bmp image file
	imgOut.fromPixelArrays(pixelsOut_r, pixelsOut_g, pixelsOut_b, pixelsOut_a,
			width, height);
	imgOut.WriteToFile("../../output images/lena_noise_filtered.bmp");

	return 0;
}