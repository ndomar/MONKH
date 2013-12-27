// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <helper_functions.h>

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>

typedef struct {
	int avgerages[9];
	int dispersions[9];
} Pair;

/* The device kernel, takes as input the noisy image
 * and outputs the filtered image
 */
template<int BLOCK_SIZE> __global__ void rotatingMaskCUDA(Pair * filtered,
		unsigned char * img, int n, int m) {
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

	if (row < m && col < n) {

		input_img[ty][tx] = img[n * row + col];

		__syncthreads();

		int numberOfBlocksx = (int) ceil((n * 1.0) / (BLOCK_SIZE - 2));
		int numberOfBlocksy = (int) ceil((m * 1.0) / (BLOCK_SIZE - 2));
		// Check if this pixel should compute the average and the dispersion
		if ((bx < numberOfBlocksx - 1
				|| (bx == numberOfBlocksx - 1
						&& (tx < n - bx * (BLOCK_SIZE - 2) - 2)))
				&& (by < numberOfBlocksy - 1
						|| (by == numberOfBlocksy - 1
								&& (ty < m - by * (BLOCK_SIZE - 2) - 2))))
			if (tx < 6 && ty < 6) {

				/* Calculate the average for the mask
				 * with the current pixel positioned at
				 * the upper-left corner */

				int sum = 0;
				for (int i = 0; i < 3; i++) {
					for (int j = 0; j < 3; j++) {

						int tmp_col = tx + j;
						int tmp_row = ty + i;

						sum += input_img[tmp_row][tmp_col];
					}
				}

				int average = sum / 9;

				/* Calculate the dispersion for the mask
				 * with the current pixel positioned at
				 * the upper-left corner */
				int dispersion = 0;
				for (int i = 0; i < 3; i++) {
					for (int j = 0; j < 3; j++) {
						int tmp_col = tx + j;
						int tmp_row = ty + i;
						dispersion += powf(
								input_img[tmp_row][tmp_col] - average, 2);
					}
				}

				dispersion /= 9;

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

						filtered[tmp_col + tmp_row * n].avgerages[index] =
								average;
						filtered[tmp_col + tmp_row * n].dispersions[index] =
								dispersion;
						index++;
					}
				}
			}
	}
}

template<int BLOCK_SIZE> __global__ void getArrayMin(unsigned char * output_img,
		Pair * input_img, int n, int m) {
	/* Calculate the index of the 2d array */
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * BLOCK_SIZE + ty;
	int col = bx * BLOCK_SIZE + tx;

	int min = 9999;
	int min_index = 0;
	for (int i = 0; i < 9; i++) {
		int tmp = input_img[col + row * n].dispersions[i];

		if (tmp < min && tmp >= 0) {
			min = tmp;
			min_index = i;
		}
	}

	output_img[col + row * n] = input_img[col + row * n].avgerages[min_index];
}

void init(int block_dim, unsigned char * img, int rows, int cols,
		unsigned char * filtered_img) {

	// Device input image and filtered image
	unsigned char *d_img, *d_filtered;

	// The temporary matrix holding the averages
	// and dispersions for all 9 mask positions
	Pair *d_tmp;

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
	dim3 block(block_dim, block_dim, 1);

	dim3 grid((int) ceil((rows * 1.0) / (block_dim - 2)),
			(int) ceil((cols * 1.0) / (block_dim - 2)), 1);

	// Kernel invocations
	rotatingMaskCUDA<8> <<<grid, block>>>(d_tmp, d_img, rows, cols);

	dim3 grid2((int) ceil((rows * 1.0) / block_dim),
			(int) ceil((cols * 1.0) / block_dim), 1);
	getArrayMin<8> <<<grid2, block>>>(d_filtered, d_tmp, rows, cols);

	// Copy the filtered image to the host memory
	cudaMemcpy(filtered_img, d_filtered, size, cudaMemcpyDeviceToHost);

	// Free allocated memory
	cudaFree(d_img);
	cudaFree(d_tmp);
	cudaFree(d_filtered);
}

int main(int argc, char **argv) {

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

	// Random number generator
	srand(time(NULL));

	// Size of input and output images
	unsigned int size = 16 * 16 * sizeof(unsigned char);

	unsigned char *matrix = (unsigned char *) malloc(size);
	unsigned char *filtered_img = (unsigned char *) malloc(size);

	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++) {
			matrix[j + i * 16] = rand() % 256;
		}
	}

	init(8, matrix, 16, 16, filtered_img);

	for (int i = 0; i < 15; i++) {
		printf("{");
		for (int j = 0; j < 15; j++) {
			printf("%d, ", matrix[j + i * 16]);
		}
		printf("%d", matrix[15 + i * 16]);
		printf("},");
		printf("\n");
	}
	printf("{");
	for (int j = 0; j < 15; j++) {
		printf("%d, ", matrix[j + 15 * 16]);
	}
	printf("%d", matrix[15 + 15 * 16]);
	printf("}");
	printf("\n");

	printf("Filtered\n");
	printf("\n");

	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++) {
			printf("%d ", filtered_img[j + i * 16]);
		}
		printf("\n");
	}

	return 1;
}
