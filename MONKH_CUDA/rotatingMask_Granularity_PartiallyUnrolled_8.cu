// System includes
#include <stdio.h>
#include <assert.h>
#include <float.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <helper_functions.h>

#include <EasyBMP.h>
#include <EasyBMP.cpp>

#define BLOCK_DIM 8

typedef struct {
	float avgerages[9];
	float dispersions[9];
} Pair;

/* The device kernel, takes as input the noisy image
 * and outputs the filtered image
 */
template<int BLOCK_SIZE> __global__ void rotatingMaskCUDA(Pair * filtered_r, Pair * filtered_g, Pair * filtered_b, Pair * filtered_a,
		unsigned char * img_r, unsigned char * img_g, unsigned char * img_b, unsigned char * img_a, 
		int rows, int cols) {
	__shared__ unsigned char input_img_r[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ unsigned char input_img_g[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ unsigned char input_img_b[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ unsigned char input_img_a[BLOCK_SIZE][BLOCK_SIZE];

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

	if (row < rows && col < cols) {
		input_img_r[ty][tx] = img_r[cols * row + col];
		input_img_g[ty][tx] = img_g[cols * row + col];
		input_img_b[ty][tx] = img_b[cols * row + col];
		input_img_a[ty][tx] = img_a[cols * row + col];
	}

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

				//---------------------RED CHANNEL---------------------

				/* Calculate the average for the mask
				 * with the current pixel positioned at
				 * the upper-left corner */

				float sum = 0;
				
				for (int i = 0; i < 3; i++) {
					int tmp_row = ty + i;

					sum += input_img_r[tmp_row][tx + 0];
					sum += input_img_r[tmp_row][tx + 1];
					sum += input_img_r[tmp_row][tx + 2];
				}

				float average = sum / 9;

				/* Calculate the dispersion for the mask
				 * with the current pixel positioned at
				 * the upper-left corner */
				float dispersion = 0;

				for (int i = 0; i < 3; i++) {
					int tmp_row = ty + i;
					dispersion += (input_img_r[tmp_row][tx + 0] - average)
							* (input_img_r[tmp_row][tx + 0] - average);

					dispersion += (input_img_r[tmp_row][tx + 1] - average)
							* (input_img_r[tmp_row][tx + 1] - average);

					dispersion += (input_img_r[tmp_row][tx + 2] - average)
							* (input_img_r[tmp_row][tx + 2] - average);
				}

//				dispersion /= 9;

				/* Assign the value of the calculated mask to each pixel
				 * i.e. the current mask will be added to index 0
				 * of the Upper left pixel, and index 1 of the
				 * Upper left-but-one pixel, and so on.
				 */
				int index = 0;

				for (int i = 0; i < 3; i++) {
					int tmp_row = row + i;

					filtered_r[col + 0 + tmp_row * cols].avgerages[index] =
							average;
					filtered_r[col + 0 + tmp_row * cols].dispersions[index] =
							dispersion;
					index++;

					filtered_r[col + 1 + tmp_row * cols].avgerages[index] =
							average;
					filtered_r[col + 1 + tmp_row * cols].dispersions[index] =
							dispersion;
					index++;

					filtered_r[col + 2 + tmp_row * cols].avgerages[index] =
							average;
					filtered_r[col + 2 + tmp_row * cols].dispersions[index] =
							dispersion;
					index++;
				}

				//---------------------GREEN CHANNEL---------------------

				/* Calculate the average for the mask
				 * with the current pixel positioned at
				 * the upper-left corner */

				sum = 0;

				for (int i = 0; i < 3; i++) {
					int tmp_row = ty + i;

					sum += input_img_g[tmp_row][tx + 0];
					sum += input_img_g[tmp_row][tx + 1];
					sum += input_img_g[tmp_row][tx + 2];
				}

				average = sum / 9;

				/* Calculate the dispersion for the mask
				 * with the current pixel positioned at
				 * the upper-left corner */
				dispersion = 0;

				for (int i = 0; i < 3; i++) {
					int tmp_row = ty + i;
					dispersion += (input_img_g[tmp_row][tx + 0] - average)
							* (input_img_g[tmp_row][tx + 0] - average);

					dispersion += (input_img_g[tmp_row][tx + 1] - average)
							* (input_img_g[tmp_row][tx + 1] - average);

					dispersion += (input_img_g[tmp_row][tx + 2] - average)
							* (input_img_g[tmp_row][tx + 2] - average);
				}

				/* Assign the value of the calculated mask to each pixel
				 * i.e. the current mask will be added to index 0
				 * of the Upper left pixel, and index 1 of the
				 * Upper left-but-one pixel, and so on.
				 */
				index = 0;

				for (int i = 0; i < 3; i++) {
					int tmp_row = row + i;

					filtered_g[col + 0 + tmp_row * cols].avgerages[index] =
							average;
					filtered_g[col + 0 + tmp_row * cols].dispersions[index] =
							dispersion;
					index++;

					filtered_g[col + 1 + tmp_row * cols].avgerages[index] =
							average;
					filtered_g[col + 1 + tmp_row * cols].dispersions[index] =
							dispersion;
					index++;

					filtered_g[col + 2 + tmp_row * cols].avgerages[index] =
							average;
					filtered_g[col + 2 + tmp_row * cols].dispersions[index] =
							dispersion;
					index++;
				}

				//---------------------BLUE CHANNEL---------------------

				/* Calculate the average for the mask
				 * with the current pixel positioned at
				 * the upper-left corner */

				sum = 0;

				for (int i = 0; i < 3; i++) {
					int tmp_row = ty + i;

					sum += input_img_b[tmp_row][tx + 0];
					sum += input_img_b[tmp_row][tx + 1];
					sum += input_img_b[tmp_row][tx + 2];
				}

				average = sum / 9;

				/* Calculate the dispersion for the mask
				 * with the current pixel positioned at
				 * the upper-left corner */
				dispersion = 0;

				for (int i = 0; i < 3; i++) {
					int tmp_row = ty + i;

					dispersion += (input_img_b[tmp_row][tx + 0] - average)
							* (input_img_b[tmp_row][tx + 0] - average);

					dispersion += (input_img_b[tmp_row][tx + 1] - average)
							* (input_img_b[tmp_row][tx + 1] - average);

					dispersion += (input_img_b[tmp_row][tx + 2] - average)
							* (input_img_b[tmp_row][tx + 2] - average);
				}
				/* Assign the value of the calculated mask to each pixel
				 * i.e. the current mask will be added to index 0
				 * of the Upper left pixel, and index 1 of the
				 * Upper left-but-one pixel, and so on.
				 */
				index = 0;

				for (int i = 0; i < 3; i++) {
					int tmp_row = row + i;

					filtered_b[col + 0 + tmp_row * cols].avgerages[index] =
							average;
					filtered_b[col + 0 + tmp_row * cols].dispersions[index] =
							dispersion;
					index++;

					filtered_b[col + 1 + tmp_row * cols].avgerages[index] =
							average;
					filtered_b[col + 1 + tmp_row * cols].dispersions[index] =
							dispersion;
					index++;

					filtered_b[col + 2 + tmp_row * cols].avgerages[index] =
							average;
					filtered_b[col + 2 + tmp_row * cols].dispersions[index] =
							dispersion;
					index++;
				}
				//---------------------Alpha CHANNEL---------------------

				/* Calculate the average for the mask
				 * with the current pixel positioned at
				 * the upper-left corner */

				sum = 0;

				for (int i = 0; i < 3; i++) {
					int tmp_row = ty + i;

					sum += input_img_a[tmp_row][tx + 0];
					sum += input_img_a[tmp_row][tx + 1];
					sum += input_img_a[tmp_row][tx + 2];
				}
				average = sum / 9;

				/* Calculate the dispersion for the mask
				 * with the current pixel positioned at
				 * the upper-left corner */
				dispersion = 0;

				for (int i = 0; i < 3; i++) {
					int tmp_row = ty + i;

					dispersion += (input_img_a[tmp_row][tx + 0] - average)
							* (input_img_a[tmp_row][tx + 0] - average);

					dispersion += (input_img_a[tmp_row][tx + 1] - average)
							* (input_img_a[tmp_row][tx + 1] - average);

					dispersion += (input_img_a[tmp_row][tx + 2] - average)
							* (input_img_a[tmp_row][tx + 2] - average);
				}

				/* Assign the value of the calculated mask to each pixel
				 * i.e. the current mask will be added to index 0
				 * of the Upper left pixel, and index 1 of the
				 * Upper left-but-one pixel, and so on.
				 */
				index = 0;
				for (int i = 0; i < 3; i++) {
					int tmp_row = row + i;

					filtered_a[col + 0 + tmp_row * cols].avgerages[index] =
							average;
					filtered_a[col + 0 + tmp_row * cols].dispersions[index] =
							dispersion;
					index++;

					filtered_a[col + 1 + tmp_row * cols].avgerages[index] =
							average;
					filtered_a[col + 1 + tmp_row * cols].dispersions[index] =
							dispersion;
					index++;

					filtered_a[col + 2 + tmp_row * cols].avgerages[index] =
							average;
					filtered_a[col + 2 + tmp_row * cols].dispersions[index] =
							dispersion;
					index++;
				}
			}
		}
	}
}

template<int BLOCK_SIZE> __global__ void getArrayMin(unsigned char * output_img_r, unsigned char * output_img_g, unsigned char * output_img_b, unsigned char * output_img_a,
		Pair * input_img_r, Pair * input_img_g, Pair * input_img_b, Pair * input_img_a, int rows, int cols) {
	/* Calculate the index of the 2d array */
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * BLOCK_SIZE + ty;
	int col = bx * BLOCK_SIZE + tx;

	float min = FLT_MAX;
	int min_index = 0;
	float  *dispersions = input_img_r[col + row * cols].dispersions;
	for (int i = 0; i < 9; i++) {
		float tmp = dispersions[i];

		if (tmp < min && tmp >= 0) {
			min = tmp;
			min_index = i;
		}
	}
	output_img_r[col + row * cols] = input_img_r[col + row * cols].avgerages[min_index];

	min = FLT_MAX;
	min_index = 0;
	dispersions = input_img_g[col + row * cols].dispersions;
	for (int i = 0; i < 9; i++) {
		float tmp = dispersions[i];

		if (tmp < min && tmp >= 0) {
			min = tmp;
			min_index = i;
		}
	}
	output_img_g[col + row * cols] = input_img_g[col + row * cols].avgerages[min_index];

	min = FLT_MAX;
	min_index = 0;
	dispersions = input_img_b[col + row * cols].dispersions;
	for (int i = 0; i < 9; i++) {
		float tmp = dispersions[i];

		if (tmp < min && tmp >= 0) {
			min = tmp;
			min_index = i;
		}
	}
	output_img_b[col + row * cols] = input_img_b[col + row * cols].avgerages[min_index];

	min = FLT_MAX;
	min_index = 0;
	dispersions = input_img_a[col + row * cols].dispersions;
	for (int i = 0; i < 9; i++) {
		float tmp = dispersions[i];

		if (tmp < min && tmp >= 0) {
			min = tmp;
			min_index = i;
		}
	}
	output_img_a[col + row * cols] = input_img_a[col + row * cols].avgerages[min_index];
}

void init(BMP* imgOut, unsigned char * img_r, unsigned char * img_g, unsigned char * img_b, unsigned char * img_a, int rows, int cols) {

	// Device input image and filtered image
	unsigned char *d_img_r, *d_img_g, *d_img_b, *d_img_a;
	unsigned char *d_filtered_r, *d_filtered_g, *d_filtered_b, *d_filtered_a;
	unsigned char *filtered_r, *filtered_g, *filtered_b, *filtered_a;

	filtered_r = (unsigned char *) malloc(sizeof(unsigned char) * cols * rows);
	filtered_g = (unsigned char *) malloc(sizeof(unsigned char) * cols * rows);
	filtered_b = (unsigned char *) malloc(sizeof(unsigned char) * cols * rows);
	filtered_a = (unsigned char *) malloc(sizeof(unsigned char) * cols * rows);

	// The temporary matrix holding the averages
	// and dispersions for all 9 mask positions
	Pair *d_tmp_r, *d_tmp_g, *d_tmp_b, *d_tmp_a;

	// Allocate and copy input image to device
	int size = rows * cols * sizeof(unsigned char);

	cudaMalloc((void**) &d_img_r, size);
	cudaMemcpy(d_img_r, img_r, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &d_img_g, size);
	cudaMemcpy(d_img_g, img_g, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &d_img_b, size);
	cudaMemcpy(d_img_b, img_b, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &d_img_a, size);
	cudaMemcpy(d_img_a, img_a, size, cudaMemcpyHostToDevice);

	// Allocate memory for output image
	cudaMalloc((void**) &d_filtered_r, size);
	cudaMalloc((void**) &d_filtered_g, size);
	cudaMalloc((void**) &d_filtered_b, size);
	cudaMalloc((void**) &d_filtered_a, size);

	// Allocate memory for tmp matrix
	int size_pair = rows * cols * sizeof(Pair);

	cudaMalloc((void**) &d_tmp_r, size_pair);
	cudaMalloc((void**) &d_tmp_g, size_pair);
	cudaMalloc((void**) &d_tmp_b, size_pair);
	cudaMalloc((void**) &d_tmp_a, size_pair);

	// Define grid and block dimensions
	dim3 block(BLOCK_DIM, BLOCK_DIM, 1);

	dim3 grid((int) ceil((cols * 1.0) / (BLOCK_DIM - 2)),
			(int) ceil((rows * 1.0) / (BLOCK_DIM - 2)), 1);

	// Kernel invocations
	rotatingMaskCUDA<BLOCK_DIM> <<<grid, block>>>(d_tmp_r, d_tmp_g, d_tmp_b, d_tmp_a,
		d_img_r, d_img_g, d_img_b, d_img_a,
		rows, cols);

	dim3 grid2((int) ceil((cols * 1.0) / BLOCK_DIM),
			(int) ceil((rows * 1.0) / BLOCK_DIM), 1);

	getArrayMin<BLOCK_DIM> <<<grid2, block>>>(d_filtered_r, d_filtered_g, d_filtered_b, d_filtered_a,
		d_tmp_r, d_tmp_g, d_tmp_b, d_tmp_a,
		rows, cols);

	// Copy the filtered image to the host memory
	cudaMemcpy(filtered_r, d_filtered_r, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(filtered_g, d_filtered_g, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(filtered_b, d_filtered_b, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(filtered_a, d_filtered_a, size, cudaMemcpyDeviceToHost);

	// Free allocated memory
	cudaFree(d_img_r); cudaFree(d_img_g); cudaFree(d_img_b); cudaFree(d_img_a);
	cudaFree(d_tmp_r); cudaFree(d_tmp_g); cudaFree(d_tmp_b); cudaFree(d_tmp_a);
	cudaFree(d_filtered_r); cudaFree(d_filtered_g); cudaFree(d_filtered_b); cudaFree(d_filtered_a);

	(*imgOut).fromPixelArrays(filtered_r, filtered_g, filtered_b, filtered_a,
			cols, rows);
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
	init(&imgOut, pixelsIn_r, pixelsIn_g, pixelsIn_b, pixelsIn_a, height, width);

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
	imgOut.WriteToFile("../../output images/lena_noise_filtered.bmp");

	return 0;
}