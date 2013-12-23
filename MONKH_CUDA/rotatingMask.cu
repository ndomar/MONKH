// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

/* The device kernel, takes as input the noisy image
 * and outputs the filtered image
 */
template <int BLOCK_SIZE> __global__ void
rotatingMaskCUDA(unsigned char * filtered, unsigned char * img, int n, int m)
{

	__shared__ unsigned char input_img [BLOCK_SIZE][BLOCK_SIZE];

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    /* Overlapping the tiles */

    // Upper-left corner
    if (bx == 0 && by == 0)
    	;

    // Lower-left corner
    else if (bx == 0 && by == ((m/BLOCK_SIZE) - 1 ))
    	row -= 2;

    // Upper-right corner
    else if (by == 0 )
    	col -= 2;

    else {
    	row -= 2;
    	col -= 2;
    }

    input_img[ty][tx] = img[n*row + col];

    __syncthreads();

    /* Calculating the mask with the current pixel
     * positioned at the upper-left corner */
    int delta [3] = {0, 1, 2};

    int sum = 0;
    for(int i = 0; i < 3; i++)
    {
    	for(int j = 0; j < 3; j++)
    	{
    		int tmp_col = tx + delta [j];
    		int tmp_row = ty + delta [i];
    		sum += input_img[tmp_row][tmp_col];
    	}
    }

    int average = sum / 9;

    /* Assign the value of the calculated mask to each pixel
     * i.e. the current mask will be added to index 0
     * of the Upper left pixel, and index 1 of the
     * Upper left-but-one pixel, and so on.
     */
    int index = 0;
    for(int i = 0; i < 3; i++)
    {
    	for(int j = 0; j < 3; j++)
    	{
    		int tmp_col = col + delta [j];
    		int tmp_row = row + delta [i];
    		filtered[tmp_col + tmp_row * n + index * tmp_col * tmp_row] = average;
    		index++;
    	}
    }
}

template <int BLOCK_SIZE> __global__ void
getArrayMin(unsigned char * output_img, unsigned char * input_img, int n, int m)
{
    /* Calculate the index of the 2d array */
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    int min = 9999;
    for(int i = 0; i < 9; i++)
    {
    	int tmp = input_img[col + row * n + i * n * m];
    	min = tmp < min ? tmp : min;
    }

    output_img[col + row * n] = min;
}

void init(int block_dim, unsigned char * img, int rows, int cols, unsigned char * filtered_img) {

	// Device input image and filtered image
	unsigned char *d_img, *d_tmp, *d_filtered;
	int size = rows * cols * sizeof(unsigned char);

	// Allocate and copy input image to device
	cudaMalloc((void**) &d_img, size);
	cudaMemcpy(d_img, img, size, cudaMemcpyHostToDevice);

	// Allocate memory for tmp matrix
	cudaMalloc((void**) &d_tmp, size * 9);

	// Allocate memory for output image
	cudaMalloc((void**) &d_filtered, size);

	// Define grid and block dimensions
	dim3 block(block_dim,block_dim,1);
	dim3 grid(ceil(cols/block_dim), ceil(rows/block_dim),1);

	// Kernel invocations
	rotatingMaskCUDA<8><<<grid,block>>> (d_tmp, d_img, rows, cols);
	getArrayMin<8><<<grid, block>>> (d_filtered, d_tmp, rows, cols);

	// Copy the filtered image to the host memory
	cudaMemcpy(filtered_img, d_filtered, size, cudaMemcpyDeviceToHost);

	// Free allocated memory
	cudaFree(d_img);
	cudaFree(d_tmp);
	cudaFree(d_filtered);
}

int main(int argc, char **argv)
{

	printf("[Rotating mask technique for image filtering Using CUDA] - Starting...\n");

	// By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    int devID = 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        cudaSetDevice(devID);
    }

    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

	// Random number generator
	srand(time(NULL));

	// Size of input and output images
	unsigned int size = 16 * 16 * sizeof(unsigned char);

	unsigned char *matrix = (unsigned char *)malloc(size);
	unsigned char *filtered_img = (unsigned char *)malloc(size);

	for(int i = 0; i < 16; i++)
	{
		for(int j = 0; j < 16; j++)
		{
			matrix[i + j * 16] = rand() % 256 ;
		}
	}

	init(8, matrix, 16, 16, filtered_img);

	for(int i = 0; i < 16; i++)
	{
		for(int j = 0; j < 16; j++)
		{
			printf("%d ", matrix[i + j * 16]);
		}
		printf("\n");
	}

	printf("Filtered\n");
	printf("\n");

	for(int i = 0; i < 16; i++)
	{
		for(int j = 0; j < 16; j++)
		{
			printf("%d ", filtered_img[i + j * 16]);
		}
		printf("\n");
	}

	return 1;
}
