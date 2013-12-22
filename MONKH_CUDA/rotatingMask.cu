// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>


int init(int block_dim, unsigned char * img, int rows, int cols, unsigned char * filtered_img) {

	// Device input image and filtered image
	unsigned char *d_img, *d_filtered;
	int size = rows * cols * sizeof(unsigned char);

	// Allocate and copy input image to device
	cudaMalloc((void**) &d_img, size);
	cudaMemcpy(d_img, img, cudaMemCpyHostToDevice);

	// Allocate memory for output image
	cudaMalloc((void**) &d_filtered. size);

	// Define grid and block dimensions
	dim3 block(block_dim,block_dim,1);
	dim3 grid(ceil(cols/block_dim),ceil(rows/block_dim),1);

	// Kernel invocations
	rotatingMaskCUDA <<grid,block>> (d_filtered, d_img, n, m);
	getArrayMin<<grid, block>>(d_filtered, n, m);

	// Copy the filtered image to the host memory
	cudaMemcpy(filtered_img, d_filtered, cudaMemCpyDeviceToHost);

	// Free allocated memory
	cudaFree(d_img); cudaFree(d_filtered);
}


/* The device kernel, takes as input the noisy image
 * and outputs the filtered image
 */
template <int BLOCK_SIZE> __global__ void
rotatingMaskCUDA(unsigned char * filtered, unsigned char * img, int n, int m)
{

	__Shared__ unsigned char input_img [BLOCK_SIZE][BLOCK_SIZE];

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
    	continue;

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
getArrayMin(unsigned char * input_img, unsigned char * output_img, int n, int m)
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