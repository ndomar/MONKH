#include <stdio.h>
#include <EasyBMP.h>
#include <Math.h>
#include <float.h>

#define FILTER_DIM 3

/* converts image to a grayscale image.
 */
void toGrayScale(BMP img, char* newImg)
{
    int i;
    int j;
    for(j = 0; j < img.TellHeight(); j++)
    {
        for(i = 0; i < img.TellWidth(); i++)
        {
            int grayPixel = (int) floor(0.299 * img(i,j)->Red 
                                  + 0.587 * img(i,j)->Green
                                  + 0.114 * img(i,j)->Blue);
            
            ebmpBYTE grayByte = (ebmpBYTE) grayPixel;
            img(i,j)->Red = grayByte;
            img(i,j)->Green = grayByte;
            img(i,j)->Blue = grayByte;
        }
    }
    img.WriteToFile(newImg);
}

/*
 * returns index of minimum element in float array
 */
int minElemIdx(float array[], int len) {
    float min = FLT_MAX;
    int minIdx = 0;
    int i;
    for(i = 0; i < len; i++) {
        if (array[i] < min) {
            min = array[i];
            minIdx = i;
        }
    }
    return minIdx;
}

/*
 * returns the average of the pixel values at the indicated channel in the 
 * FILTER_DIM x FILTER_DIM filter whose upper left corner pixel is (i,j) 
 */
float getAvgOfMask(unsigned char * imgIn, int i, int j, int width) {
    float avg = 0;
    int m, n;
    for (m = 0; m < FILTER_DIM; m++) {
        for (n = 0; n < FILTER_DIM; n++) {
					avg += imgIn[(i + m) * width + (j + n)];
        }
    }
    avg /= FILTER_DIM * FILTER_DIM * 1.0;
    return avg;
}

/*
 * returns the dispersion of the pixel values with respect to avg at the
 * indicated channel in the FILTER_DIM x FILTER_DIM filter whose upper 
 * left corner pixel is (i,j) 
 */
float getDispersionOfMask(unsigned char * imgIn, int avg, int i, int j, int width) {
    float dispersion = 0;
	  int m, n;
    for (m = 0; m < FILTER_DIM; m++) {
        for (n = 0; n < FILTER_DIM; n++) {
					dispersion += (imgIn[(i + m) * width + (j + n)] - avg) 
            * (imgIn[(i + m) * width + (j + n)] - avg);
        }
    }
    dispersion /= FILTER_DIM * FILTER_DIM * 1.0;
    return dispersion;
}

/*
 * returns a pointer to a pixel array that results from performing rotating 
 * mask filtering on the input pixel array 
 */
unsigned char * rotMaskSeq(unsigned char * imgIn, int width, int height) {

    unsigned char * imgOut;
    
    int i = 0;
    int j = 0;
    
    imgOut = (unsigned char *) malloc(sizeof(unsigned char) * width * height);

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            float averages[FILTER_DIM*FILTER_DIM];
            float dispersions[FILTER_DIM*FILTER_DIM];
			
            int count = 0;
			
            // try all windows containing pixel (i,j)
            int k, l;
            for (k = 0; k >= 1 - FILTER_DIM; k--) {
                for (l = 0; l >= 1 - FILTER_DIM; l--) {
                    // only windows fully inside image borders are used
                    if (i + k >= 0 && i + k + FILTER_DIM - 1 < height &&
                        j + l >= 0 && j + l + FILTER_DIM - 1 < width) {
            
                          float average = getAvgOfMask(imgIn, i + k, j + l, width);
                          averages[count] = average;
                          dispersions[count] = getDispersionOfMask(imgIn, average, i + k, j + l, width);
                          count++;
                    }
                }
            }
            // use average of window with minimum dispersion in output image
            imgOut[i * width + j] = (unsigned char) averages[minElemIdx(dispersions,count)];
        }
    }
    return imgOut;
}

/*
 * Writes to the file fileOut the image in the file fileIn after performing noise 
 * filtering using the rotating mask filtering algorithm
 */
void filterNoise(char* fileIn, char* fileOut) {
  BMP imgIn, imgOut;
  
  imgIn.ReadFromFile(fileIn);
  int width = imgIn.TellWidth();
  int height = imgIn.TellHeight();
  
  unsigned char *pixelsIn_r, *pixelsIn_g, *pixelsIn_b, *pixelsIn_a;
  unsigned char *pixelsOut_r, *pixelsOut_g, *pixelsOut_b, *pixelsOut_a;
  
  // read the 4 channels R, G, B and A from the BMP object
  pixelsIn_r = imgIn.getPixelArray(Red);
  pixelsIn_g = imgIn.getPixelArray(Green);
  pixelsIn_b = imgIn.getPixelArray(Blue);
  pixelsIn_a = imgIn.getPixelArray(Alpha);
  
  // compute the corresponding 4 channels after performing filtering
  pixelsOut_r = rotMaskSeq(pixelsIn_r, width, height);
  pixelsOut_g = rotMaskSeq(pixelsIn_g, width, height);
  pixelsOut_b = rotMaskSeq(pixelsIn_b, width, height);
  pixelsOut_a = rotMaskSeq(pixelsIn_a, width, height);
  
  // write the computed channels to a bmp image file
  imgOut.bmpFromPixelArrays(pixelsOut_r, pixelsOut_g, pixelsOut_b, pixelsOut_a, width, height);
  imgOut.WriteToFile(fileOut);
}

int main(int argc, char** argv) {
    filterNoise("../test images/lena_noise.bmp", "lena_noise_filtered.bmp");
    return 0;
}