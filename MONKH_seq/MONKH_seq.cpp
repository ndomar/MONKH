#include <stdio.h>
#include <EasyBMP.h>
#include <Math.h>

#define FILTER_DIM 3

enum Channel {Red, Green, Blue, Alpha};

// returns index of minimum element in array
int minElemIdx(double array[], int len) {
    double min = array[0];
    int minIdx = 0;
    int i = 0;
    while (i < len) {
        if (array[i] < min) {
            min = array[i];
            minIdx = i;
        }
        i++;
    }
    return minIdx;
}

/*
 * returns the average of the pixel values at the indicated channel in the 
 * FILTER_DIM x FILTER_DIM filter whose upper left corner pixel is (i,j) 
 */
float getAvgOfMask(BMP* imgIn, Channel channel, int i, int j) {
    float avg = 0;
    int m, n;
    for (m = 0; m < FILTER_DIM; m++) {
        for (n = 0; n < FILTER_DIM; n++) {
			switch(channel) {
				case Red:
					avg += (*imgIn)(i + m, j + n)->Red;
					break;
				case Green:
					avg += (*imgIn)(i + m, j + n)->Green;
					break;
				case Blue:
					avg += (*imgIn)(i + m, j + n)->Blue;
					break;
				case Alpha:
					avg += (*imgIn)(i + m, j + n)->Alpha;
					break;
			}
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
double getDispersionOfMask(BMP* imgIn, Channel channel, int avg, int i, int j) {
    double dispersion = 0;
	int m, n;
    for (m = 0; m < FILTER_DIM; m++) {
        for (n = 0; n < FILTER_DIM; n++) {
			switch(channel) {
				case Red:
					dispersion += powf((*imgIn)(i + m, j + n)->Red - avg, 2);
					break;
				case Green:
					dispersion += powf((*imgIn)(i + m, j + n)->Green - avg, 2);
					break;
				case Blue:
					dispersion += powf((*imgIn)(i + m, j + n)->Blue - avg, 2);
					break;
				case Alpha:
					dispersion += powf((*imgIn)(i + m, j + n)->Alpha - avg, 2);
					break;
			}
        }
    }
    dispersion /= FILTER_DIM * FILTER_DIM * 1.0;
    return dispersion;
}

void rotMaskSeq(char* fileIn, char* fileOut) {
    // read input image
    BMP imgIn;
    imgIn.ReadFromFile(fileIn);
    
    int width = imgIn.TellWidth();
    int height = imgIn.TellHeight();
    
    // create output image
    BMP imgOut;
    imgOut.SetSize(width,height);
    imgOut.SetBitDepth(24);
    
    int i = 0;
    int j = 0;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            float avgs_r[FILTER_DIM*FILTER_DIM];
			float avgs_g[FILTER_DIM*FILTER_DIM];
			float avgs_b[FILTER_DIM*FILTER_DIM];
			float avgs_a[FILTER_DIM*FILTER_DIM];
			
            double dispersions_r[FILTER_DIM*FILTER_DIM];
			double dispersions_g[FILTER_DIM*FILTER_DIM];
			double dispersions_b[FILTER_DIM*FILTER_DIM];
			double dispersions_a[FILTER_DIM*FILTER_DIM];
			
            int count = 0;
			
            // try all windows containing pixel (i,j)
            int k;
            int l;
            for (k = 0; k >= 1 - FILTER_DIM; k--) {
                for (l = 0; l >= 1 - FILTER_DIM; l--) {
                    // only windows fully inside image borders are used
                    if (i + k >= 0 && i + k + FILTER_DIM - 1 < height &&
                        j + l >= 0 && j + l + FILTER_DIM - 1 < width) {
						
						float avg_r = getAvgOfMask(&imgIn, Red, i + k, j + l);
						float avg_g = getAvgOfMask(&imgIn, Green, i + k, j + l);
						float avg_b = getAvgOfMask(&imgIn, Blue, i + k, j + l);
						float avg_a = getAvgOfMask(&imgIn, Alpha, i + k, j + l);
						
						avgs_r[count] = avg_r;
						avgs_g[count] = avg_g;
						avgs_b[count] = avg_b;
						avgs_a[count] = avg_a;
						
						dispersions_r[count] = getDispersionOfMask(&imgIn, Red, avg_r, i + k, j + l);
						dispersions_g[count] = getDispersionOfMask(&imgIn, Green, avg_g, i + k, j + l);
						dispersions_b[count] = getDispersionOfMask(&imgIn, Blue, avg_b, i + k, j + l);
						dispersions_a[count] = getDispersionOfMask(&imgIn, Alpha, avg_a, i + k, j + l);
						
                        count++;
                    }
                }
            }
            // use average of window with minimum dispersion in output image
            imgOut(i,j)->Red = (int) avgs_r[minElemIdx(dispersions_r,count)];
            imgOut(i,j)->Green = (int) avgs_g[minElemIdx(dispersions_g,count)];
            imgOut(i,j)->Blue = (int) avgs_b[minElemIdx(dispersions_b,count)];
			imgOut(i,j)->Alpha = (int) avgs_a[minElemIdx(dispersions_a,count)];
        }
    }
    imgOut.WriteToFile(fileOut);
}

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

int main(int argc, char** argv) {
    rotMaskSeq("../test images/lena_noise.bmp", "lena_noise_filtered.bmp");
    return 0;
}