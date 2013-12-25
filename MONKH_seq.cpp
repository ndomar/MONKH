//
//  MONKH_seq.cpp
//  MONKH
//
//  Created by Nada Nasr on 12/22/13.
//  Copyright (c) 2013 Nada Nasr. All rights reserved.
//

        

#include <stdio.h>
#include <EasyBMP.h>
#include <Math.h>


#define FILTER_DIM 3

enum Channel { Red, Green, Blue, Alpha};

// returns index of minimum element in array
int min_elem_idx(int array[], int len) {
    int min = array[0];
    int min_idx = 0;
    int i = 0;
    while (i < len) {
        if (array[i] < min) {
            min = array[i];
            min_idx = i;
        }
        i++;
    }
    return min_idx;
}

void rot_mask_seq(char* fileIn, char* fileOut) {
    // read input image
    BMP img_in;
    img_in.ReadFromFile(fileIn);
    
    int width = img_in.TellWidth();
    int height = img_in.TellHeight();
    
    // create output image
    BMP img_out;
    img_out.SetSize(width,height);
    img_out.SetBitDepth(24);
    
    int i = 0;
    int j = 0;

    for ( i = 0; i < height; i++) {
        for ( j = 0; j < width; j++) {
            int avgs_r[FILTER_DIM*FILTER_DIM];
			int avgs_g[FILTER_DIM*FILTER_DIM];
			int avgs_b[FILTER_DIM*FILTER_DIM];
			int avgs_a[FILTER_DIM*FILTER_DIM];
			
            int dispersions_r[FILTER_DIM*FILTER_DIM];
			int dispersions_g[FILTER_DIM*FILTER_DIM];
			int dispersions_b[FILTER_DIM*FILTER_DIM];
			int dispersions_a[FILTER_DIM*FILTER_DIM];
			
            int count = 0;
			
            // try all windows containing pixel (i,j)
            int k;
            int l;
            for ( k = 1 - FILTER_DIM; k <= 0; k++) {
                for ( l = 1 - FILTER_DIM; l <= 0; l++) {
                    // only windows fully inside image borders are used
                    if (i + k >= 0 && i + k + FILTER_DIM - 1 < height &&
                        j + l >= 0 && j + l + FILTER_DIM - 1 < width) {

						// perform computation for RED channel
                        int dispersion = 0;
                        int avg = 0;                        
                        // compute average brightness in window
                        int m;
                        int n;
                        for ( m = 0; m < FILTER_DIM; m++) {
                            for ( n = 0; n < FILTER_DIM; n++) {
                                avg += img_in(i + k + m, j + l + n)->Red;
                            }
                        }
                        avg /= FILTER_DIM * FILTER_DIM * 1.0;
                        avgs_r[count] = avg;
                        
                        // compute brightness dispersion in window
                        for ( m = 0; m < FILTER_DIM; m++) {
                            for ( n = 0; n < FILTER_DIM; n++) {
                                dispersion += powf(img_in(i + k + m, j + l + n)->Red - avg, 2);
                            }
                        }
                        dispersion /= FILTER_DIM * FILTER_DIM * 1.0;
                        dispersions_r[count] = dispersion;
						
						// perform computation for GREEN channel
                        dispersion = 0;
                        avg = 0;                        
                        // compute average brightness in window
                        for ( m = 0; m < FILTER_DIM; m++) {
                            for ( n = 0; n < FILTER_DIM; n++) {
                                avg += img_in(i + k + m, j + l + n)->Green;
                            }
                        }
                        avg /= FILTER_DIM * FILTER_DIM * 1.0;
                        avgs_g[count] = avg;
                        
                        // compute brightness dispersion in window
                        for ( m = 0; m < FILTER_DIM; m++) {
                            for ( n = 0; n < FILTER_DIM; n++) {
                                dispersion += powf(img_in(i + k + m, j + l + n)->Green - avg, 2);
                            }
                        }
                        dispersion /= FILTER_DIM * FILTER_DIM * 1.0;
                        dispersions_g[count] = dispersion;
						
						// perform computation for BLUE channel
                        dispersion = 0;
                        avg = 0;                        
                        // compute average brightness in window
                        for ( m = 0; m < FILTER_DIM; m++) {
                            for ( n = 0; n < FILTER_DIM; n++) {
                                avg += img_in(i + k + m, j + l + n)->Blue;
                            }
                        }
                        avg /= FILTER_DIM * FILTER_DIM * 1.0;
                        avgs_b[count] = avg;
                        
                        // compute brightness dispersion in window
                        for ( m = 0; m < FILTER_DIM; m++) {
                            for ( n = 0; n < FILTER_DIM; n++) {
                                dispersion += powf(img_in(i + k + m, j + l + n)->Blue - avg, 2);
                            }
                        }
                        dispersion /= FILTER_DIM * FILTER_DIM * 1.0;
                        dispersions_b[count] = dispersion;
						
						   // perform computation for ALPHA channel
                        dispersion = 0;
                        avg = 0;                        
                        // compute average brightness in window
                        for ( m = 0; m < FILTER_DIM; m++) {
                            for ( n = 0; n < FILTER_DIM; n++) {
                                avg += img_in(i + k + m, j + l + n)->Alpha;
                            }
                        }
                        avg /= FILTER_DIM * FILTER_DIM * 1.0;
                        avgs_a[count] = avg;
                        
                        // compute brightness dispersion in window
                        for ( m = 0; m < FILTER_DIM; m++) {
                            for ( n = 0; n < FILTER_DIM; n++) {
                                dispersion += powf(img_in(i + k + m, j + l + n)->Alpha - avg, 2);
                            }
                        }
                        dispersion /= FILTER_DIM * FILTER_DIM * 1.0;
                        dispersions_a[count] = dispersion;
						   
                        count++;
                    }
                }
            }
            // use average of window with minimum dispersion in output image
            img_out(i,j)->Red = (int) avgs_r[min_elem_idx(dispersions_r,count)];
            img_out(i,j)->Green = (int) avgs_g[min_elem_idx(dispersions_g,count)];
            img_out(i,j)->Blue = (int) avgs_b[min_elem_idx(dispersions_b,count)];
			img_out(i,j)->Alpha = (int) avgs_a[min_elem_idx(dispersions_a,count)];
        }
    }
    img_out.WriteToFile(fileOut);
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
    rot_mask_seq("lena_noise.bmp", "lena_noise_filtered.bmp");
    return 0;
}