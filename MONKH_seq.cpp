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

// returns index of minimum element in array
int min_elem_idx(float array[], int len) {
    float min = array[0];
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

// img_in   : input image of size ROWS x COLS
// img_out  : image after applying rotating mask filter
void rot_mask_seq(char* fileIn, char* fileOut) {
    // read input image
    BMP img_in;
    img_in.ReadFromFile(fileIn);
    
    int width = img_in.TellWidth();
    int height = img_in.TellHeight();
    
    // create output image
    BMP img_out;
    img_out.SetSize(width,height);
    img_out.SetBitDepth(8);
    
    int i = 0;
    int j = 0;

    for ( i = 0; i < height; i++) {
        for ( j = 0; j < width; j++) {
            float avgs[FILTER_DIM*FILTER_DIM];
            float dispersions[FILTER_DIM*FILTER_DIM];
            int count = 0;
            // try all windows containing pixel (i,j)
            int k;
            int l;
            for ( k = 1 - FILTER_DIM; k <= 0; k++) {
                for ( l = 1 - FILTER_DIM; l <= 0; l++) {
                    // only windows fully inside image borders are used
                    if (i + k >= 0 && i + k + FILTER_DIM - 1 < height &&
                        j + l >= 0 && j + l + FILTER_DIM - 1 < width) {
                        
                        float dispersion = 0;
                        float avg = 0;                        
                        // compute average brightness in window
                        int m;
                        int n;
                        for ( m = 0; m < FILTER_DIM; m++) {
                            for ( n = 0; n < FILTER_DIM; n++) {
                                avg += img_in(i + k + m, j + l + n)->Red;
                            }
                        }
                        avg /= FILTER_DIM * FILTER_DIM;
                        avgs[count] = avg;
                        
                        // compute brightness dispersion in window
                        for ( m = 0; m < FILTER_DIM; m++) {
                            for ( n = 0; n < FILTER_DIM; n++) {
                                dispersion += powf(img_in(i + k + m, j + l + n)->Red - avg, 2);
                            }
                        }
                        dispersion /= FILTER_DIM * FILTER_DIM;
                        dispersions[count] = dispersion;   
                        count++;
                    }
                }
            }
            // use average of window with minimum dispersion in output image
            img_out(i,j)->Red = (int) avgs[min_elem_idx(dispersions,count)];
            img_out(i,j)->Green = (int) avgs[min_elem_idx(dispersions,count)];
            img_out(i,j)->Blue = (int) avgs[min_elem_idx(dispersions,count)];
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
    BMP anImage;
    anImage.ReadFromFile("vtr.bmp");
    BMP grayImg;
    toGrayScale(anImage, "image_gray.bmp");
    grayImg.ReadFromFile("image_gray.bmp");
    rot_mask_seq("image_gray.bmp", "filtered_img.bmp");
    return 0;
}

