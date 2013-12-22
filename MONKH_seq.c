//
//  MONKH_seq.c
//  MONKH
//
//  Created by Nada Nasr on 12/22/13.
//  Copyright (c) 2013 Nada Nasr. All rights reserved.
//

#include <stdio.h>

#define ROWS 5
#define COLS 5

#define FILTER_DIM 3

// returns index of minimum element in array
int min_elem_idx(float array[], int len) {
    float min = array[0];
    int min_idx = 0;
    for (int i = 0; i < len; i++) {
        if (array[i] < min) {
            min = array[i];
            min_idx = i;
        }
    }
    return min_idx;
}

// img_in   : input image of size ROWS x COLS
// img_out  : image after applying rotating mask filter
void rot_mask_seq(float img_in[][COLS], float img_out[][COLS]) {
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
//            printf("pixel: (%i,%i):\n", i, j);
            float avgs[FILTER_DIM*FILTER_DIM];
            float dispersions[FILTER_DIM*FILTER_DIM];
            int count = 0;
            // try all windows containing pixel (i,j)
            for (int k = 1 - FILTER_DIM; k <= 0; k++) {
                for (int l = 1 - FILTER_DIM; l <= 0; l++) {
                    // only windows fully inside image borders are used
                    if (i + k >= 0 && i + k + FILTER_DIM - 1 < ROWS &&
                        j + l >= 0 && j + l + FILTER_DIM - 1 < COLS) {
                        
                        float dispersion = 0;
                        float avg = 0;
                        
                        // compute average brightness in window
                        for (int m = 0; m < FILTER_DIM; m++) {
                            for (int n = 0; n < FILTER_DIM; n++) {
                                avg += img_in[i + k + m][j + l + n];
                            }
                        }
                        avg /= FILTER_DIM * FILTER_DIM;
                        avgs[count] = avg;
                        
                        // compute brightness dispersion in window
                        for (int m = 0; m < FILTER_DIM; m++) {
                            for (int n = 0; n < FILTER_DIM; n++) {
                                dispersion += powf(img_in[i + k + m][j + l + n] - avg, 2);
                            }
                        }
                        dispersion /= FILTER_DIM * FILTER_DIM;
                        dispersions[count] = dispersion;
                        
//                        printf("        i+k: %i |  j+l: %i | avg: %f | disp: %f |\n", i+k, j+l, avg, dispersion);
                        
                        count++;
                    }
                }
            }
            // use average of window with minimum dispersion in output image
            img_out[i][j] = avgs[min_elem_idx(dispersions,count)];
        }
    }
}

int main (int arc, char **argv) {
    float img_in[ROWS][COLS];
    float img_out[ROWS][COLS];
    
    //  test matrix
    //   0   2  20   6  11
    //   1   3   7   8  15
    //  66  14   4  10   3
    //   8  16  42  52   6
    //   6   2  14   3   7
    img_in[0][0] = 0; img_in[0][1] = 2; img_in[0][2] = 20; img_in[0][3] = 6; img_in[0][4] = 11;
    img_in[1][0] = 1; img_in[1][1] = 3; img_in[1][2] = 7; img_in[1][3] = 8; img_in[1][4] = 15;
    img_in[2][0] = 66; img_in[2][1] = 14; img_in[2][2] = 4; img_in[2][3] = 10; img_in[2][4] = 3;
    img_in[3][0] = 8; img_in[3][1] = 16; img_in[3][2] = 42; img_in[3][3] = 52; img_in[3][4] = 6;
    img_in[4][0] = 6; img_in[4][1] = 2; img_in[4][2] = 14; img_in[4][3] = 3; img_in[4][4] = 7;
    
    rot_mask_seq(img_in, img_out);
    
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%f ", img_in[i][j]);
        }
        printf("%s", "\n");
    }
    
    printf("%s\n","");
    
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%f ", img_out[i][j]);
        }
        printf("%s", "\n");
    }
    
    return 0;
}