#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>


#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define IMAGE_WIDTH 512



// Define image structure
// struct Image {
//     uint8_t pixels[IMAGE_WIDTH][IMAGE_HEIGHT];
// };



// calculate DWT 
// void forward_wavelet_transform(double *image, double* output, int levels) {
//     int i, j, i_sc, j_sc;
//     double temp;
//     int height = IMAGE_WIDTH;
//     int width = IMAGE_WIDTH;
//     double a, b, c, d, A, H, V, D;
// 
//     double *temp_output = malloc(sizeof(double) * width * height);
// 
//     if (levels > 0) {
//         for (i = 0; i < height; i += 2) {
//             for (j = 0; j < width; j += 2) {
//                 a = image[i * width + j];
//                 b = image[i * width + j + 1];
//                 c = image[(i + 1) * width + j];
//                 d = image[(i + 1) * width + j + 1];
// 
//                 // Compute the averages and differences.
//                 A = (a + b + c + d) / 4; // Approximation.
//                 H = (-a - b + c + d) / 4; // Horizontal detail.
//                 V = (-a + b - c + d) / 4; // Vertical detail.
//                 D = (a - b - c + d) / 4; // Diagonal detail.
// 
//                 // Place the computed values back into the matrix.
//                 i_sc = i / 2;
//                 j_sc = j / 2;
// 
//                 temp_output[i_sc * width + j_sc] = A; // Top-left quadrant.
//                 temp_output[i_sc * width + j_sc + width / 2] = H; // Top-right quadrant.
//                 temp_output[(i_sc + height / 2) * width + j_sc] = V; // Bottom-left quadrant.
//                 temp_output[(i_sc + height / 2) * width + j_sc + width / 2] = D; // Bottom-right quadrant.
//             }
//         }
// 
//         // Copy the temporary output to the output.
//         memcpy(output, temp_output, sizeof(double) * width * height);
// 
//         // Recursively apply the transform to the approximation coefficient.
//         forward_wavelet_transform(output, output, levels - 1);
//     }
// }

int cindex(int i, int j, int width) {
    return i * width + j;
}

// calculate DWT
void forward_wavelet_transform(double *image, double* output, int levels) {
    int i, j, i_sc, j_sc;
    double temp;
    int height = IMAGE_WIDTH;
    int width = IMAGE_WIDTH;
    double a, b, c, d, A, H, V, D;

    double *image_copy = (double*)malloc(sizeof(double) * width * height);
    memcpy(image_copy, image, sizeof(double) * width * height);

    for (int level = 0; level < levels; level++) {
        for (i = 0; i < height; i += 2) {
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, IMAGE_WIDTH)];
                b = image_copy[cindex(i, j + 1, IMAGE_WIDTH)];
                c = image_copy[cindex(i + 1, j, IMAGE_WIDTH)];
                d = image_copy[cindex(i + 1, j + 1, IMAGE_WIDTH)];

                // Compute the averages and differences.
                A = (a + b + c + d); // Approximation.
                H = (-a - b + c + d); // Horizontal detail.
                V = (-a + b - c + d); // Vertical detail.
                D = (a - b - c + d); // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, IMAGE_WIDTH)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, IMAGE_WIDTH)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), IMAGE_WIDTH)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), IMAGE_WIDTH)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        memcpy(image_copy, output, sizeof(double) * IMAGE_WIDTH * IMAGE_WIDTH);

        width /= 2;
        height /= 2;
    }
}



// calculate Inverse DWT (IDWT) 
void inverse_wavelet_transform(double *image, double* output, int levels) {
    int i, j, k, level;
    int n = IMAGE_WIDTH;
    int n2;
    
    // Allocate memory for the intermediate result array g
    double *g = (double*)malloc(sizeof(double) * n * n);
    memcpy(g, image, sizeof(double) * n * n);

    // Process each level from the last to the first
    for (level = levels; level >= 1; level--) {
        printf("LOOPING");
        n2 = n / pow(2, level); //(2 * level);

        // Create temporary arrays for A, H, V, and D
        double *A = (double*)malloc(sizeof(double) * n2 * n2);
        double *H = (double*)malloc(sizeof(double) * n2 * n2);
        double *V = (double*)malloc(sizeof(double) * n2 * n2);
        double *D = (double*)malloc(sizeof(double) * n2 * n2);

        // Copy values into A, H, V, and D
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, IMAGE_WIDTH)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, IMAGE_WIDTH)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, IMAGE_WIDTH)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, IMAGE_WIDTH)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        // Free temporary arrays
        free(A);
        free(H);
        free(V);
        free(D);

        memcpy(g, image, sizeof(double) * IMAGE_WIDTH * IMAGE_WIDTH);        
    }
    // Free the intermediate result array
    free(g);
}


// improve for showing 
void improve(double *image, double* output) {
    int i, j;
    int height = IMAGE_WIDTH;
    int width = IMAGE_WIDTH;
    // Approximation coefficient 
    for (i = 0; i < height/2; i++) {
        for (j = 0; j < width/2; j++) {
            // calculate average of 4 pixels in the neighborhood
            output[i * width + j] = image[i * width + j]/4;
            output[(i + height/2) * width + j] = image[(i + height/2) * width + j]+0.5;
            output[i * width + j + height/2] = image[i * width + j + height/2]+0.5;
            output[(i + height/2) * width + j + height/2] = image[(i + height/2) * width + j + height/2]+0.5;
        }
    }
}



int main() {
    // Define sample grayscale image

    // Load image from file and allocate space for the output image
    char image_name[] = "./elaine.jpg";
    int width, height, cpp;
    // load only gray scale image
    unsigned char *h_imageIn;
    h_imageIn = stbi_load(image_name, &width, &height, &cpp, STBI_grey);
    if (h_imageIn == NULL)
    {
        printf("Error reading loading image %s!\n", image_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d.\n", image_name, width, height);
    printf("Image is %d bytes per pixel.\n", cpp);
    // Save grayscale image to file
    printf("Size of image is %ld, %ld\n", sizeof(unsigned char), sizeof(h_imageIn));

    double *image_pixels = (double*)malloc(sizeof(double) * width * height);
    double *output = (double*)malloc(sizeof(double) * width * height);
    double *show_output = (double*)malloc(sizeof(double) * width * height);

    // convert to grayscale 
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            image_pixels[i*width + j] = h_imageIn[i * width + j]/255.0;
        }
    }





    forward_wavelet_transform(image_pixels, output, 7);
    printf("DWT done\n");
    improve(output, show_output);
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            h_imageIn[i * width + j] = (char)(show_output[i * width + j]*255);
            h_imageIn[i * width + j] = (char)(show_output[i * width + j]*255);
        }
    }
    stbi_write_jpg("dwt_output_intermediate.jpg", width, height, STBI_grey, h_imageIn, 100);
    printf("Improvement done\n");
    inverse_wavelet_transform(output, output, 7);
    printf("IDWT done\n");

    // Save image to file
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            h_imageIn[i*width + j] = (char)(output[i*width + j]*255);
        }
    }





    // Free memory
    stbi_write_jpg("dwt_output.jpg", width, height,STBI_grey, h_imageIn, 100);
    free(image_pixels);

    return 0;
}
