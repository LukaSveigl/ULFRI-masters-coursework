#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <immintrin.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define NUM_CLUSTERS 16
#define MAX_ITERATIONS 10000
#define THRESHOLD 0.0001


unsigned long long rdtsc() {
    unsigned int hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

// Define cluster structure
struct Cluster {
    double centroid;
    int num_points;
    int* points;
};

// Function to calculate Euclidean distance between two points
double distance(double p1, double p2) {
    return fabs(p1 - p2);
}


// Vectorized version of assign_points_to_clusters, using AVX intrinsics.
void assign_points_to_clusters(struct Cluster clusters[], double *image, int image_size) {
    int cluster_index = 0;
    double min_distance = 100000.0;
    double d[NUM_CLUSTERS] __attribute__((aligned(32)));
    double centroids_buffer[NUM_CLUSTERS] __attribute__((aligned(32)));

    // Move the gather call out of the inner loop
    for (int k = 0; k < NUM_CLUSTERS; k += 4) {
        __m256d centroids = _mm256_i64gather_pd(&clusters[k].centroid, _mm256_set_epi64x(9, 6, 3, 0), 8);
        _mm256_store_pd(&centroids_buffer[k], centroids);
    }

    for (int i = 0; i < image_size; i++) {
        // Move the pixels = set1 call one level higher
        __m256d image_vec = _mm256_set1_pd(image[i]);

        for (int k = 0; k < NUM_CLUSTERS; k += 4) {
            __m256d centroids = _mm256_load_pd(&centroids_buffer[k]);
            __m256d distance = _mm256_sub_pd(image_vec, centroids);
            __m256d absolute_distance = _mm256_andnot_pd(_mm256_set1_pd(-0.0), distance);

            _mm256_store_pd(&d[k], absolute_distance);
        }

        min_distance = d[0];
        cluster_index = 0;
        for (int k = 1; k < NUM_CLUSTERS; k++) {
            if (d[k] < min_distance) {
                min_distance = d[k];
                cluster_index = k;
            }
        }

        clusters[cluster_index].points[clusters[cluster_index].num_points++] = i;
    }
}

void update_centroids(struct Cluster clusters[], double* image, int image_size) {
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        __m256d sum_vec = _mm256_setzero_pd();
        for (int k = 0; k < clusters[i].num_points; k += 4) {
            __m256i pixel_indices = _mm256_set_epi64x(clusters[i].points[k+3], clusters[i].points[k+2], clusters[i].points[k+1], clusters[i].points[k]);
            __m256d pixels = _mm256_i64gather_pd(image, pixel_indices, 8);
            sum_vec = _mm256_add_pd(sum_vec, pixels);
        }
        double sum[4];
        _mm256_storeu_pd(sum, sum_vec);
        clusters[i].centroid = (sum[0] + sum[1] + sum[2] + sum[3]) / clusters[i].num_points;
    }
}

// K-means clustering function
void k_means(double* image, int image_size, struct Cluster* clusters) {
    struct Cluster clusters_temp[NUM_CLUSTERS];
    
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        clusters[i].points = (int*)malloc(sizeof(int) * image_size);
    }
    
    // Initialize centroids randomly
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        clusters[i].centroid = image[rand() % image_size];
        clusters[i].num_points = 0;
    }
    
    double error = 0;
    int iterations = 0;
    
    do {
        // Assign points (pixels) to clusters
        // Save old clusters
        for (int i = 0; i < NUM_CLUSTERS; i++) {
            clusters_temp[i] = clusters[i];
        }
        // Reinitialize cluster points
        for (int i = 0; i < NUM_CLUSTERS; i++) {
            clusters[i].num_points = 0;
        }        

        // print centroids
        assign_points_to_clusters(clusters, image, image_size);
        // Update centroids
        update_centroids(clusters, image, image_size);

        printf("Cluster centroids:\n");
        for (int i = 0; i < NUM_CLUSTERS; i++) {
            printf("Cluster %d: %.2f\n", i + 1, clusters[i].centroid);
        }
        //assign_points_to_clusters(clusters, image, image_size);

        // Calculate difference between old and new centroids
        error = 0;
        for (int i = 0; i < NUM_CLUSTERS; i++) {
            error += distance(clusters[i].centroid, clusters_temp[i].centroid);
        }
        iterations++;


        printf("Iteration %d: error = %.2f\n", iterations, error);
    } while (error > THRESHOLD && iterations < MAX_ITERATIONS);
    
    //Print cluster centroids
    printf("Cluster centroids:\n");
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        printf("Cluster %d: %.2f\n", i + 1, clusters[i].centroid);
    }

    // Free memory
    // for (int i = 0; i < NUM_CLUSTERS; i++) {
    //     free(clusters[i].points);
    // }
}

// Function to segment image based on cluster values
void segment_image(double *image, struct Cluster* clusters, int image_size) {
    for (int i = 0; i < image_size; i++) {
        double min_distance = fabs(image[i] - clusters[0].centroid);
        int cluster_index = 0;
        for (int k = 1; k < NUM_CLUSTERS; k++) {
            double d = distance(image[i], clusters[k].centroid);
            if (d < min_distance) {
                min_distance = d;
                cluster_index = k;
            }
        }
        image[i] = (double)clusters[cluster_index].centroid*255.0;
    }
}

int main() {
    long long unsigned int start, end, cycles;

    // Define sample grayscale image
    // Load image from file and allocate space for the output image
    char image_name[] = "./bosko_grayscale.jpg";
    int width, height, cpp;
    // load only gray scale image
    unsigned char *h_imageIn = stbi_load(image_name, &width, &height, &cpp, STBI_grey);
    if (h_imageIn == NULL)
    {
        printf("Error reading loading image %s!\n", image_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d.\n", image_name, width, height);
    printf("Image is %d bytes per pixel.\n", cpp);
    // Save grayscale image to file
    printf("Size of image is %ld, %ld\n", sizeof(unsigned char), sizeof(h_imageIn));
    //stbi_write_jpg("bosko_grayscale.png", width, height,STBI_grey, h_imageIn, 100);
    
    


    double *image_pixels = (double*)malloc(sizeof(double) * width * height);
    // convert to grayscale 
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            image_pixels[i*width + j] = h_imageIn[i * width + j]/255.0;
        }
    }

    int image_size = width * height;

    // save image to file
    //stbi_write_jpg("bosko_grayscale_v2.jpg", width, height,STBI_grey, image.pixels[0], 100);


    // cluster centroids
    struct Cluster clusters[NUM_CLUSTERS];


    // Perform K-means clustering
    start = rdtsc();
    k_means(image_pixels, image_size, clusters);
    end = rdtsc();
    cycles = end - start;

    double CPU_FREQ = 2.0 * 1000 * 1000 * 1000; // 2GHz for AMD

    double seconds = (double)(end - start) / CPU_FREQ;

    printf("Time for K-means: %lld cycles\n", cycles);
    printf("Time for K-means: %f seconds\n", seconds);

    //print cluster centroids
    printf("Cluster centroids:\n");
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        printf("Cluster %d: %.2f\n", i + 1, clusters[i].centroid);
    }

    // // Segment image
    segment_image(image_pixels, clusters, image_size);

    // Save image to file
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            h_imageIn[i*width + j] = (char)(image_pixels[i*width + j]);
        }
    }
    // Free memory

    // free(image.pixels[0]);
    // free(image.pixels);
    //free(clusters->num_points);
    stbi_write_jpg("bosko_k-means.jpg", width, height, STBI_grey, h_imageIn, 100);
    if (image_pixels != NULL) {
        free(image_pixels);
    }

    return 0;
}
