#define MAT_SIZE 512
#define LEVELS 1

//#define PARTITION  1

extern "C" {
  
    int cindex(int i, int j, int width) {
        return i * width + j;
    }

    void dwt(double *image, double* output, int levels) {
        int i, j, i_sc, j_sc;
        double temp;
        int height = MAT_SIZE;
        int width = MAT_SIZE;
        double a, b, c, d, A, H, V, D;

        height = 1024;
        width = 1024;

        levels = 5;

        double image_copy[width * height];
        copy_loop_1:
        for (int index = 0; index < width * height; index++) {
            #pragma HLS UNROLL factor=8
            image_copy[index] = image[index];
        }

        dwt_levels_loop:
        for (int level = 0; level < 5; level++) {
            dwt_height_loop:
            for (i = 0; i < height; i += 2) {
                #pragma HLS PIPELINE
                dwt_width_loop:
                for (j = 0; j < width; j += 2) {
                    #pragma HLS UNROLL factor=2
                    // Get the 2x2 block of pixels.
                    a = image_copy[cindex(i, j, MAT_SIZE)];
                    b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                    c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                    d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                    // Compute the averages and differences.
                    A = (a + b + c + d) / 4; // Approximation.
                    H = (-a - b + c + d) / 4; // Horizontal detail.
                    V = (-a + b - c + d) / 4; // Vertical detail.
                    D = (a - b - c + d) / 4; // Diagonal detail.

                    i_sc = i / 2;
                    j_sc = j / 2;

                    output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                    output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                    output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                    output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
                }
            }

            // Copy the temporary output to the output.
            copy_loop_2:
            for (int index = 0; index < width * height; index++) {
                #pragma HLS UNROLL factor=8
                image_copy[index] = output[index];
            }

            width /= 2;
            height /= 2;
        }
    }

    void dwt2_p(double *image, double* output, int levels) {
        int i, j, i_sc, j_sc;
        double temp;
        int height = MAT_SIZE;
        int width = MAT_SIZE;
        double a, b, c, d, A, H, V, D;

        levels = LEVELS;

        double image_copy[width * height];
        copy_loop_1:
        for (int index = 0; index < width * height; index++) {
            #pragma HLS UNROLL factor=8
            image_copy[index] = image[index];
        }

        dwt_height_loop1:
        for (i = 0; i < height; i += 2) {
            #pragma HLS PIPELINE
            dwt_width_loop1:
            for (j = 0; j < width; j += 2) {
                #pragma HLS UNROLL factor=2
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_2:
        for (int index = 0; index < width * height; index++) {
            #pragma HLS UNROLL factor=8
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

        dwt_height_loop2:
        for (i = 0; i < height; i += 2) {
            #pragma HLS PIPELINE
            dwt_width_loop2:
            for (j = 0; j < width; j += 2) {
                #pragma HLS UNROLL factor=2
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_3:
        for (int index = 0; index < width * height; index++) {
            #pragma HLS UNROLL factor=8
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

    }

    void copy_data(double *src, double *dst, int size) {
        copy_loop:
        for (int i = 0; i < size; i++) {
            #pragma HLS UNROLL factor=8
            dst[i] = src[i];
        }
    }

    void compute_dwt(double *input, double *output, int width, int height, int N) {
        int i, j, i_sc, j_sc;
        double a, b, c, d, A, H, V, D;

        dwt_height_loop:
        for (i = 0; i < height; i += 2) {
            #pragma HLS PIPELINE
            dwt_width_loop:
            for (j = 0; j < width; j += 2) {
                #pragma HLS UNROLL factor=2
                // Get the 2x2 block of pixels.
                a = input[cindex(i, j, MAT_SIZE)];
                b = input[cindex(i, j + 1, MAT_SIZE)];
                c = input[cindex(i + 1, j, MAT_SIZE)];
                d = input[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (N / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (N / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (N / 2), j_sc + (N / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }
    }

    void dwt2_dataflow(double* image, double* output) {
        #pragma HLS INTERFACE m_axi port=image bundle=aximm1 max_read_burst_length=16 max_write_burst_length=16
        #pragma HLS INTERFACE m_axi port=output bundle=aximm2 max_read_burst_length=16 max_write_burst_length=1
        #pragma HLS INTERFACE s_axilite port=return bundle=control
        #pragma HLS INTERFACE ap_ctrl_chain port=return

        #pragma HLS DATAFLOW

        int height = MAT_SIZE;
        int width = MAT_SIZE;

        double image_copy[MAT_SIZE * MAT_SIZE];
        double output_local[MAT_SIZE * MAT_SIZE];
        double output2[MAT_SIZE * MAT_SIZE];
        double image_copy2[MAT_SIZE * MAT_SIZE];

        #pragma HLS STREAM variable=image_copy depth=1024
        #pragma HLS STREAM variable=output_local depth=1024
        #pragma HLS STREAM variable=image_copy2 depth=1024
        #pragma HLS STREAM variable=output2 depth=1024

        copy_data(image, image_copy, width * height);
        compute_dwt(image_copy, output_local, width, height, height);

        width /= 2;
        height /= 2;

        copy_data(output_local, image_copy2, width * height);
        compute_dwt(image_copy2, output2, width, height, height);

        copy_data(output2, output, width * height);
    }   

    void dwt1(double *image, double* output, int levels) {

        int i, j, i_sc, j_sc;
        double temp;
        int height = MAT_SIZE;
        int width = MAT_SIZE;
        double a, b, c, d, A, H, V, D;

        levels = LEVELS;

        double image_copy[width * height];
        copy_loop_1:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = image[index];
        }

        dwt_height_loop1:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop1:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_2:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;
    }

    void dwt2(double *image, double* output, int levels) {

        int i, j, i_sc, j_sc;
        double temp;
        int height = MAT_SIZE;
        int width = MAT_SIZE;
        double a, b, c, d, A, H, V, D;

        levels = LEVELS;

        double image_copy[width * height];
        copy_loop_1:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = image[index];
        }

        dwt_height_loop1:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop1:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_2:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

        dwt_height_loop2:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop2:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_3:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;
    }

    void dwt3(double *image, double* output, int levels) {

        int i, j, i_sc, j_sc;
        double temp;
        int height = MAT_SIZE;
        int width = MAT_SIZE;
        double a, b, c, d, A, H, V, D;

        levels = LEVELS;

        double image_copy[width * height];
        copy_loop_1:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = image[index];
        }

        dwt_height_loop1:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop1:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_2:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

        dwt_height_loop2:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop2:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_3:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

        dwt_height_loop3:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop3:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_4:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

    }

    void dwt4(double *image, double* output, int levels) {

        int i, j, i_sc, j_sc;
        double temp;
        int height = MAT_SIZE;
        int width = MAT_SIZE;
        double a, b, c, d, A, H, V, D;

        levels = LEVELS;

        double image_copy[width * height];
        copy_loop_1:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = image[index];
        }

        dwt_height_loop1:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop1:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_2:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

        dwt_height_loop2:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop2:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_3:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

        dwt_height_loop3:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop3:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_4:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

        dwt_height_loop4:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop4:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_5:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;
    }

    void dwt5(double *image, double* output, int levels) {

        int i, j, i_sc, j_sc;
        double temp;
        int height = MAT_SIZE;
        int width = MAT_SIZE;
        double a, b, c, d, A, H, V, D;

        levels = LEVELS;

        double image_copy[width * height];
        copy_loop_1:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = image[index];
        }

        dwt_height_loop1:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop1:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_2:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

        dwt_height_loop2:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop2:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_3:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

        dwt_height_loop3:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop3:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_4:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

        dwt_height_loop4:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop4:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_5:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

        dwt_height_loop5:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop5:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_6:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;
    }

    void dwt6(double *image, double* output, int levels) {

        int i, j, i_sc, j_sc;
        double temp;
        int height = MAT_SIZE;
        int width = MAT_SIZE;
        double a, b, c, d, A, H, V, D;

        levels = LEVELS;

        double image_copy[width * height];
        copy_loop_1:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = image[index];
        }

        dwt_height_loop1:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop1:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_2:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

        dwt_height_loop2:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop2:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_3:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

        dwt_height_loop3:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop3:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_4:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

        dwt_height_loop4:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop4:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_5:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

        dwt_height_loop5:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop5:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_6:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

        dwt_height_loop6:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop6:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_7:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;
    }

    void dwt7(double *image, double* output, int levels) {

        int i, j, i_sc, j_sc;
        double temp;
        int height = MAT_SIZE;
        int width = MAT_SIZE;
        double a, b, c, d, A, H, V, D;

        levels = LEVELS;

        double image_copy[width * height];
        copy_loop_1:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = image[index];
        }

        dwt_height_loop1:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop1:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_2:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

        dwt_height_loop2:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop2:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_3:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

        dwt_height_loop3:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop3:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_4:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

        dwt_height_loop4:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop4:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_5:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

        dwt_height_loop5:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop5:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_6:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

        dwt_height_loop6:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop6:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_7:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;

        dwt_height_loop7:
        for (i = 0; i < height; i += 2) {
            dwt_width_loop7:
            for (j = 0; j < width; j += 2) {
                // Get the 2x2 block of pixels.
                a = image_copy[cindex(i, j, MAT_SIZE)];
                b = image_copy[cindex(i, j + 1, MAT_SIZE)];
                c = image_copy[cindex(i + 1, j, MAT_SIZE)];
                d = image_copy[cindex(i + 1, j + 1, MAT_SIZE)];

                // Compute the averages and differences.
                A = (a + b + c + d) / 4; // Approximation.
                H = (-a - b + c + d) / 4; // Horizontal detail.
                V = (-a + b - c + d) / 4; // Vertical detail.
                D = (a - b - c + d) / 4; // Diagonal detail.

                // Place the computed values back into the matrix according to the diagram shown:
                // matrix[i/2][j/2] = A  # Top-left quadrant
                // matrix[i/2 + N/2][j/2 ] = H  # Top-right quadrant
                // matrix[i/2][j/2 + N/2] = V  # Bottom-left quadrant
                // matrix[(i/2) + N/2][(j/2) + N/2] = D  # Bottom-right quadrant

                i_sc = i / 2;
                j_sc = j / 2;

                output[cindex(i_sc, j_sc, MAT_SIZE)] = A; // Top-left quadrant
                output[cindex(i_sc + (height / 2), j_sc, MAT_SIZE)] = H; // Top-right quadrant
                output[cindex(i_sc, j_sc + (width / 2), MAT_SIZE)] = V; // Bottom-left quadrant
                output[cindex(i_sc + (height / 2), j_sc + (width / 2), MAT_SIZE)] = D; // Bottom-right quadrant
            }
        }

        // Copy the temporary output to the output.
        copy_loop_8:
        for (int index = 0; index < width * height; index++) {
            image_copy[index] = output[index];
        }

        width /= 2;
        height /= 2;
    }
}

