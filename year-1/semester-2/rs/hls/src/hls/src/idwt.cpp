#define MAT_SIZE 512
#define LEVELS 1

//#define PARTITION  1

extern "C" {

    int cindex(int i, int j, int width) {
        return i * width + j;
    }

    int pow(int base, int exp) {
        int result = 1;
        for (int i = 0; i < exp; i++) {
            result *= base;
        }
        return result;
    }
  
    void idwt(double *image, double* output, int levels) {
        int i, j, k, level;
        int n = MAT_SIZE;
        int n2;
        
        double g[MAT_SIZE * MAT_SIZE];
        double A[MAT_SIZE * MAT_SIZE];
        double H[MAT_SIZE * MAT_SIZE];
        double V[MAT_SIZE * MAT_SIZE];
        double D[MAT_SIZE * MAT_SIZE];

        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }

        // Process each level from the last to the first
        for (level = levels; level >= 1; level--) {
            n2 = n / pow(2, level);

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
                    output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                    output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                    output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                    output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                }
            }

            for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
                g[index] = image[index];
            }
        }
    }

    void idwt2_p_r(double *image, double* output, int levels) {
        int i, j, k, level;
        int n = MAT_SIZE;
        int n2;
        
        double g[MAT_SIZE * MAT_SIZE];
        double A[MAT_SIZE * MAT_SIZE];
        double H[MAT_SIZE * MAT_SIZE];
        double V[MAT_SIZE * MAT_SIZE];
        double D[MAT_SIZE * MAT_SIZE];

        start_copy_loop:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            #pragma HLS PIPELINE
            g[index] = image[index];
        }

        // Process each level from the last to the first
        levels_loop:
        for (level = levels; level >= 1; level--) {
            n2 = n / pow(2, level);

            // Copy values into A, H, V, and D
            assign_loop:
            for (i = 0; i < n2; i++) {
                for (j = 0; j < n2; j++) {
                    #pragma HLS PIPELINE
                    A[i * n2 + j] = g[i * n + j];
                    H[i * n2 + j] = g[(i + n2) * n + j];
                    V[i * n2 + j] = g[i * n + (j + n2)];
                    D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
                }
            }

            // Reconstruct the original values
            reconstruct_loop: 
            for (i = 0; i < n2; i++) {
                for (j = 0; j < n2; j++) {
                    #pragma HLS PIPELINE
                    output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                    output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                    output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                    output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                }
            }

            copy_loop:
            for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
                #pragma HLS PIPELINE
                g[index] = image[index];
            }
        }
    }


    void copy_data(double *src, double *dst, int size) {
        copy_loop:
        for (int i = 0; i < size; i++) {
            #pragma HLS UNROLL factor=8
            dst[i] = src[i];
        }
    }

    void compute_idwt(double *input, double *output, int n2, int n) {
        int i, j, k, level;
        double g[MAT_SIZE * MAT_SIZE];
        double A[MAT_SIZE * MAT_SIZE];
        double H[MAT_SIZE * MAT_SIZE];
        double V[MAT_SIZE * MAT_SIZE];
        double D[MAT_SIZE * MAT_SIZE];

        assign_loop:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                #pragma HLS PIPELINE
                A[i * n2 + j] = input[i * n + j];
                H[i * n2 + j] = input[(i + n2) * n + j];
                V[i * n2 + j] = input[i * n + (j + n2)];
                D[i * n2 + j] = input[(i + n2) * n + (j + n2)];
            }
        }


        reconstruct_loop: 
        for (int i = 0; i < n2; i++) {
            for (int j = 0; j < n2; j++) {
                #pragma HLS PIPELINE
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }
    }

    void idwt2_dataflow(double *image, double* output, int levels) {
        //#pragma HLS INTERFACE m_axi port=image bundle=aximm1 max_read_burst_length=16 max_write_burst_length=16
        //#pragma HLS INTERFACE m_axi port=output bundle=aximm2 max_read_burst_length=16 max_write_burst_length=1
        //#pragma HLS INTERFACE s_axilite port=return bundle=control
        //#pragma HLS INTERFACE ap_ctrl_chain port=return

        #pragma HLS DATAFLOW

        int i, j, k, level;
        int n = MAT_SIZE;
        int n2;
        level = 2;

        double output_local[MAT_SIZE * MAT_SIZE];
        double output2[MAT_SIZE * MAT_SIZE];

        double g[MAT_SIZE * MAT_SIZE];
        double g2[MAT_SIZE * MAT_SIZE];

        #pragma HLS STREAM variable=g depth=1024
        #pragma HLS STREAM variable=g2 depth=1024

        start_copy_loop:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            #pragma HLS PIPELINE
            g[index] = image[index];
        }
        n2 = n / pow(2, level);
        compute_idwt(g, output2, n2, n);
        copy_loop:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            #pragma HLS PIPELINE
            g2[index] = image[index];
        }
        level--;
        n2 = n / pow(2, level);
        compute_idwt(g2, output2, n2, n);
    }

    void idwt2_p(double *image, double* output, int levels) {
        int i, j, k, level;
        int n = MAT_SIZE;
        int n2;
        
        double g[MAT_SIZE * MAT_SIZE];
        double A[MAT_SIZE * MAT_SIZE];
        double H[MAT_SIZE * MAT_SIZE];
        double V[MAT_SIZE * MAT_SIZE];
        double D[MAT_SIZE * MAT_SIZE];

        start_copy_loop:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            #pragma HLS UNROLL factor=8
            g[index] = image[index];
        }
    
        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_1:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                #pragma HLS UNROLL factor=8 
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_1: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                #pragma HLS UNROLL factor=8
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_1:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            #pragma HLS UNROLL factor=8
            g[index] = image[index];
        }

        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_2:
       
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                #pragma HLS UNROLL factor=8
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_2: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                #pragma HLS UNROLL factor=8
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_2:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            #pragma HLS UNROLL factor=8
            g[index] = image[index];
        }

    }

    void idwt1(double *image, double* output, int levels) {
        int i, j, k, level;
        int n = MAT_SIZE;
        int n2;
        
        double g[MAT_SIZE * MAT_SIZE];
        double A[MAT_SIZE * MAT_SIZE];
        double H[MAT_SIZE * MAT_SIZE];
        double V[MAT_SIZE * MAT_SIZE];
        double D[MAT_SIZE * MAT_SIZE];

        start_copy_loop:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }
    
        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_1:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_1: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_1:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }

        n2 = n / pow(2, level);
    }

    void idwt2(double *image, double* output, int levels) {
        int i, j, k, level;
        int n = MAT_SIZE;
        int n2;
        
        double g[MAT_SIZE * MAT_SIZE];
        double A[MAT_SIZE * MAT_SIZE];
        double H[MAT_SIZE * MAT_SIZE];
        double V[MAT_SIZE * MAT_SIZE];
        double D[MAT_SIZE * MAT_SIZE];

        start_copy_loop:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }
    
        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_1:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_1: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_1:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }

        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_2:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_2: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_2:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }
    }

    void idwt4(double *image, double* output, int levels) {
        int i, j, k, level;
        int n = MAT_SIZE;
        int n2;
        
        double g[MAT_SIZE * MAT_SIZE];
        double A[MAT_SIZE * MAT_SIZE];
        double H[MAT_SIZE * MAT_SIZE];
        double V[MAT_SIZE * MAT_SIZE];
        double D[MAT_SIZE * MAT_SIZE];

        start_copy_loop:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }
    
        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_1:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_1: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_1:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }

        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_2:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_2: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_2:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }

        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_3:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_3: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_3:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }

        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_4:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_4: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_4:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }
    }

    void idwt5(double *image, double* output, int levels) {
        int i, j, k, level;
        int n = MAT_SIZE;
        int n2;
        
        double g[MAT_SIZE * MAT_SIZE];
        double A[MAT_SIZE * MAT_SIZE];
        double H[MAT_SIZE * MAT_SIZE];
        double V[MAT_SIZE * MAT_SIZE];
        double D[MAT_SIZE * MAT_SIZE];

        start_copy_loop:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }
    
        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_1:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_1: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_1:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }

        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_2:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_2: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_2:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }

        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_3:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_3: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_3:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }

        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_4:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_4: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_4:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }

        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_5:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_5: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_5:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }
    }


    void idwt6(double *image, double* output, int levels) {
        int i, j, k, level;
        int n = MAT_SIZE;
        int n2;
        
        double g[MAT_SIZE * MAT_SIZE];
        double A[MAT_SIZE * MAT_SIZE];
        double H[MAT_SIZE * MAT_SIZE];
        double V[MAT_SIZE * MAT_SIZE];
        double D[MAT_SIZE * MAT_SIZE];

        start_copy_loop:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }
    
        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_1:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_1: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_1:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }

        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_2:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_2: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_2:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }

        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_3:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_3: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_3:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }

        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_4:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_4: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_4:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }

        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_5:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_5: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_5:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }

        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_6:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_6: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_6:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }
    }

    void idwt7(double *image, double* output, int levels) {
        int i, j, k, level;
        int n = MAT_SIZE;
        int n2;
        
        double g[MAT_SIZE * MAT_SIZE];
        double A[MAT_SIZE * MAT_SIZE];
        double H[MAT_SIZE * MAT_SIZE];
        double V[MAT_SIZE * MAT_SIZE];
        double D[MAT_SIZE * MAT_SIZE];

        start_copy_loop:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }
    
        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_1:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_1: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_1:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }

        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_2:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_2: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_2:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }

        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_3:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_3: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_3:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }

        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_4:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_4: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_4:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }

        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_5:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_5: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_5:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }

        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_6:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_6: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_6:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }

        n2 = n / pow(2, level);

        // Copy values into A, H, V, and D
        assign_loop_7:
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                A[i * n2 + j] = g[i * n + j];
                H[i * n2 + j] = g[(i + n2) * n + j];
                V[i * n2 + j] = g[i * n + (j + n2)];
                D[i * n2 + j] = g[(i + n2) * n + (j + n2)];
            }
        }

        // Reconstruct the original values
        reconstruct_loop_7: 
        for (i = 0; i < n2; i++) {
            for (j = 0; j < n2; j++) {
                output[cindex(2 * i, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] - V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] - H[cindex(i, j, n2)] + V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] - V[cindex(i, j, n2)] - D[cindex(i, j, n2)]) / 4;
                output[cindex(2 * i + 1, 2 * j + 1, MAT_SIZE)] = (A[cindex(i, j, n2)] + H[cindex(i, j, n2)] + V[cindex(i, j, n2)] + D[cindex(i, j, n2)]) / 4;
            }
        }

        copy_loop_7:
        for (int index = 0; index < MAT_SIZE * MAT_SIZE; index++) {
            g[index] = image[index];
        }
    }
   
}