
#include <stdio.h>
#include <stdlib.h>

#define N 500

int main() {
    // Basic matrix multiplication heavily dependent on optimizations
    double *A = (double*)malloc(N * N * sizeof(double));
    double *B = (double*)malloc(N * N * sizeof(double));
    double *C = (double*)malloc(N * N * sizeof(double));
    
    // Initialize matrices
    for(int i = 0; i < N*N; i++) {
        A[i] = (double)(i % 100) / 100.0;
        B[i] = (double)(i % 50) / 50.0;
        C[i] = 0.0;
    }
    
    // Matrix mult
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            double sum = 0.0;
            for(int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
    
    // Prevent dead code elimination
    double final_sum = 0.0;
    for(int i = 0; i < N*N; i++) {
        final_sum += C[i];
    }
    printf("Result: %f\n", final_sum);
    
    free(A); free(B); free(C);
    return 0;
}
