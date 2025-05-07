#include "matrix.h"
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>  

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

matrix_t * alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t * res = (matrix_t*) malloc( sizeof(matrix_t) );
    // res->m = (double *) calloc(columns * rows, sizeof(double));
    res->columns = columns;
    res->rows = rows;

    size_t size = rows * columns * sizeof(double);
    cudaMallocManaged(&res->m, size);       // allocation unifiée
    // cudaMemset(res->m, 0, size);            // initialise à zéro (comme calloc)
    // for(int i = 0; i < size; ++i) {
    //     m[i] = 0;
    // }

    return res;
}

void destroy_matrix(matrix_t *m)
{
    //printf("free %p %p\n", m, m->m);
    cudaFree(m->m);
    // free(m->m);
    free(m);
}

void print_matrix(matrix_t *m, bool is_short){
    unsigned lim_rows = 0;
    unsigned lim_col = 0;

    if (is_short)
    {
        lim_rows = MIN(m->rows, 4);
        lim_col = MIN(m->columns, 10);
    }
    else
    {
        lim_rows = m->rows;
        lim_col = m->columns;
    }

    for (int row = 0; row < lim_rows; row ++)
    {
        for (int col = 0; col < lim_col; col ++)
        {
            printf("%.2lf ", m->m[col + row * m->columns]);
        }
        if (is_short && lim_col != m->columns) printf("...");
        printf("\n");
    }
    if (is_short && lim_rows != m->rows) printf("...\n");
}

__global__
void hadamard_product_kernel(double* m1, double* m2, double* res, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        res[idx] = m1[idx] * m2[idx];
    }
}

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));

    int size = m1->rows * m1->columns;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    hadamard_product_kernel<<<blocksPerGrid, threadsPerBlock>>>(m1->m, m2->m, res->m, size);
    cudaDeviceSynchronize();
}

__global__
void matrix_sum_kernel(double* m1, double* m2, double* res, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        res[idx] = m1[idx] + m2[idx];
    }
}

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    int size = m1->rows * m1->columns;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    matrix_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(m1->m, m2->m, res->m, size);
    cudaDeviceSynchronize();
}

__global__
void matrix_minus_kernel(double* m1, double* m2, double* res, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        res[idx] = m1[idx] - m2[idx];
    }
}

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));
             
    int size = m1->rows * m1->columns;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    matrix_minus_kernel<<<blocksPerGrid, threadsPerBlock>>>(m1->m, m2->m, res->m, size);
    cudaDeviceSynchronize();
}

__global__
void matrix_dot_kernel(const double *A, const double *B, double *C,
                       int A_rows, int A_cols, int B_cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < B_cols) {
        double acc = 0.0;
        for (int k = 0; k < A_cols; ++k) {
            acc += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = acc;
    }
}

void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert((m1->columns == m2->rows) &&
           (m1->rows == res->rows) &&
           (m2->columns == res->columns));

    dim3 blockDim(16, 16);
    dim3 gridDim((m2->columns + 15) / 16, (m1->rows + 15) / 16);

    matrix_dot_kernel<<<gridDim, blockDim>>>(m1->m, m2->m, res->m,
        m1->rows, m1->columns, m2->columns);

    cudaDeviceSynchronize(); // attendre que le résultat soit prêt
}

__device__ double sigmoid_device(double x) {
    return 1.0 / (1.0 + exp(-x));
}

__global__
void matrix_sigmoid_kernel(double* m, double* res, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        res[idx] = sigmoid_device(m[idx]);
    }
}

void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res)
{
    assert ( (m1->columns == res->columns) &&             
             (m1->rows == res->rows));

    int size = m1->rows * m1->columns;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    matrix_sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(m1->m, res->m, size);
    cudaDeviceSynchronize();
}

void matrix_transpose(matrix_t *m1, matrix_t *res)
{
    assert ( (m1->columns == res->rows) &&             
             (m1->rows == res->columns));
    
    for (int row = 0; row < m1->rows; row++)
    {
        for (int col = 0; col < m1->columns; col ++)
        {
            res->m[row + col * m1->rows] = m1->m[col + row * m1->columns];
        }
    }
}

__global__
void matrix_scalar_kernel(double* m, double scalar, double* res, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        res[idx] = m[idx] * scalar;
    }
}

void matrix_scalar(matrix_t *m1, double s, matrix_t *res)
{
    assert ( (m1->rows == res->rows) &&             
             (m1->columns == res->columns));

    int size = m1->rows * m1->columns;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    matrix_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(m1->m, s, res->m, size);
    cudaDeviceSynchronize();
}

void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert ( (dest->rows == src->rows)      &&             
             (dest->columns == src->columns));

    memcpy(dest->m, src->m, src->columns * src->rows * sizeof(double));     
}