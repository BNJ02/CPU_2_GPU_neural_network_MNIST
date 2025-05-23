#include "matrix.h"
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>  

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define TILE_SIZE 16

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
}

__global__
void matrix_dot_kernel(const double* __restrict__ A,
                      const double* __restrict__ B,
                      double* C,
                      int A_rows, int A_cols, int B_cols)
{
    __shared__ double tileA[TILE_SIZE][TILE_SIZE];
    __shared__ double tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    double acc = 0.0;

    /* balayage de la dimension k par tuiles */
    for (int t = 0; t < (A_cols + TILE_SIZE - 1) / TILE_SIZE; ++t) {

        /* chargement collaboratif dans la SM (tests de bord ⇒ 0.0) */
        int kA = t * TILE_SIZE + threadIdx.x;
        int kB = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < A_rows && kA < A_cols) ? A[row * A_cols + kA] : 0.0;

        tileB[threadIdx.y][threadIdx.x] =
            (kB < A_cols && col < B_cols) ? B[kB * B_cols + col] : 0.0;

        __syncthreads();                                   // barrière :contentReference[oaicite:1]{index=1}

        /* produit interne local à la tuile */
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();                                   // ré-utilisation des buffers
    }

    if (row < A_rows && col < B_cols)
        C[row * B_cols + col] = acc;
}

void matrix_dot(matrix_t* m1, matrix_t* m2, matrix_t* res)
{
    assert(m1->columns == m2->rows &&
           m1->rows    == res->rows &&
           m2->columns == res->columns);

    /* dimensions de la grille */
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((res->columns + TILE_SIZE - 1) / TILE_SIZE,
                 (res->rows    + TILE_SIZE - 1) / TILE_SIZE);

    /* (optionnel) préfetch vers le GPU pour abaisser la latence */
    int dev;  cudaGetDevice(&dev);
    cudaMemPrefetchAsync(m1->m, m1->rows * m1->columns * sizeof(double), dev);
    cudaMemPrefetchAsync(m2->m, m2->rows * m2->columns * sizeof(double), dev);
    cudaMemPrefetchAsync(res->m, res->rows * res->columns * sizeof(double), dev);

    matrix_dot_kernel<<<gridDim, blockDim>>>(
        m1->m, m2->m, res->m,
        m1->rows, m1->columns, m2->columns);
}

__device__
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

__device__
double dsigmoid(double x)
{
    return sigmoid(x) * (1.0 - sigmoid(x));
}

__global__
void matrix_function_kernel(double* in, double* out, int size, bool derivative)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        if (derivative)
            out[idx] = dsigmoid(in[idx]);
        else
            out[idx] = sigmoid(in[idx]);
    }
}

void matrix_function(matrix_t *m1, matrix_t *res, bool derivative)
{
    assert((m1->columns == res->columns) &&
           (m1->rows == res->rows));

    int size = m1->rows * m1->columns;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    matrix_function_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        m1->m, res->m, size, derivative
    );
}

__global__
void matrix_transpose_kernel(const double* in, double* out, int rows, int cols) {
    int in_row = blockIdx.y * blockDim.y + threadIdx.y;
    int in_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (in_row < rows && in_col < cols) {
        int in_idx  = in_col + in_row * cols;
        int out_idx = in_row + in_col * rows;
        out[out_idx] = in[in_idx];
    }
}

void matrix_transpose(matrix_t *m1, matrix_t *res)
{
    assert((m1->columns == res->rows) &&
           (m1->rows == res->columns));

    dim3 blockDim(16, 16);
    dim3 gridDim((m1->columns + 15) / 16, (m1->rows + 15) / 16);

    matrix_transpose_kernel<<<gridDim, blockDim>>>(
        m1->m, res->m, m1->rows, m1->columns
    );
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
}

void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert ( (dest->rows == src->rows)      &&             
             (dest->columns == src->columns));

    memcpy(dest->m, src->m, src->columns * src->rows * sizeof(double));     
}