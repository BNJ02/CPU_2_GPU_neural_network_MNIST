#include "ann.h"
#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>

double normalRand(double mu, double sigma);
void init_weight(matrix_t* w, unsigned nneurones_prev);
void print_layer(layer_t *layer);

double normalRand(double mu, double sigma)
{
	const double epsilon = DBL_MIN;
	const double two_pi = 2.0*M_PI;
    static bool generate = false;
    static double z1;

	generate = !generate;

	if (!generate)
	   return z1 * sigma + mu;

	double u1, u2;
	do
	 {
	   u1 = (double) rand() / RAND_MAX;
	   u2 = (double) rand() / RAND_MAX;
	 }
	while ( u1 <= epsilon );

	double z0;
	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}

void init_weight(matrix_t* w, unsigned nneurones_prev)
{
    for (int idx = 0; idx < w->columns * w->rows; idx ++)
    {
        w->m[idx] = normalRand(0, 1 / sqrt(nneurones_prev));
    }
}

ann_t * create_ann(double alpha, unsigned minibatch_size, unsigned number_of_layers, unsigned* nneurons_per_layer)
{
    ann_t * nn = (ann_t *)malloc(sizeof(ann_t));

    nn->layers = (layer_t **)malloc(number_of_layers * sizeof(layer_t *));
    nn->number_of_layers = number_of_layers;
    nn->alpha = alpha;
    nn->minibatch_size = minibatch_size;

    nn->layers[0] = create_layer(0, nneurons_per_layer[0], minibatch_size, minibatch_size);
    for (int l = 1; l < number_of_layers; l++)
    {
        nn->layers[l] = create_layer(l, nneurons_per_layer[l], nneurons_per_layer[l-1], minibatch_size);
    }

    return nn;
}

layer_t * create_layer(unsigned layer_number, unsigned number_of_neurons, unsigned nneurons_previous_layer, unsigned minibatch_size)
{
    layer_t * layer = (layer_t*) malloc(sizeof(layer_t));

    layer->number_of_neurons = number_of_neurons;
    layer->minibatch_size = minibatch_size;    
    layer->activations = alloc_matrix(number_of_neurons, minibatch_size);
    layer->z = alloc_matrix(number_of_neurons, minibatch_size);
    layer->delta = alloc_matrix(number_of_neurons, minibatch_size);
    layer->weights = alloc_matrix(number_of_neurons, nneurons_previous_layer);    
    layer->biases = alloc_matrix(number_of_neurons, 1);

    if (layer_number > 0)
    {
        init_weight(layer->weights, nneurons_previous_layer);
    }

    return layer;
}

void set_input(ann_t *nn, matrix_t* input){
    matrix_memcpy(nn->layers[0]->activations, input);
}

void print_layer(layer_t *layer)
{
    printf("-- neurons:%d, minibatch size:%d\n", layer->number_of_neurons, layer->minibatch_size);

    printf(">> Weighted inputs --\n");
    print_matrix(layer->z, true);
    printf(">> Activations --\n");
    print_matrix(layer->activations, true);
    
    printf(">> Weights --\n");
    print_matrix(layer->weights, true);
    printf(">> Biases --\n");
    print_matrix(layer->biases, true);

    printf(">> Delta --\n");
    print_matrix(layer->delta, true);
    
}

void print_nn(ann_t *nn)
{
    printf("ANN -- nlayers:%d, alpha:%lf, minibatch size: %d\n", nn->number_of_layers, nn->alpha, nn->minibatch_size);
    for (int l = 0; l < nn->number_of_layers; l++)
    {
        printf("Layer %d ", l);
        print_layer(nn->layers[l]);
    }
}

/* ============================================================================
* Forward propagation
*   ‑ persistent scratch:
*       z1[l]  : W_l * a_{l‑1}
*       z2[l]  : b_l * 1
*       one    : vector of 1 used for bias broadcast
* ============================================================================
*/
void forward(ann_t *nn) {
    /* persistent scratch : one, z1, z2 ------------------------------------ */
    static matrix_t **z1 = nullptr, **z2 = nullptr;
    static matrix_t  *one1 = nullptr;            // 1 × m (bias broadcast)
    static unsigned   lay_cached = 0, mb_cached = 0;

    if (lay_cached != nn->number_of_layers || mb_cached != nn->minibatch_size)
    {
        /* free previous */
        if (z1)
        {
            for (unsigned l = 1; l < lay_cached; ++l) {
                destroy_matrix(z1[l]);
                destroy_matrix(z2[l]);
            }
            free(z1); free(z2); destroy_matrix(one1);
        }

        lay_cached = nn->number_of_layers;
        mb_cached  = nn->minibatch_size;

        z1  = (matrix_t**)malloc(lay_cached * sizeof(matrix_t*));
        z2  = (matrix_t**)malloc(lay_cached * sizeof(matrix_t*));

        for (unsigned l = 1; l < lay_cached; ++l) {
            z1[l] = alloc_matrix(nn->layers[l]->number_of_neurons, mb_cached);
            z2[l] = alloc_matrix(nn->layers[l]->number_of_neurons, mb_cached);
        }
        one1 = alloc_matrix(1, mb_cached);
        for (unsigned i = 0; i < one1->rows * one1->columns; ++i) one1->m[i] = 1.0;
    }

    /* actual propagation --------------------------------------------------- */
    for (unsigned l = 1; l < nn->number_of_layers; ++l) {
        matrix_dot(nn->layers[l]->weights, nn->layers[l-1]->activations, z1[l]);
        matrix_dot(nn->layers[l]->biases,  one1,                         z2[l]);
        matrix_sum(z1[l], z2[l], nn->layers[l]->z);
        matrix_function(nn->layers[l]->z, nn->layers[l]->activations, false);
    }
}
 
 
/* ============================================================================
* Back‑propagation
*   ‑ persistent scratch:
*       tw[l]        : transposed weights / grad_w
*       delta_tmp[l] : temporary δ before Hadamard
*       dfz[l]       : f'(z_{l})
*       b1[l]        : grad_b
*       one          : vector of 1 for bias reduction
* ============================================================================
*/
void backward(ann_t *nn, matrix_t *y)
{
    const unsigned L = nn->number_of_layers - 1;

    /* persistent scratch ---------------------------------------------------- */
    static matrix_t **tw = nullptr, **delta_tmp = nullptr,
            **dfz = nullptr, **w1 = nullptr, **ta = nullptr,
            **b1 = nullptr;                    // ∇b
    static matrix_t  *one2 = nullptr;                  // m × 1 (bias reduce)
    static unsigned   lay_cached = 0, mb_cached = 0;

    if (lay_cached != nn->number_of_layers || mb_cached != nn->minibatch_size)
    {
        /* free previous */
        if (tw) {
            for (unsigned l = 1; l < lay_cached; ++l) {
                destroy_matrix(tw[l]); destroy_matrix(delta_tmp[l]);
                destroy_matrix(dfz[l]); destroy_matrix(w1[l]);
                destroy_matrix(ta[l]); destroy_matrix(b1[l]);
            }

            free(tw); free(delta_tmp); free(dfz);
            free(w1); free(ta); free(b1); destroy_matrix(one2);
        }

        lay_cached = nn->number_of_layers;
        mb_cached  = nn->minibatch_size;

        tw        = (matrix_t**)malloc(lay_cached * sizeof(matrix_t*));
        delta_tmp = (matrix_t**)malloc(lay_cached * sizeof(matrix_t*));
        dfz       = (matrix_t**)malloc(lay_cached * sizeof(matrix_t*));
        w1        = (matrix_t**)malloc(lay_cached * sizeof(matrix_t*));
        ta        = (matrix_t**)malloc(lay_cached * sizeof(matrix_t*));
        b1        = (matrix_t**)malloc(lay_cached * sizeof(matrix_t*));

        for (unsigned l = 1; l < lay_cached; ++l) {
            tw[l]        = alloc_matrix(nn->layers[l-1]->number_of_neurons,
                                        nn->layers[l]->number_of_neurons);
            delta_tmp[l] = alloc_matrix(nn->layers[l-1]->number_of_neurons,
                                        mb_cached);
            dfz[l]       = alloc_matrix(nn->layers[l]->number_of_neurons,
                                        mb_cached);
            w1[l]        = alloc_matrix(nn->layers[l]->number_of_neurons,
                                        nn->layers[l-1]->number_of_neurons);
            ta[l]        = alloc_matrix(mb_cached,
                                        nn->layers[l-1]->number_of_neurons);
            b1[l]        = alloc_matrix(nn->layers[l]->number_of_neurons, 1);
        }
        /* dfz[0] never used but keep consistent for simplicity */
        dfz[0] = alloc_matrix(nn->layers[0]->number_of_neurons, mb_cached);

        one2 = alloc_matrix(mb_cached, 1);
        for (unsigned i = 0; i < one2->rows * one2->columns; ++i) one2->m[i] = 1.0;
    }

    /* --- δ^L -------------------------------------------------------------- */
    matrix_minus(nn->layers[L]->activations, y, nn->layers[L]->delta);
    matrix_function(nn->layers[L]->z, dfz[L], true);
    hadamard_product(nn->layers[L]->delta, dfz[L], nn->layers[L]->delta);

    /* --- layers L .. 1 ---------------------------------------------------- */
    for (int l = L; l > 0; --l) {
        /* ∇Wᶫ : delta^l × (a^{l-1})ᵀ   ------------------------------------ */
        matrix_transpose(nn->layers[l-1]->activations, ta[l]);
        matrix_dot(nn->layers[l]->delta, ta[l], w1[l]);
        matrix_scalar(w1[l], nn->alpha / mb_cached, w1[l]);
        matrix_minus(nn->layers[l]->weights, w1[l], nn->layers[l]->weights);

        /* ∇bᶫ : delta^l × 1 ---------------------------------------------- */
        matrix_dot(nn->layers[l]->delta, one2, b1[l]);
        matrix_scalar(b1[l], nn->alpha / mb_cached, b1[l]);
        matrix_minus(nn->layers[l]->biases, b1[l], nn->layers[l]->biases);

        if (l > 1) {
            /* δ^{l-1} ------------------------------------------------------ */
            matrix_transpose(nn->layers[l]->weights, tw[l]);
            matrix_dot(tw[l], nn->layers[l]->delta, delta_tmp[l]);
            matrix_function(nn->layers[l-1]->z, dfz[l-1], true);
            hadamard_product(delta_tmp[l], dfz[l-1], nn->layers[l-1]->delta);
        }
    }
}

 