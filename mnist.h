#ifndef MNIST_H
#define MNIST_H

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

//#include "bitmap_image.hpp"

typedef uint8_t byte;
typedef byte image[28*28];

uint32_t make_uint32(byte buffer[]);
byte* read_labels(const char filename[], unsigned* n );
image* read_images(const char filename[], unsigned* n );
//void draw_image(image img);

#endif