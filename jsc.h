#ifndef JSC_H_
#define JSC_H_

#define _GNU_SOURCE

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#define SEED 0
#define BITSPERCHUNK 64
#define SHAREDSIZE 49152
#define THREADSPERBLOCK 32
#define CPUTHREADS 8
#define MAXVAR 100

#define N1 100
#define M1 70
#define C1 CEIL(M1, BITSPERCHUNK)

#define CEIL(X, Y) (1 + (((X) - 1) / (Y)))

typedef uint64_t chunk;
typedef uint16_t var;
typedef uint32_t dim;

typedef struct {
	chunk *data, mask;
	dim n, m, c, s;
	var *vars;
} func;

#include "marsenne.h"
#include "preproc.h"
#include "qsort.h"
#include "crc32.h"

#endif /* JSC_H_ */
