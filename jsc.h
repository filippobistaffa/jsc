#ifndef JSC_H_
#define JSC_H_

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>

#ifdef __CUDACC__ // CUDA

#include <cub/cub.cuh>
#define SHAREDSIZE (44 * 1024)
#define CONSTANTSIZE (60 * 1024)
#define THREADSPERBLOCK 1024
#define MEMORY(I) ((sizeof(chunk) * f1.c + sizeof(value)) * f1.h[I] + (sizeof(chunk) * f2.c + sizeof(value)) * f2.h[I] + \
		   (sizeof(chunk) * (OUTPUTC - f1.m / BITSPERCHUNK) + sizeof(value)) * hp[I] + sizeof(dim) * 3)

#endif

#define JOINTOPERATION(res, x, y) ((res) = (x) + (y))

#ifndef SEED
#define SEED 1057
#endif
#define BITSPERCHUNK 64
#define CPUTHREADS 8
#define MAXVAR 800
#define MAXVALUE 1000

#ifndef ID
#define ID
typedef uint16_t id;
#endif

#ifndef VALUE
#define VALUE
typedef float value;
#endif

typedef uint64_t chunk;
typedef uint32_t dim;

typedef struct {
	chunk *data, mask, *rmask, *hmask;
	dim n, m, c, s, *h, hn;
	id *vars;
	value *v;
} func;

#ifdef ROWMAJOR
#include "rowmajor.h"
#else
#include "columnmajor.h"
#endif

#include "marsenne.h"
#include "common.h"
#include "crc32.h"

#endif /* JSC_H_ */
