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

#ifdef __cplusplus // CUDA

#include <cudpp.h>

#define SHAREDSIZE 49152
#define THREADSPERBLOCK 512
#define MAXTHREADSPERBLOCK 1024

#endif

#define SEED 1057
#define BITSPERCHUNK 64
#define CPUTHREADS 8
#define MAXVAR 200

typedef uint64_t chunk;
typedef uint16_t var;
typedef uint32_t dim;

typedef struct {
	chunk *data, mask, *rmask, *hmask;
	dim n, m, c, s, *h, hn;
	var *vars;
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
