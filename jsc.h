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
#include "jsc.cuh"
#endif

#include "../cucop/params.h"

#define JOINTOPERATION(res, x, y) ((res) = (x) + (y))

#ifdef PRINTTIME
#define TIMER_START(msg) do { printf(msg " "); fflush(stdout); gettimeofday(&t1, NULL); } while (0)
#define TIMER_STOP do { gettimeofday(&t2, NULL); printf("%f seconds\n", (double)(t2.tv_usec - t1.tv_usec) / 1e6 + t2.tv_sec - t1.tv_sec); } while (0)
#else
#define TIMER_START(msg) do {} while (0)
#define TIMER_STOP do {} while (0)
#endif

#ifndef SEED
#define SEED 0
#endif
#define CPUTHREADS 8
#define MAXVAR 800
#define MAXVALUE 1000

#include "colours.h"
#include "types.h"

#ifdef ROWMAJOR
#include "rowmajor.h"
#else
#include "columnmajor.h"
#endif

#include "marsenne.h"
#include "common.h"
#include "crc32.h"

#endif /* JSC_H_ */
