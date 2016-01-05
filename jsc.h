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

#ifndef JSCMAIN
#include "../cucop/params.h"
#endif

#define JOINOPERATION(res, x, y) ((res) = (x) + (y))

#ifdef PRINTTIME
#define TIMER_START(msg) do { printf(msg " "); fflush(stdout); gettimeofday(&t1, NULL); } while (0)
#define TIMER_STOP do { gettimeofday(&t2, NULL); printf("%f seconds\n", (double)(t2.tv_usec - t1.tv_usec) / 1e6 + t2.tv_sec - t1.tv_sec); } while (0)
#else
#define TIMER_START(msg)
#define TIMER_STOP
#endif

#define ADDTIME_START gettimeofday(&t1a, NULL)
#define ADDTIME_STOP do { gettimeofday(&t2a, NULL); at += (double)(t2a.tv_usec - t1a.tv_usec) / 1e6 + t2a.tv_sec - t1a.tv_sec; } while (0)

#ifndef SEED
#define SEED 0
#endif
#define CPUTHREADS 8
#define MAXVAR 29000

#include "colours.h"
#include "types.h"
#include "macros.h"
#include "compare.h"

#include "marsenne.h"
#include "common.h"
#include "crc32.h"

#include "rowmajor.cpp"
#ifdef __CUDACC__ // CUDA
#include "sort.cpp"
#include "jsc.cuh"
#include "jsc.cu"
#endif

#endif /* JSC_H_ */
