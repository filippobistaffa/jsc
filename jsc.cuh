#ifndef JSC_CUH_
#define JSC_CUH_

#include <cub/cub.cuh>
#include "types.h"

#define SHAREDSIZE (44 * 1024)
#define CONSTANTSIZE (60 * 1024)
#define THREADSPERBLOCK 1024
#define MEMORY(I) ((sizeof(chunk) * f1.c + sizeof(value)) * f1.h[I] + (sizeof(chunk) * f2.c + sizeof(value)) * f2.h[I] + \
                   (sizeof(chunk) * (OUTPUTC - f1.m / BITSPERCHUNK) + sizeof(value)) * hp[I] + sizeof(dim) * 3)

#define gpuerrorcheck(ans) { gpuassert((ans), __FILE__, __LINE__); }
inline void gpuassert(cudaError_t code, const char *file, int line, bool abort = true) {

        if (code != cudaSuccess) {
                fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}

func jointsum(func f1, func f2);

#endif  /* JSC_CUH_ */

