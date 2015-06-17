#ifndef JSC_CUH_
#define JSC_CUH_

#include <cub/cub.cuh>
#include "types.h"

#define GLOBALSIZE 4294246400
#define SHAREDSIZE (44 * 1024)
#define SHAREDMARGIN 128
#define CONSTANTSIZE (60 * 1024)
#define THREADSPERBLOCK 1024
#define MEMORY(R1, R2, R3) ((sizeof(chunk) * f1->c + sizeof(value)) * (R1) + \
			    (sizeof(chunk) * f2->c + sizeof(value)) * (R2) + \
                   	    (sizeof(chunk) * (CEIL(f1->m + f2->m - f1->s, BITSPERCHUNK) - \
			    f1->m / BITSPERCHUNK) + sizeof(value)) * (R3) + sizeof(dim) * 3)

#define gpuerrorcheck(ans) { gpuassert((ans), __FILE__, __LINE__); }
inline void gpuassert(cudaError_t code, const char *file, int line, bool abort = true) {

        if (code != cudaSuccess) {
                fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}

__global__ void histogramproductkernel(dim *h1, dim *h2, dim *hr, dim hn);

__global__ void jointsumkernel(func f1, func f2, func f3, chunk *d1, chunk *d2, chunk *d3, value *v1, value *v2, value *v3, dim *pfxh1, dim *pfxh2, dim *pfxhp, uint4 *bd);

#endif  /* JSC_CUH_ */

