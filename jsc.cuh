#ifndef JSC_CUH_
#define JSC_CUH_

#include <cub/cub.cuh>
#include "types.h"

#define MEMORYLIMIT 4294246400
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

char jointsum(func *f1, func *f2, func *fo);

#endif  /* JSC_CUH_ */

