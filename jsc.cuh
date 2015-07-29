#ifndef JSC_CUH_
#define JSC_CUH_

#include <cub/cub.cuh>
#include "types.h"

#define GLOBALSIZE 4294246400
#define SHAREDSIZE (44 * 1024)
#define SHAREDMARGIN 128
#define CONSTANTSIZE (60 * 1024)
#define THREADSPERBLOCK 1024
#define MEMORY(R1, R2, R3) ((sizeof(chunk) * 2 * f1->c + sizeof(value)) * (R1) + \
			    (sizeof(chunk) * 2 * f2->c + sizeof(value)) * (R2) + \
                   	    (sizeof(chunk) * 2 * (CEILBPC(f1->m + f2->m - f1->s) - DIVBPC(f1->m)) + sizeof(value)) * (R3) + sizeof(dim) * 3)

#define gpuerrorcheck(ans) { gpuassert((ans), __FILE__, __LINE__); }
inline void gpuassert(cudaError_t code, const char *file, int line, bool abort = true) {

        if (code != cudaSuccess) {
                fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}

#endif  /* JSC_CUH_ */

