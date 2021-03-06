#ifndef JSC_CUH_
#define JSC_CUH_

#include <cub/cub.cuh>
#include "types.h"

#ifdef INPLACE
#include "inplace/inplace/transpose.h"
#define TRANSPOSEFACTOR 1
#else
#define TRANSPOSEFACTOR 2
#endif

#define SHAREDSIZE (44 * 1024)
#define SHAREDMARGIN 128
#define CONSTANTSIZE (60 * 1024)
#define THREADSPERBLOCK 1024
#define RESULTDATA(R3) (sizeof(chunk) * (R3) * CEILBPC(f1->m + f2->m - f1->s))
#define MEMORY(R1, R2, R3) ((sizeof(chunk) * f1->c + sizeof(value)) * (R1) + \
			    (sizeof(chunk) * f2->c + sizeof(value)) * (R2) + \
                   	    RESULTDATA(R3) + sizeof(value) * (R3) + sizeof(dim) * 3)

#define THRESHOLD 10 // Execute the join sum on the CPU if hn < THRESHOLD

#define gpuerrorcheck(ans) { gpuassert((ans), __FILE__, __LINE__); }
inline void gpuassert(cudaError_t code, const char *file, int line, bool abort = true) {

        if (code != cudaSuccess) {
                fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}

#ifdef KERNELERRORCHECK
#define GPUERRORCHECK do { gpuerrorcheck(cudaPeekAtLastError()); gpuerrorcheck(cudaDeviceSynchronize()); } while (0)
#else
#define GPUERRORCHECK
#endif

#endif  /* JSC_CUH_ */

