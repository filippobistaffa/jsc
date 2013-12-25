#ifndef JSC_H_
#define JSC_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "marsenne.h"

#define SEED 0
#define BITSPERCHUNK 64
#define SHAREDSIZE 49152
#define THREADSPERBLOCK 32

#define N1 100
#define M1 70
#define C1 CEIL(M1, BITSPERCHUNK)

#define UNROLL 2 // MIN(C1, C2)

#define CEIL(X, Y) (1 + (((X) - 1) / (Y)))
#define SET(C, I) ((C)[(I) / BITSPERCHUNK] |= 1ULL << ((I) % BITSPERCHUNK))
#define SWAP(V, I, J, N) ({ \
        uint64_t d = (((V)[((I) / BITSPERCHUNK) * (N)] >> ((I) % BITSPERCHUNK)) ^ ((V)[((J) / BITSPERCHUNK) * (N)] >> ((J) % BITSPERCHUNK))) & 1; \
        (V)[((I) / BITSPERCHUNK) * (N)] ^= d << ((I) % BITSPERCHUNK); (V)[((J) / BITSPERCHUNK) * (N)] ^= d << ((J) % BITSPERCHUNK); })

typedef uint64_t chunk;
typedef uint16_t var;
typedef uint32_t dim;

typedef struct {
	dim s, m, n, c;
	chunk *data, *mask;
	var* vars;
} func;

#endif /* JSC_H_ */
