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

#define CEIL(X, Y) (1 + (((X) - 1) / (Y)))
#define SET(C, I) ((C)[(I) / BITSPERCHUNK] |= 1ULL << ((I) % BITSPERCHUNK))
#define SWAP(V, I, J, N) ({ \
        uint64_t d = (((V)[((I) / BITSPERCHUNK) * (N)] >> ((I) % BITSPERCHUNK)) ^ ((V)[((J) / BITSPERCHUNK) * (N)] >> ((J) % BITSPERCHUNK))) & 1; \
        (V)[((I) / BITSPERCHUNK) * (N)] ^= d << ((I) % BITSPERCHUNK); (V)[((J) / BITSPERCHUNK) * (N)] ^= d << ((J) % BITSPERCHUNK); })

typedef uint64_t chunk;
typedef uint16_t var;
typedef uint32_t dim;

typedef struct {
	dim n, m, c, s;
	chunk* data;
	var* vars;
} func;

#endif /* JSC_H_ */
