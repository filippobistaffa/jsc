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
#define SWAP(V, I, J) \
({ if ((((V)[(I) / BITSPERCHUNK] >> ((I) % BITSPERCHUNK)) ^ ((V)[(J) / BITSPERCHUNK] >> ((J) % BITSPERCHUNK))) & 1) { \
	(V)[(I) / BITSPERCHUNK] ^= (1ULL << ((I) % BITSPERCHUNK)); \
	(V)[(J) / BITSPERCHUNK] ^= (1ULL << ((J) % BITSPERCHUNK)); }})

typedef uint64_t chunk;
typedef uint16_t var;
typedef uint16_t dim;

typedef struct {
	dim n, m, c, s;
	chunk* data;
	var* vars;
} func;

#endif /* JSC_H_ */
