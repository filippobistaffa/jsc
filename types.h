#ifndef TYPES_H_
#define TYPES_H_

#include <stdint.h>

typedef uint16_t id;
#define MAXID USHRT_MAX
typedef uint16_t value;
#define MAXVALUE USHRT_MAX

typedef uint64_t chunk;
#define BITSPERCHUNK 64
typedef uint32_t dim;

typedef struct {
	chunk *data, mask, *hmask;
	dim n, m, c, s, *h, hn;
	value *v;
	id *vars;
} func;

#endif /* TYPES_H_ */

