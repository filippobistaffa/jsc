#ifndef TYPES_H_
#define TYPES_H_

#include <stdint.h>

typedef uint16_t id;
#define MAXID USHRT_MAX
typedef uint16_t value;
#define MAXVALUE USHRT_MAX

typedef uint32_t chunk;
#define BITSPERCHUNK 32
#define ZERO 0U
#define ONE 1U
#define BITFORMAT "%" WIDTH "u"
typedef uint32_t dim;

typedef struct {
	chunk *data, mask, *hmask;
	dim n, m, c, s, *h, hn;
	value *v;
	id *vars;
} func;

#endif /* TYPES_H_ */

