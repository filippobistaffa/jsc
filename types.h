#ifndef TYPES_H_
#define TYPES_H_

#include <stdint.h>

typedef uint32_t id;
#define MAXID UINT_MAX
typedef float value;

typedef uint64_t chunk;
#define BITSPERCHUNK 64
typedef uint32_t dim;

typedef struct {
        chunk *data, mask, *rmask, *hmask;
        dim n, m, c, s, *h, hn;
        id *vars;
        value *v;
} func;

#endif /* TYPES_H_ */

