#ifndef TYPES_H_
#define TYPES_H_

typedef uint32_t id;
typedef float value;

typedef uint64_t chunk;
typedef uint32_t dim;

typedef struct {
        chunk *data, mask, *rmask, *hmask;
        dim n, m, c, s, *h, hn;
        id *vars;
        value *v;
} func;

#endif /* TYPES_H_ */

