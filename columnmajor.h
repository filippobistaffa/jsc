#ifndef COLUMNMAJOR_H_
#define COLUMNMAJOR_H_

#define GET(V, I, N) ((V)[((I) / BITSPERCHUNK) * (N)] >> ((I) % BITSPERCHUNK) & 1)
#define SWAP(V, I, J, N) do { \
        register chunk d = GET(V, I, N) ^ GET(V, J, N); \
        (V)[((I) / BITSPERCHUNK) * (N)] ^= d << ((I) % BITSPERCHUNK); (V)[((J) / BITSPERCHUNK) * (N)] ^= d << ((J) % BITSPERCHUNK); } while (0)

#include "qsort.h"

#ifdef __cplusplus
extern "C"
#endif
void reordershared(func f, var *vars);

#ifdef __cplusplus
extern "C"
#endif
void shared2least(func f, chunk* m);

#ifdef __cplusplus
extern "C"
#endif
void histogram(func f, dim *h);

#ifdef __cplusplus
extern "C"
#endif
dim uniquecombinations(func f);

#ifdef __cplusplus
extern "C"
#endif
void print(func f, chunk *s);

#ifdef __cplusplus
extern "C"
#endif
void randomdata(func f);

#endif  /* COLUMNMAJOR_H_ */
