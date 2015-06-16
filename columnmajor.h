#ifndef COLUMNMAJOR_H_
#define COLUMNMAJOR_H_

#define SWAP(V, I, J, N) do { \
        register chunk d = GET(V, I, N) ^ GET(V, J, N); \
        (V)[((I) / BITSPERCHUNK) * (N)] ^= d << ((I) % BITSPERCHUNK); (V)[((J) / BITSPERCHUNK) * (N)] ^= d << ((J) % BITSPERCHUNK); } while (0)

#include "qsort.h"

#ifdef __cplusplus
extern "C"
#endif
void markmatchingrows(func f1, func f2, dim *n1, dim *n2, dim *hn);

#ifdef __cplusplus
extern "C"
#endif
void copymatchingrows(func *f1, func *f2, dim n1, dim n2, dim hn); // Fast, but not in-place

#ifdef __cplusplus
extern "C"
#endif
void removenonmatchingrows(func *f1, func *f2); // Very slow, but in-place

#ifdef __cplusplus
extern "C"
#endif
void reordershared(func f, id *vars);

#ifdef __cplusplus
extern "C"
#endif
void shared2least(func f, chunk* m);

#ifdef __cplusplus
extern "C"
#endif
void histogram(func f);

#ifdef __cplusplus
extern "C"
#endif
void invhistogram(func f);

#ifdef __cplusplus
extern "C"
#endif
dim uniquecombinations(func f);

#ifdef __cplusplus
extern "C"
#endif
dim invuniquecombinations(func f);

#ifdef __cplusplus
extern "C"
void print(func f, chunk *s = NULL);
#else
void print(func f, chunk *s);
#endif

#ifdef __cplusplus
extern "C"
#endif
void randomdata(func f);

#endif  /* COLUMNMAJOR_H_ */
