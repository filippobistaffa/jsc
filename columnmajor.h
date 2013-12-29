#ifndef COLUMNMAJOR_H_
#define COLUMNMAJOR_H_

#define GET(V, I, N) ((V)[((I) / BITSPERCHUNK) * (N)] >> ((I) % BITSPERCHUNK) & 1)
#define SWAP(V, I, J, N) do { \
        register chunk d = GET(V, I, N) ^ GET(V, J, N); \
        (V)[((I) / BITSPERCHUNK) * (N)] ^= d << ((I) % BITSPERCHUNK); (V)[((J) / BITSPERCHUNK) * (N)] ^= d << ((J) % BITSPERCHUNK); } while (0)

#include "qsort.h"

void reordershared(func f, var *vars);
void shared2least(func f, chunk* m);
dim uniquecombinations(func f);
void print(func f, chunk *s);
void randomdata(func f);

#endif  /* COLUMNMAJOR_H_ */
