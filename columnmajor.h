#ifndef COLUMNMAJOR_H_
#define COLUMNMAJOR_H_

#define SWAP(V, I, J, N) do { \
        register chunk d = GET(V, I, N) ^ GET(V, J, N); \
        (V)[((I) / BITSPERCHUNK) * (N)] ^= d << ((I) % BITSPERCHUNK); (V)[((J) / BITSPERCHUNK) * (N)] ^= d << ((J) % BITSPERCHUNK); } while (0)

void markmatchingrows(func f1, func f2, dim *n1, dim *n2, dim *hn);

void removenonmatchingrows(func *f1, func *f2); // Very slow, but in-place

void reordershared(func f, id *vars);

void shared2least(func f, chunk* m);

void histogram(func f);

void invhistogram(func f);

dim uniquecombinations(func f);

dim invuniquecombinations(func f);

void print(func f, chunk *s = NULL);

void randomdata(func f);

#endif  /* COLUMNMAJOR_H_ */
