#ifndef COLUMNMAJOR_H_
#define COLUMNMAJOR_H_

#define SWAP(V, CARE, I, J, N) do { register chunk d = GET(V, I, N) ^ GET(V, J, N); (V)[DIVBPC(I) * (N)] ^= d << MODBPC(I); (V)[DIVBPC(J) * (N)] ^= d << MODBPC(J); \
				    if (CARE) { d = GET(CARE, I) ^ GET(CARE, J); (CARE)[DIVBPC(I)] ^= d << MODBPC(I); (CARE)[DIVBPC(J)] ^= d << MODBPC(J); } } while (0)

void markmatchingrows(func f1, func f2, dim *n1, dim *n2, dim *hn);

void instancedontcare(func *f, chunk* m);

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
