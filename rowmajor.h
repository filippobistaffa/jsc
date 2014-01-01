#ifndef ROWMAJOR_H_
#define ROWMAJOR_H_

#define GET(V, I) ((V)[(I) / BITSPERCHUNK] >> ((I) % BITSPERCHUNK) & 1)
#define SWAP(V, I, J) do { \
	register chunk d = GET(V, I) ^ GET(V, J); \
	(V)[(I) / BITSPERCHUNK] ^= d << ((I) % BITSPERCHUNK); (V)[(J) / BITSPERCHUNK] ^= d << ((J) % BITSPERCHUNK); } while (0)

inline int compare(const void* a, const void* b, void* c);
void reordershared(func f, var *vars);
void shared2least(func f, chunk* m);
dim uniquecombinations(func f);
void print(func f, chunk *s);
void randomdata(func f);
void sort(func f);

#endif  /* ROWMAJOR_H_ */
