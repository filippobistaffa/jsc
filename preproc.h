#ifndef PREPROC_H_
#define PREPROC_H_

#define CMP(X, Y) ((X) == (Y) ? 0 : ((X) > (Y) ? 1 : -1))

#define GET_RM(V, I) ((V)[(I) / BITSPERCHUNK] >> ((I) % BITSPERCHUNK))
#define GET_CM(V, I, N) ((V)[((I) / BITSPERCHUNK) * (N)] >> ((I) % BITSPERCHUNK))

#define SET_RM(V, I) ((V)[(I) / BITSPERCHUNK] |= 1ULL << ((I) % BITSPERCHUNK))
#define SET_CM(V, I, N) ((V)[((I) / BITSPERCHUNK) * (N)] |= 1ULL << ((I) % BITSPERCHUNK))

#define SWAP_RM(V, I, J) do { \
	register chunk d = (GET_RM(V, I) ^ GET_RM(V, J)) & 1; \
	(V)[(I) / BITSPERCHUNK] ^= d << ((I) % BITSPERCHUNK); (V)[(J) / BITSPERCHUNK] ^= d << ((J) % BITSPERCHUNK); } while (0)

#define SWAP_CM(V, I, J, N) do { \
        register chunk d = (GET_CM(V, I, N) ^ GET_CM(V, J, N)) & 1; \
        (V)[((I) / BITSPERCHUNK) * (N)] ^= d << ((I) % BITSPERCHUNK); (V)[((J) / BITSPERCHUNK) * (N)] ^= d << ((J) % BITSPERCHUNK); } while (0)

void sharedmasks(func *f1, chunk* s1, func *f2, chunk* s2);
void transpose(chunk *data, dim r, dim c);

void reordershared(func f, var *vars);
void shared2least(func f, chunk* m);
dim uniquecombinations(func f);
inline int compare(const void* a, const void* b, void* c);
void pqsort(func f);

void reordershared_cm(func f, var *vars);
void shared2least_cm(func f, chunk* m);
dim uniquecombinations_cm(func f);

#endif  /* PREPROC_H_ */
