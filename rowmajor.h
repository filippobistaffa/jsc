#ifndef ROWMAJOR_H_
#define ROWMAJOR_H_

#define SWAP(V, I, J) do { \
	register chunk d = GET(V, I) ^ GET(V, J); \
	(V)[(I) / BITSPERCHUNK] ^= d << ((I) % BITSPERCHUNK); (V)[(J) / BITSPERCHUNK] ^= d << ((J) % BITSPERCHUNK); } while (0)

#ifdef __cplusplus
extern "C"
#endif
inline int compare(const void* a, const void* b, void* c);

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
dim uniquecombinations(func f);

#ifdef __cplusplus
extern "C"
#endif
void print(func f, chunk *s);

#ifdef __cplusplus
extern "C"
#endif
void randomdata(func f);

#ifdef __cplusplus
extern "C"
#endif
void sort(func f);

#endif  /* ROWMAJOR_H_ */
