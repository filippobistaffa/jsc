#ifndef COMMON_H_
#define COMMON_H_

#define SET(V, I) ((V)[(I) / BITSPERCHUNK] |= 1ULL << ((I) % BITSPERCHUNK)) // Row-major SET
#define CMP(X, Y) ((X) == (Y) ? 0 : ((X) > (Y) ? 1 : -1))
#define CEIL(X, Y) (1 + (((X) - 1) / (Y)))

#define GETC(V, I, N) ((V)[((I) / BITSPERCHUNK) * (N)] >> ((I) % BITSPERCHUNK) & 1)
#define GETR(V, I) ((V)[(I) / BITSPERCHUNK] >> ((I) % BITSPERCHUNK) & 1)
#define GETMACRO(_1, _2, _3, NAME, ...) NAME
#define GET(...) GETMACRO(__VA_ARGS__, GETC, GETR)(__VA_ARGS__)

#ifdef __cplusplus
extern "C"
#endif
void sharedmasks(func *f1, chunk* s1, func *f2, chunk* s2);

#ifdef __cplusplus
extern "C"
#endif
void transpose(chunk *data, dim r, dim c);

#ifdef __cplusplus
extern "C"
#endif
void randomvars(func f);

#ifdef __cplusplus
extern "C"
#endif
void randomvalues(func f);

#ifndef __cplusplus
void prefixsum(dim *hi, dim *ho, dim hn);
#endif

#ifndef __cplusplus
void histogramproduct(dim *h1, dim *h2, dim *ho, dim hn);
#endif

#endif  /* COMMON_H_ */
