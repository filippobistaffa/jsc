#ifndef COMMON_H_
#define COMMON_H_

#define SET(V, I) ((V)[(I) / BITSPERCHUNK] |= 1ULL << ((I) % BITSPERCHUNK)) // Row-major SET
#define CMP(X, Y) ((X) == (Y) ? 0 : ((X) > (Y) ? 1 : -1))

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

#endif  /* COMMON_H_ */
