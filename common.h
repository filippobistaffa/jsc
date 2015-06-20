#ifndef COMMON_H_
#define COMMON_H_

#define MAX(_x, _y) ((_x) > (_y) ? (_x) : (_y))
#define MIN(_x, _y) ((_x) < (_y) ? (_x) : (_y))

#define SET(V, I) ((V)[(I) / BITSPERCHUNK] |= 1ULL << ((I) % BITSPERCHUNK)) // Row-major SET
#define CMP(X, Y) ((X) == (Y) ? 0 : ((X) > (Y) ? 1 : -1))
#define CEIL(X, Y) (1 + (((X) - 1) / (Y)))

#define GETBIT(V, I) (((V) >> (I)) & 1)
#define GETC(V, I, N) ((V)[((I) / BITSPERCHUNK) * (N)] >> ((I) % BITSPERCHUNK) & 1)
#define GETR(V, I) ((V)[(I) / BITSPERCHUNK] >> ((I) % BITSPERCHUNK) & 1)
#define GETMACRO(_1, _2, _3, NAME, ...) NAME
#define GET(...) GETMACRO(__VA_ARGS__, GETC, GETR)(__VA_ARGS__)

#define OUTPUTC CEIL(f1.m + f2.m - f1.s, BITSPERCHUNK)

#define ALLOCFUNC(F, DATATYPE, VARTYPE, VALUETYPE) do { (F).c = CEIL((F).m, BITSPERCHUNK); \
							(F).vars = (VARTYPE *)malloc(sizeof(VARTYPE) * (F).m); \
							(F).v = (VALUETYPE *)calloc((F).n, sizeof(VALUETYPE)); \
							(F).data = (DATATYPE *)calloc(1, sizeof(DATATYPE) * (F).n * (F).c); \
							(F).care = (chunk **)calloc((F).n, sizeof(chunk *)); } while (0)

#define RANDOMFUNC(F) do { randomdata(F); randomvars(F); randomvalues(F); } while (0)
#define FREEFUNC(F) do { free((F).vars); free((F).data); free((F).v); register dim _i; \
			 for (_i = 0; _i < (F).n; _i++) if ((F).care[_i]) free((F).care[_i]); free((F).care); } while (0)

#define DIVBPC(x) ((x) / BITSPERCHUNK)
#define MODBPC(x) ((x) % BITSPERCHUNK)

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
