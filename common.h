#ifndef COMMON_H_
#define COMMON_H_

#define MAX(_x, _y) ((_x) > (_y) ? (_x) : (_y))
#define MIN(_x, _y) ((_x) < (_y) ? (_x) : (_y))

#define DIVBPC(x) ((x) / BITSPERCHUNK)
#define MODBPC(x) ((x) % BITSPERCHUNK)
#define CEILBPC(x) CEIL(x, BITSPERCHUNK)

#define SET(V, I) ((V)[DIVBPC(I)] |= 1ULL << MODBPC(I)) // Row-major SET
#define CLEAR(V, I) ((V)[DIVBPC(I)] &= ~(1ULL << MODBPC(I))) // Row-major CLEAR

#define CMP(X, Y) ((X) == (Y) ? 0 : ((X) > (Y) ? 1 : -1))
#define CEIL(X, Y) (1 + (((X) - 1) / (Y)))

#define GETBIT(V, I) (((V) >> (I)) & 1)
#define GETC(V, I, N) ((V)[DIVBPC(I) * (N)] >> MODBPC(I) & 1)
#define GETR(V, I) ((V)[DIVBPC(I)] >> MODBPC(I) & 1)
#define GETMACRO(_1, _2, _3, NAME, ...) NAME
#define GET(...) GETMACRO(__VA_ARGS__, GETC, GETR)(__VA_ARGS__)

#define ALLOCFUNC(F) do { (F)->c = CEIL((F)->m, BITSPERCHUNK); \
			  (F)->vars = (id *)malloc(sizeof(id) * (F)->m); \
			  (F)->v = (value *)calloc((F)->n, sizeof(value)); \
			  (F)->data = (chunk *)calloc(1, sizeof(chunk) * (F)->n * (F)->c); \
			  (F)->care = (chunk **)calloc((F)->n, sizeof(chunk *)); } while (0)

#define RANDOMFUNC(F) do { randomdata(F); randomvars(F); randomvalues(F); } while (0)
#define FREEFUNC(F) do { free((F)->vars); free((F)->data); free((F)->v); register dim _i; \
			 for (_i = 0; _i < (F)->n; _i++) if ((F)->care[_i]) free((F)->care[_i]); free((F)->care); } while (0)

#define MASKOR(A, B, R, C) do { register dim _i; for (_i = 0; _i < (C); _i++) (R)[_i] = (A)[_i] | (B)[_i]; } while (0)
#define MASKXOR(A, B, R, C) do { register dim _i; for (_i = 0; _i < (C); _i++) (R)[_i] = (A)[_i] ^ (B)[_i]; } while (0)
#define MASKANDNOT(A, B, R, C) do { register dim _i; for (_i = 0; _i < (C); _i++) (R)[_i] = (A)[_i] & ~(B)[_i]; } while (0)
#define MASKNOTAND(A, B, R, C) do { register dim _i; for (_i = 0; _i < (C); _i++) (R)[_i] = ~(A)[_i] & (B)[_i]; } while (0)
#define MASKPOPCNT(A, C) ({ register dim _i, _c = 0; for (_i = 0; _i < (C); _i++) _c += __builtin_popcountll((A)[_i]); _c; })
#define MASKFFS(A, C) ({ register dim _i = 0, _ffs = 0; register const chunk *_buf = (A); \
			 while (!(*_buf) && _i < (C)) { _ffs += BITSPERCHUNK; _buf++; _i++; } \
			 if (_i == (C)) _ffs = 0; else _ffs += __builtin_ffsll(*_buf) - 1; _ffs; })
#define MASKCLEARANDFFS(A, B, C) ({ CLEAR(A, B); MASKFFS(A, C); })

template <typename type>
__attribute__((always_inline)) inline
void exclprefixsum(type *hi, type *ho, unsigned hn) {

	if (hn) {
		register unsigned i;
		ho[0] = 0;
		for (i = 1; i < hn; i++) ho[i] = hi[i - 1] + ho[i - 1];
	}
}

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
void randomvars(func *f);

#ifdef __cplusplus
extern "C"
#endif
void randomvalues(func *f);

#ifndef __cplusplus
void prefixsum(dim *hi, dim *ho, dim hn);
#endif

#ifndef __cplusplus
void histogramproduct(dim *h1, dim *h2, dim *ho, dim hn);
#endif

#endif  /* COMMON_H_ */
