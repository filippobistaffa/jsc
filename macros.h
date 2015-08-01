#ifndef MACROS_H_
#define MACROS_H_

#define MAX(_x, _y) ((_x) > (_y) ? (_x) : (_y))
#define MIN(_x, _y) ((_x) < (_y) ? (_x) : (_y))

#define DIVBPC(x) ((x) / BITSPERCHUNK)
#define MODBPC(x) ((x) % BITSPERCHUNK)
#define CEILBPC(x) CEIL(x, BITSPERCHUNK)
#define CEIL(X, Y) (1 + (((X) - 1) / (Y)))

#define SET(V, I) ((V)[DIVBPC(I)] |= 1ULL << MODBPC(I)) // Row-major SET
#define CLEAR(V, I) ((V)[DIVBPC(I)] &= ~(1ULL << MODBPC(I))) // Row-major CLEAR

#define GETBIT(V, I) (((V) >> (I)) & 1)
#define GETC(V, I, N) ((V)[DIVBPC(I) * (N)] >> MODBPC(I) & 1)
#define GETR(V, I) ((V)[DIVBPC(I)] >> MODBPC(I) & 1)
#define GETMACRO(_1, _2, _3, NAME, ...) NAME
#define GET(...) GETMACRO(__VA_ARGS__, GETC, GETR)(__VA_ARGS__)

#define DATA(F, I) ((F)->data + 2 * (I) * (F)->c)
#define CARE(F, I) ((F)->data + (2 * (I) + 1) * (F)->c)

#define COPYFIELDS(FO, FI) do { (FO)->s = (FI)->s; (FO)->mask = (FI)->mask; } while (0)
#define ALLOCFUNC(F) do { (F)->c = CEILBPC((F)->m); \
			  (F)->vars = (id *)malloc(sizeof(id) * (F)->m); \
			  (F)->v = (value *)calloc((F)->n, sizeof(value)); \
			  (F)->data = (chunk *)calloc(1, sizeof(chunk) * 2 * (F)->n * (F)->c); } while (0)

#define COPYFUNC(FO, FI, IDX) do { memcpy((FO)->v + (IDX), (FI)->v, sizeof(value) * (FI)->n); \
                                   memcpy((FO)->data + (IDX) * 2 * (FI)->c, (FI)->data, sizeof(chunk) * 2 * (FI)->c * (FI)->n); } while (0)

#define RANDOMFUNC(F) do { randomdata(F); randomvars(F); randomvalues(F); } while (0)
#define FREEFUNC(F) do { free((F)->vars); free((F)->data); free((F)->v); } while (0)

#define MASKOR(A, B, R, C) do { register dim _i; for (_i = 0; _i < (C); _i++) (R)[_i] = (A)[_i] | (B)[_i]; } while (0)
#define MASKAND(A, B, R, C) do { register dim _i; for (_i = 0; _i < (C); _i++) (R)[_i] = (A)[_i] & (B)[_i]; } while (0)
#define MASKXOR(A, B, R, C) do { register dim _i; for (_i = 0; _i < (C); _i++) (R)[_i] = (A)[_i] ^ (B)[_i]; } while (0)
#define MASKANDNOT(A, B, R, C) do { register dim _i; for (_i = 0; _i < (C); _i++) (R)[_i] = (A)[_i] & ~(B)[_i]; } while (0)
#define MASKNOTAND(A, B, R, C) do { register dim _i; for (_i = 0; _i < (C); _i++) (R)[_i] = ~(A)[_i] & (B)[_i]; } while (0)
#define MASKNOTANDNOT(A, B, R, C) do { register dim _i; for (_i = 0; _i < (C); _i++) (R)[_i] = ~(A)[_i] & ~(B)[_i]; } while (0)
#define MASKPOPCNT(A, C) ({ register dim _i, _c = 0; for (_i = 0; _i < (C); _i++) _c += __builtin_popcountll((A)[_i]); _c; })
#define MASKFFS(A, C) ({ register dim _i = 0, _ffs = 0; register const chunk *_buf = (A); \
			 while (!(*_buf) && _i < (C)) { _ffs += BITSPERCHUNK; _buf++; _i++; } \
			 if (_i == (C)) _ffs = 0; else _ffs += __builtin_ffsll(*_buf) - 1; _ffs; })
#define MASKCLEARANDFFS(A, B, C) ({ CLEAR(A, B); MASKFFS(A, C); })
#define MASKFFSANDCLEAR(A, C) ({ register dim _idx = MASKFFS(A, C); CLEAR(A, _idx); _idx; })

#define BREAKPOINT(MSG) do { puts(MSG); fflush(stdout); while (getchar() != '\n'); } while (0)

#define ONES(V, I, C) do { register dim _i; register const dim _mi = MODBPC(I); for (_i = 0; _i < (C); _i++) (V)[_i] = ~0ULL; \
			   if (_mi) (V)[(C) - 1] = (1ULL << _mi) - 1; } while (0)

#endif  /* MACROS_H_ */