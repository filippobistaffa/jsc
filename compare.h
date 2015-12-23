#ifndef COMPARE_H_
#define COMPARE_H_

//#define CMP(X, Y) (((X) == (Y)) ? 0 : (((X) < (Y)) ? -1 : 1))
#define CMP(X, Y) ((X) > (Y)) - ((X) < (Y))

#define COMPARE(A, B, C, S, M) (2 * memcmp(A, B, sizeof(chunk) * DIVBPC(S)) + (MODBPC(S) ? CMP((A)[DIVBPC(S)] & (M), (B)[DIVBPC(S)] & (M)) : 0))

/*#define COMPARE(A, B, C, S, M) ({ \
					chunk _t1 = 0, _t2 = 0; \
					if (M) { \
						_t1 = (A)[DIVBPC(S)] & ~(M); \
						_t2 = (B)[DIVBPC(S)] & ~(M); \
						(A)[DIVBPC(S)] &= (M); \
						(B)[DIVBPC(S)] &= (M); \
					} \
					int _cmp = memcmp(A, B, sizeof(chunk) * CEILBPC(S)); \
					if (M) { \
						(A)[DIVBPC(S)] |= _t1; \
						(B)[DIVBPC(S)] |= _t2; \
					} \
					_cmp; \
				})*/

#define INVCOMPARE(A, B, C, S, M) (2 * memcmp((A) + CEILBPC(S), (B) + CEILBPC(S), sizeof(chunk) * ((C) - CEILBPC(S))) + (MODBPC(S) ? CMP((A)[DIVBPC(S)] & ~(M), (B)[DIVBPC(S)] & ~(M)) : 0))

/*#define INVCOMPARE(A, B, C, S, M) ({ \
					chunk _t1 = 0, _t2 = 0; \
					if (M) { \
						_t1 = (A)[DIVBPC(S)] & (M); \
						_t2 = (B)[DIVBPC(S)] & (M); \
						(A)[DIVBPC(S)] &= ~(M); \
						(B)[DIVBPC(S)] &= ~(M); \
					} \
					int _cmp = memcmp(A + DIVBPC(S), B + DIVBPC(S), sizeof(chunk) * ((C) - DIVBPC(S))); \
					if (M) { \
						(A)[DIVBPC(S)] |= _t1; \
						(B)[DIVBPC(S)] |= _t2; \
					} \
					_cmp; \
				})*/

#endif  /* COMPARE_H_ */
