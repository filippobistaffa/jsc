#ifndef COMPARE_H_
#define COMPARE_H_

//#define CMP(X, Y) (((X) == (Y)) ? 0 : (((X) < (Y)) ? -1 : 1))
#define CMP(X, Y) ((X) > (Y)) - ((X) < (Y))

#define COMPARE(A, B, S, M) (2 * (memcmp(A, B, sizeof(chunk) * DIVBPC(S))) + (MODBPC(S) ? CMP((A)[DIVBPC(S)] & (M), (B)[DIVBPC(S)] & (M)) : 0))

#define INTERSECTMASK(F1, I, F2, J, S, M, T1, T2) ({ register const dim cs = CEILBPC(S); register const dim ds = DIVBPC(S); \
						     MASKAND(CARE(F1, I), CARE(F2, J), T2, cs); \
						     MASKAND(DATA(F1, I), T2, T1, cs); MASKAND(DATA(F2, J), T2, T2, cs); \
						     if (M) { (T1)[ds] &= (M); (T2)[ds] &= (M); } \
						     register bool its = true; MASKXOR(T1, T2, T1, cs); \
						     for (dim i = 0; i < cs; i++) if ((T1)[i]) { its = false; break; } its; })

#define INTERSECT(F1, I, F2, J, T1, T2) INTERSECTMASK(F1, I, F2, J, (F1)->s, (F1)->mask, T1, T2)

#endif  /* COMPARE_H_ */
