#ifndef COMPARE_H_
#define COMPARE_H_

//#define CMP(X, Y) (((X) == (Y)) ? 0 : (((X) < (Y)) ? -1 : 1))
#define CMP(X, Y) ((X) > (Y)) - ((X) < (Y))

#define COMPARE(A, B, S, M) (memcmp(A, B, sizeof(chunk) * DIVBPC(S)) + (MODBPC(S) ? CMP((A)[DIVBPC(S)] & (M), (B)[DIVBPC(S)] & (M)) : 0))

#endif  /* COMPARE_H_ */
