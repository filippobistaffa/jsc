#ifndef QSORT_H_
#define QSORT_H_

#define COMPARE(A, B, F, G) ({ register dim i; register char cmp = 0; for (i = 0; i < (F).s / BITSPERCHUNK; i++) if ((cmp = CMP((A)[i * (F).n], (B)[i * (G).n]))) break; \
			       if (!cmp) cmp = ((F).mask ? CMP((F).mask & (A)[((F).s / BITSPERCHUNK) * (F).n], (F).mask & (B)[((F).s / BITSPERCHUNK) * (G).n]) : 0); cmp; })

#ifdef __cplusplus
extern "C"
#endif
void sort(func f);

#endif  /* QSORT_H_ */
