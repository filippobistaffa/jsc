#ifndef QSORT_H_
#define QSORT_H_

// Compare row at address A in funcion F with row at address B in function G

#define COMPARE(A, B, F, G) ({ register dim i; register char cmp = 0; for (i = 0; i < DIVBPC((F).s); i++) if ((cmp = CMP((A)[i * (F).n], (B)[i * (G).n]))) break; \
			       if (!cmp) cmp = ((F).mask ? CMP((F).mask & (A)[(DIVBPC((F).s)) * (F).n], (F).mask & (B)[(DIVBPC((F).s)) * (G).n]) : 0); cmp; })

#define INVCOMPARE(A, B, F, G) ({ register char cmp = (F).mask ? CMP((A)[(DIVBPC((F).s)) * (F).n] >> MODBPC((F).s), (B)[(DIVBPC((F).s)) * (G).n] >> MODBPC((F).s)) : 0; \
				  register dim i; if (!cmp) for (i = DIVBPC((F).s) + (f.mask ? 1 : 0); i < (F).c; i++) if ((cmp = CMP((A)[i * (F).n], (B)[i * (G).n]))) break; cmp; })

#ifdef __cplusplus
extern "C"
#endif
void sort(func f);

#ifdef __cplusplus
extern "C"
#endif
void invsort(func f);

#endif  /* QSORT_H_ */
