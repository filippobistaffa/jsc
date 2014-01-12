#ifndef QSORT_H_
#define QSORT_H_

#ifdef __cplusplus
extern "C"
#endif
void sort(func f);

#ifdef __cplusplus
extern "C"
#endif
void sharedrows(func f1, func f2);

#ifdef __cplusplus
extern "C"
#endif
void removenonshared(func *f1, func *f2);

#endif  /* QSORT_H_ */
