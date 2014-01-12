#ifndef QSORT_H_
#define QSORT_H_

#ifdef __cplusplus
extern "C"
#endif
inline int compare(chunk* a, chunk* b, func f, func g);

#ifdef __cplusplus
extern "C"
#endif
void sort(func f);

#endif  /* QSORT_H_ */
