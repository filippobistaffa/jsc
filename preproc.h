#ifndef PREPROC_H_
#define PREPROC_H_

void sharedmasks(func *f1, chunk* s1, func *f2, chunk* s2);
void reordershared(func f, var *vars);
void shared2least(func f, chunk* m);
void row2columnmajor(func f);
void pqsort(func f);

#endif  /* PREPROC_H_ */
