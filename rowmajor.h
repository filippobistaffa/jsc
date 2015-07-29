#ifndef ROWMAJOR_H_
#define ROWMAJOR_H_

void randomdata(func *f);

void printrow(const func *f, dim i);

void print(const func *f, const char *title = NULL, const chunk *s = NULL);

void shared2least(const func *f, chunk* m);

void reordershared(const func *f, id *vars);

dim uniquecombinations(const func *f, dim idx = 0);

void histogram(const func *f, dim idx = 0);

void markmatchingrows(const func *f1, const func *f2, dim *n1, dim *n2, dim *hn);

void copymatchingrows(func *f1, func *f2, dim n1, dim n2, dim hn);

#endif  /* ROWMAJOR_H_ */
