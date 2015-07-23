#ifndef COLUMNMAJOR_H_
#define COLUMNMAJOR_H_

void randomdata(func *f);

void printrow(const func *f, dim i);

void print(const func *f, const char *title = NULL, const chunk *s = NULL);

void shared2least(func *f, chunk* m);

void reordershared(func *f, id *vars);

dim uniquecombinations(const func *f, dim idx = 0);

void histogram(const func *f, dim idx = 0);

dim intuniquecombinations(const func *f);

void inthistogram(const func *f);

#endif  /* COLUMNMAJOR_H_ */
