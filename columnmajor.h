#ifndef COLUMNMAJOR_H_
#define COLUMNMAJOR_H_

void randomdata(func *f);

void printrow(const func *f, dim i);

void print(const func *f, const chunk *s = NULL);

void shared2least(func *f, chunk* m);

void reordershared(func *f, id *vars);

dim uniquecombinations(func *f);

void histogram(func *f);

#endif  /* COLUMNMAJOR_H_ */
