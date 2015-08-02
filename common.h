#ifndef COMMON_H_
#define COMMON_H_

template <typename type>
__attribute__((always_inline)) inline
void exclprefixsum(type *hi, type *ho, unsigned hn) {

	if (hn) {
		register unsigned i;
		ho[0] = 0;
		for (i = 1; i < hn; i++) ho[i] = hi[i - 1] + ho[i - 1];
	}
}

__attribute__((always_inline)) inline
void printmask(const chunk *m, dim n) {

	register dim i;
	for (i = 0; i < n; i++) printf(GET(m, i) ? "1 " : "0 ");
	printf("\n");
}

#include <iostream>
template <typename type>
__attribute__((always_inline)) inline
void printbuf(const type *buf, unsigned n, const char *name) {

	printf("%s = [ ", name);
	while (n--) std::cout << *(buf++) << " ";
	printf("]\n");
}

#ifdef __cplusplus
extern "C"
#endif
void sharedmasks(func *f1, chunk* s1, func *f2, chunk* s2);

#ifdef __cplusplus
extern "C"
#endif
void transpose(chunk *data, dim r, dim c);

#ifdef __cplusplus
extern "C"
#endif
void randomvars(func *f);

#ifdef __cplusplus
extern "C"
#endif
void randomvalues(func *f);

#ifndef __cplusplus
void prefixsum(dim *hi, dim *ho, dim hn);
#endif

#ifndef __cplusplus
void histogramproduct(dim *h1, dim *h2, dim *ho, dim hn);
#endif

unsigned crc32func(const func *f);

#endif  /* COMMON_H_ */
