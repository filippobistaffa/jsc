#ifndef COMMON_H_
#define COMMON_H_

template <typename type>
__attribute__((always_inline)) inline
void exclprefixsum(const type *hi, type *ho, unsigned hn) {

	if (hn) {
		ho[0] = 0;
		for (unsigned i = 1; i < hn; i++)
			ho[i] = hi[i - 1] + ho[i - 1];
	}
}

template <typename type>
__attribute__((always_inline)) inline
void bufproduct(const type *h1, const type *h2, type *ho, unsigned hn) {

	if (hn) {
		for (unsigned i = 0; i < hn; i++)
			ho[i] = h1[i] * h2[i];
	}
}

template <typename type>
__attribute__((always_inline)) inline
type sumreduce(const type *buf, unsigned n) {

	register type sum = 0;

	if (n) {
		for (unsigned i = 0; i < n; i++)
			sum += buf[i];
	}

	return sum;
}

__attribute__((always_inline)) inline
void printmask(const chunk *m, dim n) {

	for (dim i = 0; i < n; i++) printf(GET(m, i) ? "1 " : "0 ");
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
