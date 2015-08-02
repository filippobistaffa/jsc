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

/*__attribute__((always_inline)) inline
void instancezeros(func *f) {

	register UT_array *dataUT;
	utarray_new(dataUT, &chunk_icd);

	chunk maskstack[(f->m + 1) * f->c];
	memset(maskstack, 0, sizeof(chunk) * f->c);
	ONES(maskstack, f->m, f->c);
	func fstack[f->m + 1];
	fstack->n = 1;
	fstack->m = f->m;
	ALLOCFUNC(fstack);
	dim rowmap[f->n];
	for (dim i = 0; i < f->n; i++) rowmap[i] = i;
	register const dim ext = computerows(maskstack, f->c, f->m, fstack, f, f->m, rowmap, rowmap, f->n, dataUT);
	//printf("ext = %u\n", ext);

	if (ext) {
		f->data = (chunk *)realloc(f->data, sizeof(chunk) * 2 * f->c * (f->n + ext));
		f->v = (value *)realloc(f->v, sizeof(value) * (f->n + ext));
		memcpy(f->data + 2 * f->c * f->n, dataUT->d, sizeof(chunk) * 2 * f->c * ext);
		memset(f->v + f->n, 0, sizeof(value) * ext);
		f->n += ext;
	}

	utarray_free(dataUT);
}*/
