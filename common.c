#include "jsc.h"

/*__attribute__((always_inline)) inline
void shuffle(void *array, size_t n, size_t size) {

	uint8_t tmp[size];
	uint8_t *arr = (uint8_t *)array;

	if (n > 1) {
		size_t i;
		for (i = 0; i < n - 1; ++i) {
			size_t rnd = (size_t) rand();
			size_t j = i + rnd / (RAND_MAX / (n - i) + 1);
			memcpy(tmp, arr + j * size, size);
			memcpy(arr + j * size, arr + i * size, size);
			memcpy(arr + i * size, tmp, size);
		}
	}
}

void randomvars(func *f) {

	assert(MAXVAR > f->m);
	id vars[MAXVAR];
	register dim i;

	for (i = 0; i < MAXVAR; i++) vars[i] = i;
	shuffle(vars, MAXVAR, sizeof(id));
	memcpy(f->vars, vars, sizeof(id) * f->m);
}

void randomvalues(func *f) {

	register dim i;
	for (i = 0; i < f->n; i++) f->v[i] = (value)rand() * MAXVALUE / RAND_MAX;
}*/

void sharedmasks(func *f1, chunk* s1, func *f2, chunk* s2) {

	f1->s = f2->s = 0;

	for (dim i = 0; i < f1->m; i++)
		for (dim j = 0; j < f2->m; j++)
			if (f1->vars[i] == f2->vars[j]) {
				SET(s1, i);
				SET(s2, j);
				(f1->s)++;
				(f2->s)++;
				break;
			}
}

__attribute__((always_inline))
inline void move(chunk *data, dim c, dim n1, dim n2) {

	chunk *t = (chunk *)malloc(sizeof(chunk) * n2);

	for (dim i = 0; i < c - 1; i++) {
		memcpy(t, data + i * (n1 + n2) + (c - i) * n1, sizeof(chunk) * n2);
		memmove(data + (i + 1) * (n1 + n2), data + i * (n1 + n2) + n1, sizeof(chunk) * (c - i - 1) * n1);
		memcpy(data + i * (n1 + n2) + n1, t, sizeof(chunk) * n2);
	}

	free(t);
}

__attribute__((always_inline)) inline
void parallelmove(chunk *data, dim c, dim exp) {

	register dim i, j, k, h = ONE << (exp - 1);
	for (i = 0, k = 1; i < exp; i++, k <<= 1, h >>= 1)
	#pragma omp parallel for private(j)
	for (j = 0; j < h; j++) move(data + 2 * j * c * k, c, k, k);
}

void transpose(chunk *data, dim r, dim c) {

	if (c > 1 && r > 1) {
		register dim j, i = 0, nt = r, n = __builtin_popcountll(r);
		dim count[n];
		while (nt) nt ^= ONE << (count[n - 1 - i++] = __builtin_ctzll(nt));
		puts("Parallel moves...");
		for (i = 0, j = 0; i < n; i++, j += ONE << count[i - 1]) parallelmove(data + j * c, c, count[i]);
		puts("Single moves...");
		for (i = 1, j = ONE << count[0]; i < n; i++, j += ONE << count[i - 1]) move(data, c, j, ONE << count[i]);
	}
}

void prefixsum(const dim *hi, dim *ho, dim hn) {

	ho[0] = hi[0];

	for (dim i = 1; i < hn; i++)
		ho[i] = hi[i] + ho[i - 1];
}

unsigned crc32func(const func *f) {

	unsigned crc[2] = { 0 };
	crc[0] = crc32(f->data, sizeof(chunk) * f->n * f->c);
	crc[1] = crc32(f->v, sizeof(value) * f->n);
	//printf("%u %u\n", crc[0], crc[1]);
	return crc32(crc, sizeof(unsigned) * 2);
}
