#include "jsc.h"

void randomvars(func f) {

	assert(MAXVAR > f.m);
	register dim i, j;
	register id v;

	for (i = 0; i < f.m; i++) {
		random:
		v = rand() % MAXVAR;
		for (j = 0; j < i; j++)
			if (f.vars[j] == v)
			goto random;
		f.vars[i] = v;
	}
}

void randomvalues(func f) {

	register dim i;
	for (i = 0; i < f.n; i++) f.v[i] = (value)rand()*MAXVALUE/RAND_MAX;
}

void sharedmasks(func *f1, chunk* s1, func *f2, chunk* s2) {

	register dim i, j;
	f1->s = f2->s = 0;

	for (i = 0; i < f1->m; i++)
		for (j = 0; j < f2->m; j++)
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

	register dim i;
	chunk *t = malloc(sizeof(chunk) * n2);

	for (i = 0; i < c - 1; i++) {
		memcpy(t, data + i * (n1 + n2) + (c - i) * n1, sizeof(chunk) * n2);
		memmove(data + (i + 1) * (n1 + n2), data + i * (n1 + n2) + n1, sizeof(chunk) * (c - i - 1) * n1);
		memcpy(data + i * (n1 + n2) + n1, t, sizeof(chunk) * n2);
	}

	free(t);
}

__attribute__((always_inline))
inline void parallelmove(chunk *data, dim c, dim exp) {

	register dim i, j, k, h = 1ULL << (exp - 1);
	for (i = 0, k = 1; i < exp; i++, k <<= 1, h >>= 1)
	#pragma omp parallel for private(j)
	for (j = 0; j < h; j++) move(data + 2 * j * c * k, c, k, k);
}

void transpose(chunk *data, dim r, dim c) {

	if (c > 1) {
		register dim j, i = 0, nt = r, n = __builtin_popcountll(r);
		dim count[n];
		while (nt) nt ^= 1ULL << (count[n - 1 - i++] = __builtin_ctzll(nt));
		puts("Parallel moves...");
		for (i = 0, j = 0; i < n; i++, j += 1ULL << count[i - 1]) parallelmove(data + j * c, c, count[i]);
		puts("Single moves...");
		for (i = 1, j = 1ULL << count[0]; i < n; i++, j += 1ULL << count[i - 1]) move(data, c, j, 1ULL << count[i]);
	}
}

void prefixsum(dim *hi, dim *ho, dim hn) {

	register dim i;
	ho[0] = hi[0];
	for (i = 1; i < hn; i++) ho[i] = hi[i] + ho[i - 1];
}

void histogramproduct(dim *h1, dim *h2, dim *ho, dim hn) {

	register dim i;
	for (i = 0; i < hn; i++) ho[i] = h1[i] * h2[i];
}
