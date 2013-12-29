#include "jsc.h"

void sharedmasks(func *f1, chunk* s1, func *f2, chunk* s2) {

	register dim i, j;
	f1->s = f2->s = 0;

	for (i = 0; i < f1->m; i++)
		for (j = 0; j < f2->m; j++)
			if (f1->vars[i] == f2->vars[j]) {
				SET_RM(s1, i);
				SET_RM(s2, j);
				(f1->s)++;
				(f2->s)++;
				break;
			}
}

__attribute__((always_inline))
inline int compare(const void* a, const void* b, void* c) {

	register func f = *(func *)c;
	register char cmp;
	register dim i;
	
	for (i = 0; i < f.s / BITSPERCHUNK; i++) if ((cmp = CMP(*((chunk *)a + i), *((chunk *)b + i)))) return cmp;
	if (!f.mask) return cmp;
	else {
		register chunk x = f.mask & *((chunk *)a + f.s / BITSPERCHUNK);
		register chunk y = f.mask & *((chunk *)b + f.s / BITSPERCHUNK);
		return CMP(x, y);
	}
}

void merge(func f, chunk *a, chunk *b, size_t m, size_t n) {

	register size_t i = 0, j = 0, k = 0;
	register size_t s = m + n;
	chunk *c = malloc(sizeof(chunk) * f.c * s);

	while (i < m && j < n)
	if (compare(a + f.c * i, b + f.c * j, &f) <= 0) memcpy(c + f.c * k++, a + f.c * i++, sizeof(chunk) * f.c);
	else memcpy(c + f.c * k++, b + f.c * j++, sizeof(chunk) * f.c);

	if (i < m) memcpy(c + f.c * k, a + f.c * i, sizeof(chunk) * f.c * (m - i));
	else memcpy(c + f.c * k, b + f.c * j, sizeof(chunk) * f.c * (n - j));
	memcpy(a, c, sizeof(chunk) * f.c * s);
	free(c);
}

void arraymerge(func f) {

	uint8_t t = CPUTHREADS;
	size_t in[t + 1];
	register dim i;

	while (t > 1) {
		for (i = 0; i < t; i++) in[i] = i * f.n / t; in[t] = f.n;
		#pragma omp parallel for private(i)
		for (i = 0; i < t; i += 2)
		merge(f, f.data + in[i] * f.c, f.data + in[i + 1] * f.c, in[i + 1] - in[i], in[i + 2] - in[i + 1]);
		t /= 2;
	}
}

void pqsort(func f) {

	size_t in[CPUTHREADS + 1];
	register dim i;

	for (i = 0; i < CPUTHREADS; i++) in[i] = i * f.n / CPUTHREADS; in[CPUTHREADS] = f.n;
	#pragma omp parallel for private(i)
	for (i = 0; i < CPUTHREADS; i++) qsort_r(f.data + in[i] * f.c, in[i + 1] - in[i], sizeof(chunk) * f.c, compare, &f);
	if (CPUTHREADS > 1) arraymerge(f);
}

void shared2least(func f, chunk* m) {

	register dim x, y, i;
	register var t;
	chunk* s = (chunk *)calloc(f.c, sizeof(chunk));
	chunk* z = (chunk *)calloc(f.c, sizeof(chunk));
	chunk* a = (chunk *)malloc(sizeof(chunk) * f.c);
	chunk* o = (chunk *)malloc(sizeof(chunk) * f.c);

	for (i = 0; i < f.s / BITSPERCHUNK; i++) s[i] = ~(0ULL);
	if (f.s % BITSPERCHUNK) s[f.s / BITSPERCHUNK] = f.mask;

	for (i = 0; i < f.c; i++) {
		a[i] = s[i] & ~m[i];
		o[i] = m[i] & ~s[i];
	}

	memcpy(m, s, sizeof(chunk) * f.c);

	while (memcmp(o, z, f.c * sizeof(chunk))) {
		i = x = y = 0;
		while (!o[i++]) x += BITSPERCHUNK;
		x += __builtin_ctzll(o[i - 1]);
		i = 0;
		while (!a[i++]) y += BITSPERCHUNK;
		y += __builtin_ctzll(a[i - 1]);
		t = f.vars[x];
		f.vars[x] = f.vars[y];
		f.vars[y] = t;
		#pragma omp parallel for private(i)
		for (i = 0; i < f.n; i++) SWAP_RM(f.data + i * f.c, x, y);
		o[x / BITSPERCHUNK] ^= 1ULL << (x % BITSPERCHUNK);
		a[y / BITSPERCHUNK] ^= 1ULL << (y % BITSPERCHUNK);
	}

	free(s);
	free(z);
	free(a);
	free(o);
}

void shared2least_cm(func f, chunk* m) {

	register dim x, y, i;
	register var t;
	chunk* s = (chunk *)calloc(f.c, sizeof(chunk));
	chunk* z = (chunk *)calloc(f.c, sizeof(chunk));
	chunk* a = (chunk *)malloc(sizeof(chunk) * f.c);
	chunk* o = (chunk *)malloc(sizeof(chunk) * f.c);

	for (i = 0; i < f.s / BITSPERCHUNK; i++) s[i] = ~(0ULL);
	if (f.s % BITSPERCHUNK) s[f.s / BITSPERCHUNK] = f.mask;

	for (i = 0; i < f.c; i++) {
		a[i] = s[i] & ~m[i];
		o[i] = m[i] & ~s[i];
	}

	memcpy(m, s, sizeof(chunk) * f.c);

	while (memcmp(o, z, f.c * sizeof(chunk))) {
		i = x = y = 0;
		while (!o[i++]) x += BITSPERCHUNK;
		x += __builtin_ctzll(o[i - 1]);
		i = 0;
		while (!a[i++]) y += BITSPERCHUNK;
		y += __builtin_ctzll(a[i - 1]);
		t = f.vars[x];
		f.vars[x] = f.vars[y];
		f.vars[y] = t;
		#pragma omp parallel for private(i)
		for (i = 0; i < f.n; i++) SWAP_CM(f.data + i, x, y, f.n);
		o[x / BITSPERCHUNK] ^= 1ULL << (x % BITSPERCHUNK);
		a[y / BITSPERCHUNK] ^= 1ULL << (y % BITSPERCHUNK);
	}

	free(s);
	free(z);
	free(a);
	free(o);
}

void reordershared(func f, var *vars) {

	chunk s[CEIL(f.s, BITSPERCHUNK)];
	register dim i, j;
	var v[MAXVAR];

	for (i = 0; i < f.s; i++) v[vars[i]] = i;
	#pragma omp parallel for private(i, s)
	for (i = 0; i < f.n; i++) {
		memset(s, 0, sizeof(chunk) * CEIL(f.s, BITSPERCHUNK));
		for (j = 0; j < f.s; j++) if GET_RM(f.data + i * f.c, j) SET_RM(s, v[f.vars[j]]);
		memcpy(f.data + i * f.c, s, sizeof(chunk) * (f.s / BITSPERCHUNK));
		if (f.mask) {
			f.data[i * f.c + f.s / BITSPERCHUNK] &= ~f.mask;
			f.data[i * f.c + f.s / BITSPERCHUNK] |= s[f.s / BITSPERCHUNK];
		}
	}
	memcpy(f.vars, vars, sizeof(var) * f.s);
}

void reordershared_cm(func f, var *vars) {

	chunk s[CEIL(f.s, BITSPERCHUNK)];
	register dim i, j;
	var v[MAXVAR];

	for (i = 0; i < f.s; i++) v[vars[i]] = i;
	#pragma omp parallel for private(i, s)
	for (i = 0; i < f.n; i++) {
		memset(s, 0, sizeof(chunk) * CEIL(f.s, BITSPERCHUNK));
		for (j = 0; j < f.s; j++) if GET_CM(f.data + i, j, f.n) SET_RM(s, v[f.vars[j]]);
		for (j = 0; j < f.s / BITSPERCHUNK; j++) f.data[j * f.n + i] = s[j];
		if (f.mask) {
			f.data[(f.s / BITSPERCHUNK) * f.n + i] &= ~f.mask;
			f.data[(f.s / BITSPERCHUNK) * f.n + i] |= s[f.s / BITSPERCHUNK];
		}
	}
	memcpy(f.vars, vars, sizeof(var) * f.s);
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

dim uniquecombinations(func f) {

	register dim i, u = 1;

	for (i = 1; i < f.n; i++)
		if (memcmp(f.data + (i - 1) * f.c, f.data + i * f.c, sizeof(chunk) * (f.s / BITSPERCHUNK)) || \
		(f.mask & (f.data[(i - 1) * f.c + f.s / BITSPERCHUNK] ^ f.data[i * f.c + f.s / BITSPERCHUNK]))) u++;

	return u;
}

dim uniquecombinations_cm(func f) {

	register dim i, j, u = 1;

	for (i = 1; i < f.n; i++) {
		for (j = 0; j < f.s / BITSPERCHUNK; j++) if (f.data[j * f.n + i] != f.data[j * f.n + i - 1]) { u++; goto next; }
		if (f.mask & (f.data[(f.s / BITSPERCHUNK) * f.n + i] ^ f.data[(f.s / BITSPERCHUNK) * f.n + i - 1])) u++;
		next:;
	}

	return u;
}
