#include "jsc.h"

void randomdata(func f) { // assumes BITSPERCHUNK == 64

	register dim i, j;

	for (i = 0; i < f.n; i++) {
		for (j = 0; j < f.m / BITSPERCHUNK; j++) f.data[i * f.c + j] = genrand64_int64();
		if (f.m % BITSPERCHUNK) f.data[i * f.c + f.c - 1] = genrand64_int64() & ((1ULL << (f.m % BITSPERCHUNK)) - 1);
	}
}

void print(func f, chunk *s) {

	register dim i, j, k;

	for (i = 0; i < f.m; i++) {
		if (i & 1) printf("\033[1m%2u\033[0m", i);
		else printf("%2u", i);
	}
	printf("\n");

	for (i = 0; i < f.m; i++) {
		if (i & 1) printf("\033[1m");
		if (s && ((s[i / BITSPERCHUNK] >> (i % BITSPERCHUNK)) & 1)) printf("\x1b[31m%2u\x1b[0m", f.vars[i]);
		else printf("%2u", f.vars[i]);
		if (i & 1) printf("\033[0m");
	}
	printf("\n");

	for (i = 0; i < f.n; i++) {
		for (j = 0; j < f.m / BITSPERCHUNK; j++)
			for (k = 0; k < BITSPERCHUNK; k++)
				printf("%2zu", (f.data[i * f.c + j] >> k) & 1);
		for (k = 0; k < f.m % BITSPERCHUNK; k++)
			printf("%2zu", (f.data[i * f.c + f.c - 1] >> k) & 1);
		printf("\n");
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
		for (i = 0; i < f.n; i++) SWAP(f.data + i * f.c, x, y);
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
		for (j = 0; j < f.s; j++) if GET(f.data + i * f.c, j) SET(s, v[f.vars[j]]);
		memcpy(f.data + i * f.c, s, sizeof(chunk) * (f.s / BITSPERCHUNK));
		if (f.mask) {
			f.data[i * f.c + f.s / BITSPERCHUNK] &= ~f.mask;
			f.data[i * f.c + f.s / BITSPERCHUNK] |= s[f.s / BITSPERCHUNK];
		}
	}
	memcpy(f.vars, vars, sizeof(var) * f.s);
}



dim uniquecombinations(func f) {

	register dim i, u = 1;

	for (i = 1; i < f.n; i++)
		if (memcmp(f.data + (i - 1) * f.c, f.data + i * f.c, sizeof(chunk) * (f.s / BITSPERCHUNK)) || \
		(f.mask & (f.data[(i - 1) * f.c + f.s / BITSPERCHUNK] ^ f.data[i * f.c + f.s / BITSPERCHUNK]))) u++;

	return u;
}


