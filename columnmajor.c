#include "jsc.h"

void randomdata(func f) { // assumes BITSPERCHUNK == 64

	register dim i, j;

	for (i = 0; i < f.n; i++) {
		for (j = 0; j < f.m / BITSPERCHUNK; j++) f.data[j * f.n + i] = genrand64_int64();
		if (f.m % BITSPERCHUNK) f.data[(f.m / BITSPERCHUNK) * f.n + i] = genrand64_int64() & ((1ULL << (f.m % BITSPERCHUNK)) - 1);
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
				printf("%2zu", (f.data[j * f.n + i] >> k) & 1);
		for (k = 0; k < f.m % BITSPERCHUNK; k++)
			printf("%2zu", (f.data[(f.m / BITSPERCHUNK) * f.n + i] >> k) & 1);
		printf("\n");
	}
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
		for (i = 0; i < f.n; i++) SWAP(f.data + i, x, y, f.n);
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
		for (j = 0; j < f.s; j++) if GET(f.data + i, j, f.n) SET(s, v[f.vars[j]]);
		for (j = 0; j < f.s / BITSPERCHUNK; j++) f.data[j * f.n + i] = s[j];
		if (f.mask) {
			f.data[(f.s / BITSPERCHUNK) * f.n + i] &= ~f.mask;
			f.data[(f.s / BITSPERCHUNK) * f.n + i] |= s[f.s / BITSPERCHUNK];
		}
	}
	memcpy(f.vars, vars, sizeof(var) * f.s);
}

dim uniquecombinations(func f) {

	register dim i, j, u = 1;

	for (i = 1; i < f.n; i++) {
		for (j = 0; j < f.s / BITSPERCHUNK; j++) if (f.data[j * f.n + i] != f.data[j * f.n + i - 1]) { u++; goto next; }
		if (f.mask & (f.data[(f.s / BITSPERCHUNK) * f.n + i] ^ f.data[(f.s / BITSPERCHUNK) * f.n + i - 1])) u++;
		next:;
	}

	return u;
}
