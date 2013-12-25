#include "jsc.h"

void randomdata(func f) { // assumes BITSPERCHUNK == 64

	register dim i, j;

	for (i = 0; i < f.n; i++) {
		for (j = 0; j < f.m / BITSPERCHUNK; j++) f.data[i * f.c + j] = genrand64_int64();
		if (f.m % BITSPERCHUNK) f.data[i * f.c + f.c - 1] = genrand64_int64() & ((1ULL << (f.m % BITSPERCHUNK)) - 1);
	}
}

void randomvars(func f, dim max) {

	register dim i, j;
	register var v;

	for (i = 0; i < f.m; i++) {
		random:
		v = rand() % max;
		for (j = 0; j < i; j++)
			if (f.vars[j] == v)
			goto random;
		f.vars[i] = v;
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
inline int compare(const void* a, const void* b, void* c)
{
	register func f = *(func *)c;
	register uint8_t cmp = memcmp(a, b, sizeof(chunk) * (f.s / BITSPERCHUNK));

	if (cmp || !f.mask) return cmp;
	else {
		register chunk x = *(chunk *)a & f.mask;
		register chunk y = *(chunk *)b & f.mask;
		if (x == y) return 0;
		else if (x < y) return -1;
		else return 1;
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
		//printf("switched %u with %u\n", x, y);
		t = f.vars[x];
		f.vars[x] = f.vars[y];
		f.vars[y] = t;
		#pragma omp parallel for private(i)
		for (i = 0; i < f.n; i++) SWAPRM(f.data + i * f.c, x, y);
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
		for (j = 0; j < f.s; j++) if ((f.data[i * f.c + j / BITSPERCHUNK] >> (j % BITSPERCHUNK)) & 1) SET(s, v[f.vars[j]]);
		memcpy(f.data + i * f.c, s, sizeof(chunk) * (f.s / BITSPERCHUNK));
		if (f.mask) {
			f.data[i * f.c + f.s / BITSPERCHUNK] &= ~f.mask;
			f.data[i * f.c + f.s / BITSPERCHUNK] |= s[f.s / BITSPERCHUNK];
		}
	}
	memcpy(f.vars, vars, sizeof(var) * f.s);
}

int main(int argc, char *argv[]) {

	srand(SEED);
	init_genrand64(SEED);
	func f1, f2;

	f1.n = 5e7;
	f1.m = 70;
	f2.n = 1e8;
	f2.m = 80;

	f1.c = CEIL(f1.m, BITSPERCHUNK);
	f2.c = CEIL(f2.m, BITSPERCHUNK);
	f1.vars = malloc(sizeof(var) * f1.m);
	f2.vars = malloc(sizeof(var) * f2.m);
	f1.data = calloc(1, sizeof(chunk) * f1.n * f1.c);
	f2.data = calloc(1, sizeof(chunk) * f2.n * f2.c);

	if (!f1.data || !f1.data) {
		printf("Not enough memory!\n");
		return 1;
	}

	randomvars(f1, MAXVAR);
	randomdata(f1);
	randomvars(f2, MAXVAR);
	randomdata(f2);

	chunk *c1 = calloc(f1.c, sizeof(chunk));
	chunk *c2 = calloc(f2.c, sizeof(chunk));
	sharedmasks(&f1, c1, &f2, c2);
	f1.mask = f2.mask = (1ULL << (f1.s % BITSPERCHUNK)) - 1;

	//print(f1, c1);
	//print(f2, c2);
	puts("Shift...");
	shared2least(f1, c1);
	shared2least(f2, c2);
	reordershared(f2, f1.vars);
	//print(f1, c1);
	//print(f2, c2);
	puts("Sort...");
	//qsort_r(f1.data, f1.n, sizeof(chunk) * f1.c, compare, &f1);
	//qsort_r(f2.data, f2.n, sizeof(chunk) * f2.c, compare, &f2);
	pqsort(f1);
	pqsort(f2);
	//print(f1, c1);
	//print(f2, c2);
	puts("Checksum...");
	printf("Checksum 1 = %u (size = %zu bytes)\n", crc32(f1.data, sizeof(chunk) * f1.n * f1.c), sizeof(chunk) * f1.n * f1.c);
	printf("Checksum 2 = %u (size = %zu bytes)\n", crc32(f2.data, sizeof(chunk) * f2.n * f2.c), sizeof(chunk) * f2.n * f2.c);

	free(f1.vars);
	free(f1.data);
	free(f2.vars);
	free(f2.data);
	return 0;
}
