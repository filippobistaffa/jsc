#include "jsc.h"

void randomdata(func f) { // assumes BITSPERCHUNK == 64

	register dim i, j;

	for (i = 0; i < f.n; i++) {
		for (j = 0; j < f.c - 1; j++) f.data[i * f.c + j] = genrand64_int64();
		f.data[i * f.c + f.c - 1] = genrand64_int64() & ((1ULL << (f.m % BITSPERCHUNK)) - 1);
	}
}

void randomvars(func f, dim max) {

	register dim i, j;
	register var v;
	srand(SEED);

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
		for (j = 0; j < f.c - 1; j++)
			for (k = 0; k < BITSPERCHUNK; k++)
				printf("%2zu", (f.data[i * f.c + j] >> k) & 1);
		for (k = 0; k < f.m % BITSPERCHUNK; k++)
			printf("%2zu", (f.data[i * f.c + f.c - 1] >> k) & 1);
		printf("\n");
	}
}

void sharedmasks(func f1, chunk* s1, func f2, chunk* s2) {

	register dim i, j;

	for (i = 0; i < f1.m; i++)
		for (j = 0; j < f2.m; j++)
			if (f1.vars[i] == f2.vars[j]) {
				SET(s1, i);
				SET(s2, j);
				(f1.s)++;
				(f2.s)++;
				break;
			}
}

void shared2least(func f, chunk* m) {

	register dim x, y, i, c = CEIL(f.s, BITSPERCHUNK);
	register var t;
	chunk* s = (chunk *)calloc(f.c, sizeof(chunk));
	chunk* z = (chunk *)calloc(f.c, sizeof(chunk));
	chunk* a = (chunk *)malloc(sizeof(chunk) * f.c);
	chunk* o = (chunk *)malloc(sizeof(chunk) * f.c);

	for (i = 0; i < c - 1; i++) s[i] = ~(0ULL);
	s[c - 1] = (1ULL << (f.s % BITSPERCHUNK)) - 1;

	for (i = 0; i < f.c; i++) {
		a[i] = s[i] & ~m[i];
		o[i] = m[i] & ~s[i];
	}

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
		for (i = 0; i < f.n; i++) SWAP(f.data + i * f.c, x, y);
		o[x / BITSPERCHUNK] ^= 1ULL << (x % BITSPERCHUNK);
		a[y / BITSPERCHUNK] ^= 1ULL << (y % BITSPERCHUNK);
	}

	free(s);
	free(z);
	free(a);
	free(o);
}

int main(int argc, char *argv[]) {

	init_genrand64(SEED);

	func f_h, f_d;
	f_h.n = 20;
	f_h.m = 70;
	f_h.c = CEIL(f_h.m, BITSPERCHUNK);
	f_h.vars = (var *)malloc(sizeof(var) * f_h.m);
	f_h.data = (chunk *)calloc(f_h.n * f_h.c, sizeof(chunk));

	randomvars(f_h, 100);
	randomdata(f_h);

	chunk c[2] = {1561564548000000000, 10};
	f_h.s = __builtin_popcountll(c[0]) + __builtin_popcountll(c[1]);

	f_d = f_h;
	cudaMalloc(&(f_d.vars), sizeof(var) * f_h.m);
	cudaMemcpy(f_d.vars, f_h.vars, sizeof(var) * f_h.m, cudaMemcpyHostToDevice);
	cudaMalloc(&(f_d.data), sizeof(chunk) * f_h.n * f_h.c);
	cudaMemcpy(f_d.data, f_h.data, sizeof(chunk) * f_h.n * f_h.c, cudaMemcpyHostToDevice);

	print(f_h, c);
	shared2least(f_h, c);

	cudaMemcpy(f_h.vars, f_d.vars, sizeof(var) * f_h.m, cudaMemcpyDeviceToHost);
	cudaMemcpy(f_h.data, f_d.data, sizeof(chunk) * f_h.n * f_h.c, cudaMemcpyDeviceToHost);

	print(f_h, NULL);

	cudaFree(f_d.vars);
	cudaFree(f_d.data);

	free(f_h.vars);
	free(f_h.data);

	return 0;
}
