#include "jsc.h"

void randomdata(func f) { // assumes BITSPERCHUNK == 64

	register dim i, j;

	for (i = 0; i < f.n; i++) {
		for (j = 0; j < f.m / BITSPERCHUNK; j++) f.data[i * f.c + j] = genrand64_int64();
		if (f.m % BITSPERCHUNK) f.data[i * f.c + f.c - 1] = genrand64_int64() & ((1ULL << (f.m % BITSPERCHUNK)) - 1);
	}
}

void randomdata_cm(func f) { // assumes BITSPERCHUNK == 64

	register dim i, j;

	for (i = 0; i < f.n; i++) {
		for (j = 0; j < f.m / BITSPERCHUNK; j++) f.data[j * f.n + i] = genrand64_int64();
		if (f.m % BITSPERCHUNK) f.data[(f.m / BITSPERCHUNK) * f.n + i] = genrand64_int64() & ((1ULL << (f.m % BITSPERCHUNK)) - 1);
	}
}

void randomvars(func f) {

	assert(MAXVAR > f.m);
	register dim i, j;
	register var v;

	for (i = 0; i < f.m; i++) {
		random:
		v = rand() % MAXVAR;
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

int main(int argc, char *argv[]) {

	srand(SEED);
	init_genrand64(SEED);
	func f1, f2;

	/*f1.n = 5e7;
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
	free(f2.data);*/

	dim i;
	/*chunk test[40] = {1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5};
	move(test, 5, 5, 3);
	
	for (i = 0; i < 8 * 5; i++) printf("%zu ", test[i]);
	puts("");*/
	
	f1.n = 5e7;
	f1.c = 2;
	f1.data = calloc(1, sizeof(chunk) * f1.n * f1.c);
	
	chunk row[f1.c];
	
	for (i = 0; i < f1.c; i++) row[i] = i;
	for (i = 0; i < f1.n; i++) memcpy(f1.data + i * f1.c, row, sizeof(chunk) * f1.c);
	
	//for (i = 0; i < f1.n * f1.c; i++) printf("%zu ", f1.data[i]);
	//puts("");
	
	rowtocolumnmajor(f1);
	
	//for (i = 0; i < f1.n * f1.c; i++) printf("%zu ", f1.data[i]);
	//puts("");
	
	printf("Checksum 1 = %u (size = %zu bytes)\n", crc32(f1.data, sizeof(chunk) * f1.n * f1.c), sizeof(chunk) * f1.n * f1.c);
	
	free(f1.data);
	
	return 0;
}
