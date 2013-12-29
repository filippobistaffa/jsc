#include "jsc.h"

int main(int argc, char *argv[]) {

	srand(SEED);
	init_genrand64(SEED);
	func f1, f2;

	f1.n = 5e7;
	f1.m = 80;
	f2.n = 30;
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

	randomvars(f1);
	randomdata(f1);
	randomvars(f2);
	randomdata(f2);

	chunk *c1 = calloc(f1.c, sizeof(chunk));
	chunk *c2 = calloc(f2.c, sizeof(chunk));
	sharedmasks(&f1, c1, &f2, c2);
	f1.mask = f2.mask = (1ULL << (f1.s % BITSPERCHUNK)) - 1;
	printf("%u shared variables\n", f1.s);

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
	//pqsort(f1);
	//pqsort(f2);
	//print(f1, c1);
	//pqsort(f1);
	//qsort_r(f1.data, f1.n, sizeof(chunk) * f1.c, compare, &f1);
	qsort_cm(f1);
	//print(f1, c1);
	//print(f2, c2);
	//printf("%u unique combinations\n", uniquecombinations_cm(f1));
	//printf("%u unique combinations\n", uniquecombinations_cm(f2));
	//transpose(f1.data, f1.n, f1.c);
	//transpose(f2.data, f2.c, f2.n);
	puts("Checksum...");
	printf("Checksum 1 = %u (size = %zu bytes)\n", crc32(f1.data, sizeof(chunk) * f1.n * f1.c), sizeof(chunk) * f1.n * f1.c);
	//printf("Checksum 2 = %u (size = %zu bytes)\n", crc32(f2.data, sizeof(chunk) * f2.n * f2.c), sizeof(chunk) * f2.n * f2.c);

	free(f1.vars);
	free(f1.data);
	free(f2.vars);
	free(f2.data);
	
	return 0;
}
