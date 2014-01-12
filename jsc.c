#include "jsc.h"

int main(int argc, char *argv[]) {

	func f1, f2;
	struct timeval t1, t2;
	init_genrand64(SEED);
	srand(SEED);

	f1.n = 1e8;
	f1.m = 80;
	f2.n = 1e8;
	f2.m = 50;

	f1.c = CEIL(f1.m, BITSPERCHUNK);
	f2.c = CEIL(f2.m, BITSPERCHUNK);
	f1.vars = malloc(sizeof(var) * f1.m);
	f2.vars = malloc(sizeof(var) * f2.m);
	f1.data = calloc(1, sizeof(chunk) * f1.n * f1.c);
	f2.data = calloc(1, sizeof(chunk) * f2.n * f2.c);

	if (!f1.data || !f2.data) {
		printf("Not enough memory!\n");
		return 1;
	}

	printf("Random data... ");
	fflush(stdout);
	gettimeofday(&t1, NULL);
	randomvars(f1);
	randomdata(f1);
	randomvars(f2);
	randomdata(f2);
	gettimeofday(&t2, NULL);
	printf("%f seconds\n", (double)(t2.tv_usec - t1.tv_usec) / 1e6 + t2.tv_sec - t1.tv_sec);

	chunk *c1 = calloc(f1.c, sizeof(chunk));
	chunk *c2 = calloc(f2.c, sizeof(chunk));
	sharedmasks(&f1, c1, &f2, c2);
	f1.mask = f2.mask = (1ULL << (f1.s % BITSPERCHUNK)) - 1;
	printf("%u shared variables\n", f1.s);

	printf("Shift & Reorder... ");
	fflush(stdout);
	gettimeofday(&t1, NULL);
	shared2least(f1, c1);
	shared2least(f2, c2);
	reordershared(f2, f1.vars);
	gettimeofday(&t2, NULL);
	printf("%f seconds\n", (double)(t2.tv_usec - t1.tv_usec) / 1e6 + t2.tv_sec - t1.tv_sec);

	printf("Sort... ");
	fflush(stdout);
	gettimeofday(&t1, NULL);
	//qsort_r(f1.data, f1.n, sizeof(chunk) * f1.c, compare, &f1);
	//qsort_r(f2.data, f2.n, sizeof(chunk) * f2.c, compare, &f2);
	sort(f1);
	sort(f2);
	gettimeofday(&t2, NULL);
	printf("%f seconds\n", (double)(t2.tv_usec - t1.tv_usec) / 1e6 + t2.tv_sec - t1.tv_sec);

	printf("%u unique combinations\n", f1.hn = uniquecombinations(f1));
	printf("%u unique combinations\n", f2.hn = uniquecombinations(f2));
	f1.h = calloc(f1.hn, sizeof(dim));
	f2.h = calloc(f2.hn, sizeof(dim));

	printf("Histogram... ");
	fflush(stdout);
	gettimeofday(&t1, NULL);
	histogram(f1);
	histogram(f2);
	gettimeofday(&t2, NULL);
	printf("%f seconds\n", (double)(t2.tv_usec - t1.tv_usec) / 1e6 + t2.tv_sec - t1.tv_sec);
	//print(f1, c1);
	//print(f2, c2);

	puts("Checksum...");
	printf("Checksum 1 = %u (size = %zu bytes)\n", crc32(f1.data, sizeof(chunk) * f1.n * f1.c), sizeof(chunk) * f1.n * f1.c);
	printf("Checksum Histogram 1 = %u (size = %zu bytes)\n", crc32(f1.h, sizeof(dim) * f1.hn), sizeof(dim) * f1.hn);
	printf("Checksum 2 = %u (size = %zu bytes)\n", crc32(f2.data, sizeof(chunk) * f2.n * f2.c), sizeof(chunk) * f2.n * f2.c);
	printf("Checksum Histogram 2 = %u (size = %zu bytes)\n", crc32(f2.h, sizeof(dim) * f2.hn), sizeof(dim) * f2.hn);

	f1.rmask = calloc(CEIL(f1.n, BITSPERCHUNK), sizeof(chunk));
	f1.hmask = calloc(CEIL(f1.hn, BITSPERCHUNK), sizeof(chunk));
	f2.rmask = calloc(CEIL(f2.n, BITSPERCHUNK), sizeof(chunk));
	f2.hmask = calloc(CEIL(f2.hn, BITSPERCHUNK), sizeof(chunk));
	sharedrows(f1, f2);
	//transpose(f1.data, f1.n, f1.c);
	//transpose(f2.data, f2.n, f2.c);

	puts("Checksum...");
	printf("Checksum 1 = %u (size = %zu bytes)\n", crc32(f1.data, sizeof(chunk) * f1.n * f1.c), sizeof(chunk) * f1.n * f1.c);
	printf("Checksum Histogram 1 = %u (size = %zu bytes)\n", crc32(f1.h, sizeof(dim) * f1.hn), sizeof(dim) * f1.hn);
	printf("Checksum 2 = %u (size = %zu bytes)\n", crc32(f2.data, sizeof(chunk) * f2.n * f2.c), sizeof(chunk) * f2.n * f2.c);
	printf("Checksum Histogram 2 = %u (size = %zu bytes)\n", crc32(f2.h, sizeof(dim) * f2.hn), sizeof(dim) * f2.hn);

	free(f1.vars);
	free(f1.data);
	free(f2.vars);
	free(f2.data);

	return 0;
}
