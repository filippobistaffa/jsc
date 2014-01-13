#include "jsc.h"
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

using namespace thrust;

__global__ void histogramproduct(dim *h1, dim *h2, dim *hr, dim hn) {

	dim tid = blockIdx.x * THREADSPERBLOCK + threadIdx.x;
	if (tid < hn) hr[tid] = h1[tid] * h2[tid];
}

int main(int argc, char *argv[]) {

	func f1, f2;
	struct timeval t1, t2;
	init_genrand64(SEED);
	srand(SEED);

	f1.n = 1e8;
	f1.m = 80;
	f2.n = 1e7;
	f2.m = 50;

	f1.c = CEIL(f1.m, BITSPERCHUNK);
	f2.c = CEIL(f2.m, BITSPERCHUNK);
	f1.vars = (var *)malloc(sizeof(var) * f1.m);
	f2.vars = (var *)malloc(sizeof(var) * f2.m);
	f1.data = (chunk *)calloc(1, sizeof(chunk) * f1.n * f1.c);
	f2.data = (chunk *)calloc(1, sizeof(chunk) * f2.n * f2.c);

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

	chunk *c1 = (chunk *)calloc(f1.c, sizeof(chunk));
	chunk *c2 = (chunk *)calloc(f2.c, sizeof(chunk));
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
	free(c1);
	free(c2);

	printf("Sort... ");
	fflush(stdout);
	gettimeofday(&t1, NULL);
	sort(f1);
	sort(f2);
	gettimeofday(&t2, NULL);
	printf("%f seconds\n", (double)(t2.tv_usec - t1.tv_usec) / 1e6 + t2.tv_sec - t1.tv_sec);

	printf("%u unique combinations\n", f1.hn = uniquecombinations(f1));
	printf("%u unique combinations\n", f2.hn = uniquecombinations(f2));
	f1.h = (dim *)calloc(f1.hn, sizeof(dim));
	f2.h = (dim *)calloc(f2.hn, sizeof(dim));

	printf("Histogram... ");
	fflush(stdout);
	gettimeofday(&t1, NULL);
	histogram(f1);
	histogram(f2);
	gettimeofday(&t2, NULL);
	printf("%f seconds\n", (double)(t2.tv_usec - t1.tv_usec) / 1e6 + t2.tv_sec - t1.tv_sec);

	printf("Matching Rows... ");
	fflush(stdout);
	gettimeofday(&t1, NULL);
	f1.hmask = (chunk *)calloc(CEIL(f1.hn, BITSPERCHUNK), sizeof(chunk));
	f2.hmask = (chunk *)calloc(CEIL(f2.hn, BITSPERCHUNK), sizeof(chunk));
	dim n1, n2, hn;
	markmatchingrows(f1, f2, &n1, &n2, &hn);
	copymatchingrows(&f1, &f2, n1, n2, hn);
	gettimeofday(&t2, NULL);
	printf("%f seconds\n", (double)(t2.tv_usec - t1.tv_usec) / 1e6 + t2.tv_sec - t1.tv_sec);

	printf("%u matching rows\n", f1.n);
	printf("%u matching rows\n", f2.n);

	dim *h1d, *h2d, *hpd;
	cudaMalloc(&(h1d), sizeof(dim) * hn);
	cudaMalloc(&(h2d), sizeof(dim) * hn);
	cudaMalloc(&(hpd), sizeof(dim) * hn);
	cudaMemcpy(h1d, f1.h, sizeof(dim) * hn, cudaMemcpyHostToDevice);
	cudaMemcpy(h2d, f2.h, sizeof(dim) * hn, cudaMemcpyHostToDevice);

	histogramproduct<<<CEIL(hn, THREADSPERBLOCK), THREADSPERBLOCK>>>(h1d, h2d, hpd, hn);

	device_ptr<dim> h1t(h1d);
	device_ptr<dim> h2t(h2d);
	device_ptr<dim> hpt(hpd);
	device_vector<dim> pfxh1t(hn);
	device_vector<dim> pfxh2t(hn);
	device_vector<dim> pfxhpt(hn);

	exclusive_scan(h1t, h1t + hn, pfxh1t.begin());
	exclusive_scan(h2t, h2t + hn, pfxh2t.begin());
	exclusive_scan(hpt, hpt + hn, pfxhpt.begin());
	dim no = reduce(hpt, hpt + hn);

	printf("Result size = %zu bytes\n", sizeof(chunk) * no * CEIL(f1.m + f2.m - f1.s, BITSPERCHUNK));

	puts("Checksum...");
	printf("Checksum 1 = %u (size = %zu bytes)\n", crc32(f1.data, sizeof(chunk) * f1.n * f1.c), sizeof(chunk) * f1.n * f1.c);
	printf("Checksum Histogram 1 = %u (size = %zu bytes)\n", crc32(f1.h, sizeof(dim) * f1.hn), sizeof(dim) * f1.hn);
	printf("Checksum 2 = %u (size = %zu bytes)\n", crc32(f2.data, sizeof(chunk) * f2.n * f2.c), sizeof(chunk) * f2.n * f2.c);
	printf("Checksum Histogram 2 = %u (size = %zu bytes)\n", crc32(f2.h, sizeof(dim) * f2.hn), sizeof(dim) * f2.hn);

	cudaFree(h1d);
	cudaFree(h2d);
	cudaFree(hpd);

	free(f1.hmask);
	free(f2.hmask);
	free(f1.vars);
	free(f1.data);
	free(f2.vars);
	free(f2.data);
	free(f1.h);
	free(f2.h);

	return 0;
}
