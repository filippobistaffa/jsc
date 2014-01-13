#include "jsc.h"

template<dim N, dim M, dim C>
__global__ void shared2least(func f) {

	dim tid, tx, bx = blockIdx.x;
	tid = bx * THREADSPERBLOCK + (tx = threadIdx.x);

	if (tid < N) {

		__shared__ chunk s[C], a[C], o[C], m[C], sh[C * THREADSPERBLOCK];
		__shared__ var v[M];
		dim x, y, i;
		var t;

		if (!bx) for (i = tx; i < M; i += THREADSPERBLOCK) v[i] = f.vars[i];
		if (tx < C) {
			s[tx] = 0;
			//m[tx] = f.mask[tx];
		}

		//if (!bx && !tx) for (i = 0; i < M; i++) printf("v[%u] = %u\n", i, v[i]);

		__syncthreads();

		#ifdef UNROLL
                #pragma unroll 2
                #endif
		for (i = 0; i < C; i++) sh[THREADSPERBLOCK * i + tx] = f.data[N * i + tid];

		if (tx < f.s / BITSPERCHUNK) s[tx] = ~(0ULL);
		if ((tx == 0) && (f.s % BITSPERCHUNK)) s[f.s / BITSPERCHUNK] = (1ULL << (f.s % BITSPERCHUNK)) - 1;
		if (tx < C) {
			a[tx] = s[tx] & ~m[tx];
			o[tx] = m[tx] & ~s[tx];
		}

		__syncthreads();

	        while (__any(tx < C && o[tx])) {
        	        i = x = y = 0;
                	while (!o[i++]) x += BITSPERCHUNK;
    	        	x += __ffsll(o[i - 1]) - 1;
       	        	i = 0;
                	while (!a[i++]) y += BITSPERCHUNK;
                	y += __ffsll(a[i - 1]) - 1;
			SWAP(sh + tx, x, y, THREADSPERBLOCK);
			if (!tx) {
				if (!bx) {
					//printf("switched v[%u] = %u with v[%u] = %u\n", x, v[x], y, v[y]);
					t = v[x];
	                		v[x] = v[y];
        	        		v[y] = t;
				}
	               		o[x / BITSPERCHUNK] ^= 1ULL << (x % BITSPERCHUNK);
	                	a[y / BITSPERCHUNK] ^= 1ULL << (y % BITSPERCHUNK);
			}
			__syncthreads();
        	}

		#ifdef UNROLL
		#pragma unroll 2
		#endif
        	for (i = 0; i < C; i++) f.data[N * i + tid] = sh[THREADSPERBLOCK * i + tx];

		if (!bx) for (i = tx; i < M; i += THREADSPERBLOCK) f.vars[i] = v[i];
	}
}

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

	dim *h1d, *h2d;
	cudaMalloc(&(h1d), sizeof(dim) * hn);
	cudaMalloc(&(h2d), sizeof(dim) * hn);

	cudaMemcpy(h1d, f1.h, sizeof(dim) * hn, cudaMemcpyHostToDevice);
	cudaMemcpy(h2d, f2.h, sizeof(dim) * hn, cudaMemcpyHostToDevice);

	puts("Checksum...");
	printf("Checksum 1 = %u (size = %zu bytes)\n", crc32(f1.data, sizeof(chunk) * f1.n * f1.c), sizeof(chunk) * f1.n * f1.c);
	printf("Checksum Histogram 1 = %u (size = %zu bytes)\n", crc32(f1.h, sizeof(dim) * f1.hn), sizeof(dim) * f1.hn);
	printf("Checksum 2 = %u (size = %zu bytes)\n", crc32(f2.data, sizeof(chunk) * f2.n * f2.c), sizeof(chunk) * f2.n * f2.c);
	printf("Checksum Histogram 2 = %u (size = %zu bytes)\n", crc32(f2.h, sizeof(dim) * f2.hn), sizeof(dim) * f2.hn);

	cudaFree(h1d);
	cudaFree(h2d);

	free(f1.vars);
	free(f1.data);
	free(f2.vars);
	free(f2.data);

	return 0;
}

