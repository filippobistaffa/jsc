#include "jsc.h"

template<dim N, dim M, dim C>
void randomdata(func f) { // assumes BITSPERCHUNK == 64

	register dim i, j;
	for (i = 0; i < N; i++) {
		for (j = 0; j < M / BITSPERCHUNK; j++) f.data[j * N + i] = genrand64_int64();
		if (M % BITSPERCHUNK) f.data[(C - 1) * N + i] = genrand64_int64() & ((1ULL << (M % BITSPERCHUNK)) - 1);
	}
}

template<dim M>
void randomvars(func f, dim max) {

	register dim i, j;
	register var v;
	srand(SEED);

	for (i = 0; i < M; i++) {
		random:
		v = rand() % max;
		for (j = 0; j < i; j++)
			if (f.vars[j] == v)
			goto random;
		f.vars[i] = v;
	}
}

template<dim N, dim M, dim C>
void print(func f, chunk *s) {

	register dim i, j, k;

	for (i = 0; i < M; i++) {
		if (i & 1) printf("\033[1m%2u\033[0m", i);
		else printf("%2u", i);
	}
	printf("\n");

	for (i = 0; i < M; i++) {
		if (i & 1) printf("\033[1m");
		if (s && ((s[i / BITSPERCHUNK] >> (i % BITSPERCHUNK)) & 1)) printf("\x1b[31m%2u\x1b[0m", f.vars[i]);
		else printf("%2u", f.vars[i]);
		if (i & 1) printf("\033[0m");
	}
	printf("\n");

	for (i = 0; i < N; i++) {
		for (j = 0; j < M / BITSPERCHUNK; j++)
			for (k = 0; k < BITSPERCHUNK; k++)
				printf("%2zu", (f.data[j * N + i] >> k) & 1);
		for (k = 0; k < M % BITSPERCHUNK; k++)
			printf("%2zu", (f.data[(C - 1) * N + i] >> k) & 1);
		printf("\n");
	}
}

template<dim MA, dim MB>
void sharedmasks(func f1, chunk* s1, func f2, chunk* s2) {

	register dim i, j;

	for (i = 0; i < MA; i++)
		for (j = 0; j < MB; j++)
			if (f1.vars[i] == f2.vars[j]) {
				SET(s1, i);
				SET(s2, j);
				(f1.s)++;
				(f2.s)++;
				break;
			}
}

template<dim N, dim M, dim C>
__global__ void shared2least(func f) {

	dim tid, tx, bx = blockIdx.x;
	tid = bx * THREADSPERBLOCK + (tx = threadIdx.x);

	if (tid < N) {

		__shared__ chunk s[C], a[C], o[C], m[C], sh[C * THREADSPERBLOCK];
		__shared__ var v[M];
		dim x, y, i;
		var t;

		if (!tx) printf("Block #%u\n", bx);

		if (!bx) for (i = tx; i < M; i += THREADSPERBLOCK) {
			v[i] = f.vars[i];
		//printf("v[%u] = %u\n", i, v[i]);
		}
		if (tx < C) {
			s[tx] = 0;
			m[tx] = f.mask[tx];
		}

		//if (!bx && !tx) for (i = 0; i < M; i++) printf("v[%u] = %u\n", i, v[i]);

		__syncthreads();

		#ifdef UNROLL
                #pragma unroll UNROLL
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
		#pragma unroll UNROLL
		#endif
        	for (i = 0; i < C; i++) f.data[N * i + tid] = sh[THREADSPERBLOCK * i + tx];

		if (!bx) for (i = tx; i < M; i += THREADSPERBLOCK) f.vars[i] = v[i];
	}
}

int main(int argc, char *argv[]) {

	init_genrand64(SEED);

	func f_h, f_d;
	f_h.vars = (var *)malloc(sizeof(var) * M1);
	f_h.data = (chunk *)calloc(N1 * C1, sizeof(chunk));

	randomvars<M1>(f_h, 100);
	randomdata<N1, M1, C1>(f_h);

	chunk c_h[2] = {1561500000000000000, 28};
	f_h.s = __builtin_popcountll(c_h[0]) +  + __builtin_popcountll(c_h[1]);

	f_d = f_h;
	cudaMalloc(&(f_d.vars), sizeof(var) * M1);
	cudaMemcpy(f_d.vars, f_h.vars, sizeof(var) * M1, cudaMemcpyHostToDevice);
	cudaMalloc(&(f_d.mask), sizeof(chunk) * C1);
	cudaMemcpy(f_d.mask, c_h, sizeof(chunk) * C1, cudaMemcpyHostToDevice);
	cudaMalloc(&(f_d.data), sizeof(chunk) * N1 * C1);
	cudaMemcpy(f_d.data, f_h.data, sizeof(chunk) * N1 * C1, cudaMemcpyHostToDevice);

	print<N1, M1, C1>(f_h, c_h);
	shared2least<N1, M1, C1><<<CEIL(N1, THREADSPERBLOCK), THREADSPERBLOCK>>>(f_d);

	cudaMemcpy(f_h.vars, f_d.vars, sizeof(var) * M1, cudaMemcpyDeviceToHost);
	cudaMemcpy(f_h.data, f_d.data, sizeof(chunk) * N1 * C1, cudaMemcpyDeviceToHost);

	print<N1, M1, C1>(f_h, NULL);

	cudaFree(f_d.vars);
	cudaFree(f_d.data);

	free(f_h.vars);
	free(f_h.data);

	return 0;
}

