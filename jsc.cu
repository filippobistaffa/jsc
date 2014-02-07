#include "jsc.h"

__constant__ uint4 bd[CONSTANTSIZE / sizeof(uint4)];

__global__ void cudahistogramproduct(dim *h1, dim *h2, dim *hr, dim hn) {

	dim tid = blockIdx.x * THREADSPERBLOCK + threadIdx.x;
	if (tid < hn) hr[tid] = h1[tid] * h2[tid];
}

__global__ void computeoutput(func f1, func f2, chunk *data1, chunk *data2, dim *pfxh1, dim *pfxh2, dim *pfxhp) {

	dim bx = blockIdx.x, tx = threadIdx.x;

	__shared__ uint4 i;
	if (!tx) i = bd[bx];
	__syncthreads();

	dim m = i.y ? 1 : i.z;
	__shared__ dim shpfx[SHAREDSIZE / sizeof(dim)];
	chunk *shdata = ((chunk *)shpfx) + CEIL(3 * (m + 1) * sizeof(dim), sizeof(chunk));
	// assume THREADSPERBLOCK > m + 1
	if (tx < m + 1 && (tx || i.x)) {
		shpfx[tx] = pfxh1[i.x + tx - 1];
		shpfx[tx + m + 1] = pfxh2[i.x + tx - 1];
		shpfx[tx + 2 * m + 2] = pfxhp[i.x + tx - 1];
	}
	if (!i.x) shpfx[0] = shpfx[m + 1] = shpfx[2 * m + 2] = 0;
	__syncthreads();

	uint2 j, k;
	if (i.y) {
		j = make_uint2((shpfx[1] - shpfx[0]) / i.y, (shpfx[m + 2] - shpfx[m + 1]) / i.y);
		k = make_uint2(j.x * i.z, j.y * i.w);
	}
	else {
		j = make_uint2(shpfx[i.z] - shpfx[0], shpfx[i.z + m + 1] - shpfx[m + 1]);
		k = make_uint2(0, 0);
	}

	if (i.y == i.z + 1) j.x += (shpfx[1] - shpfx[0]) % i.y;
	if (i.y == i.w + 1) j.y += (shpfx[m + 2] - shpfx[m + 1]) % i.y;

	//if (tx < j.x) printf("bx=%02u tx=%02u block=%02u [1] %02u -> %02u\n", bx, tx, i.x, shpfx[0] + k.x + tx, tx, data1[shpfx[0] + k.x + tx]);
	//if (tx < j.y) printf("bx=%02u tx=%02u block=%02u [2] %02u -> %02u\n", bx, tx, i.x, shpfx[m + 1] + k.y + tx, tx);

	dim h;
	if (tx < j.x) for (h = 0; h < f1.c; h++) shdata[h * j.x + tx] = data1[h * f1.n + shpfx[0] + k.x + tx];
	if (tx < j.y) for (h = 0; h < f2.c; h++) shdata[j.x + h * j.y + tx] = data2[h * f2.n + shpfx[m + 1] + k.y + tx];
	__syncthreads();
}

dim linearbinpacking(func f1, func f2, dim *hp, uint4 *o) {

	register size_t m, mb = MEMORY(0);
	register dim a, b, c, i, j = 0, k = 0;

	for (i = 1; i <= f1.hn; i++)
		if ((m = MEMORY(i)) + mb > SHAREDSIZE || i == f1.hn) {
			a = c = CEIL(mb, SHAREDSIZE);
			do {
				b = c;
				do o[j++] = make_uint4(k, c > 1 ? c : 0, c > 1 ? c - a : i - k, c > 1 ? c - b : 0);
				while (--b);
			} while (--a);
			mb = m;
			k = i;
		}
		else mb += m;

	return j;
}

int main(int argc, char *argv[]) {

	func f1, f2;
	struct timeval t1, t2;
	init_genrand64(SEED);
	srand(SEED);

	f1.n = 100;
	f1.m = 80;
	f2.n = 30;
	f2.m = 100;

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
	if (!f1.s) return 1;

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

	chunk *d1d, *d2d;
	dim on, *h1d, *h2d, *hpd, *pfxh1d, *pfxh2d, *pfxhpd;
	printf("Allocating... ");
	fflush(stdout);
	gettimeofday(&t1, NULL);
	cudaMalloc(&d1d, sizeof(chunk) * f1.n * f1.c);
	cudaMalloc(&d2d, sizeof(chunk) * f2.n * f2.c);
	cudaMalloc(&h1d, sizeof(dim) * hn);
	cudaMalloc(&h2d, sizeof(dim) * hn);
	cudaMalloc(&hpd, sizeof(dim) * hn);
        cudaMalloc(&pfxh1d, sizeof(dim) * hn);
        cudaMalloc(&pfxh2d, sizeof(dim) * hn);
        cudaMalloc(&pfxhpd, sizeof(dim) * hn);
	gettimeofday(&t2, NULL);
        printf("%f seconds\n", (double)(t2.tv_usec - t1.tv_usec) / 1e6 + t2.tv_sec - t1.tv_sec);

	cudaMemcpy(d1d, f1.data, sizeof(chunk) * f1.n * f1.c, cudaMemcpyHostToDevice);
	cudaMemcpy(d2d, f2.data, sizeof(chunk) * f2.n * f2.c, cudaMemcpyHostToDevice);
	cudaMemcpy(h1d, f1.h, sizeof(dim) * hn, cudaMemcpyHostToDevice);
	cudaMemcpy(h2d, f2.h, sizeof(dim) * hn, cudaMemcpyHostToDevice);

	cudahistogramproduct<<<CEIL(hn, THREADSPERBLOCK), THREADSPERBLOCK>>>(h1d, h2d, hpd, hn);

	CUDPPHandle cudpp, pfxsum = 0;
	cudppCreate(&cudpp);
	CUDPPConfiguration config;
	config.op = CUDPP_ADD;
	config.datatype = CUDPP_UINT;
	config.algorithm = CUDPP_SCAN;
	config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
	cudppPlan(cudpp, &pfxsum, config, hn, 1, 0);
	cudppScan(pfxsum, pfxh1d, h1d, hn);
	cudppScan(pfxsum, pfxh2d, h2d, hn);
	cudppScan(pfxsum, pfxhpd, hpd, hn);
	cudppDestroyPlan(pfxsum);
	cudppDestroy(cudpp);

	cudaMemcpy(&on, pfxhpd + hn - 1, sizeof(dim), cudaMemcpyDeviceToHost);
	printf("Result size = %zu bytes (%u lines)\n", sizeof(chunk) * on * CEIL(f1.m + f2.m - f1.s, BITSPERCHUNK), on);

	dim hp[hn], bn;
	uint4 *bh = (uint4 *)malloc(sizeof(uint4) * on);
	cudaMemcpy(hp, hpd, sizeof(dim) * hn, cudaMemcpyDeviceToHost);
	bn = linearbinpacking(f1, f2, hp, bh);
	bh = (uint4 *)realloc(bh, sizeof(uint4) * bn);
	cudaMemcpyToSymbol(bd, bh, sizeof(uint4) * bn);
	printf("Used constant memory = %zu bytes\n", sizeof(uint3) * bn);

	dim i;
	for (i = 0; i < hn; i++) printf("%u * %u = %u (%zu)\n", f1.h[i], f2.h[i], hp[i], MEMORY(i));
	for (i = 0; i < bn; i++) printf("%u %u %u %u\n", bh[i].x, bh[i].y, bh[i].z, bh[i].w);

	computeoutput<<<bn, THREADSPERBLOCK>>>(f1, f2, d1d, d2d, pfxh1d, pfxh2d, pfxhpd);

	puts("Checksum...");
	printf("Checksum 1 = %u (size = %zu bytes)\n", crc32(f1.data, sizeof(chunk) * f1.n * f1.c), sizeof(chunk) * f1.n * f1.c);
	printf("Checksum Histogram 1 = %u (size = %zu bytes)\n", crc32(f1.h, sizeof(dim) * f1.hn), sizeof(dim) * f1.hn);
	printf("Checksum 2 = %u (size = %zu bytes)\n", crc32(f2.data, sizeof(chunk) * f2.n * f2.c), sizeof(chunk) * f2.n * f2.c);
	printf("Checksum Histogram 2 = %u (size = %zu bytes)\n", crc32(f2.h, sizeof(dim) * f2.hn), sizeof(dim) * f2.hn);

	cudaFree(d1d);
	cudaFree(d2d);
	cudaFree(h1d);
	cudaFree(h2d);
	cudaFree(hpd);
	cudaFree(pfxh1d);
	cudaFree(pfxh2d);
        cudaFree(pfxhpd);

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
