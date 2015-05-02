#include "jsc.h"

#ifdef PRINTTIME
static struct timeval t1, t2;
#endif

__constant__ uint3 bd[CONSTANTSIZE / sizeof(uint3)];

__global__ void histogramproductkernel(dim *h1, dim *h2, dim *hr, dim hn) {

	dim tid = blockIdx.x * THREADSPERBLOCK + threadIdx.x;
	if (tid < hn) hr[tid] = h1[tid] * h2[tid];
}

__global__ void jointsumkernel(func f1, func f2, func f3, chunk *d1, chunk *d2, chunk *d3, value *v1, value *v2, value *v3, dim *pfxh1, dim *pfxh2, dim *pfxhp) {

	dim bx = blockIdx.x, tx = threadIdx.x;
	uint2 k;
	uint3 j, l, o = bd[bx];
	uint4 i = make_uint4(o.x, o.y, o.z / max(o.y, 1), o.z % max(o.y, 1));
	dim h, m = i.y ? 2 : i.z + 1;
	__shared__ dim shpfx[SHAREDSIZE / sizeof(dim)];
	chunk *shd = ((chunk *)shpfx) + CEIL(3 * m * sizeof(dim), sizeof(chunk));

	assert(THREADSPERBLOCK >= m);
	if (tx < m && (tx || i.x)) {
		shpfx[tx] = pfxh1[i.x + tx - 1];
		shpfx[tx + m ] = pfxh2[i.x + tx - 1];
		shpfx[tx + 2 * m] = pfxhp[i.x + tx - 1];
	}
	if (!i.x) shpfx[0] = shpfx[m] = shpfx[2 * m] = 0;
	__syncthreads();

	l = make_uint3(shpfx[0], shpfx[m], shpfx[2 * m]);
	o.x = 0;

	if (i.y) {
		j = make_uint3((o.y = shpfx[1] - l.x) / i.y, (o.z = shpfx[m + 1] - l.y) / i.y, 0);
		k = make_uint2(j.x * i.z, j.y * i.w);
	}
	else {
		j = make_uint3(shpfx[i.z] - l.x, shpfx[i.z + m] - l.y, shpfx[i.z + 2 * m] - l.z);
		k = make_uint2(0, 0);
	}

	if (i.y) o.x = (i.y * i.z + i.w) * j.x * j.y + i.z * j.x * (o.z % i.y);
	if (i.y == i.z + 1) {
		j.x += o.y % i.y;
		o.x += i.w * j.y * (o.y % i.y);
	}
	if (i.y == i.w + 1) j.y += o.z % i.y;
	if (i.y) j.z = j.x * j.y;
	assert(THREADSPERBLOCK >= j.z);

	// j.x = number of lines for input 1
	// j.y = number of lines for input 2
	// j.z = number of lines for output

	//if (tx < j.x * f1.c) shd[tx] = d1[(tx / j.x) * f1.n + l.x + k.x + tx % j.x];
	//if (tx < j.y * f2.c) shd[j.x * f1.c + tx] = d2[(tx / j.y) * f2.n + l.y + k.y + tx % j.y];
	if (tx < j.x) for (h = 0; h < f1.c; h++) shd[h * j.x + tx] = d1[h * f1.n + l.x + k.x + tx];
	if (tx < j.y) for (h = 0; h < f2.c; h++) shd[j.x * f1.c + h * j.y + tx] = d2[h * f2.n + l.y + k.y + tx];

	value *shv = (value *)(shd + j.x * f1.c + j.y * f2.c + j.z * (f3.c - f1.m / BITSPERCHUNK));
	if (tx < j.x) { shv[tx] = v1[l.x + k.x + tx]; /*printf("[%02u] (1) shv[%02u] <- %f\n", tx, tx, v1[l.x + k.x + tx]);*/ }
	if (tx < j.y) { shv[j.x + tx] = v2[l.y + k.y + tx]; /*printf("[%02u] (2) shv[%02u] <- %f\n", tx, j.x + tx, v2[l.y + k.y + tx]);*/ }

	__syncthreads();

	if (tx < j.z) {
		o.y = 0;
		if (i.y || i.z == 1) {
			i = make_uint4(0, j.x * f1.c, j.x, j.y);
			o.z = tx;
			h = j.x;
		} else {
			for (; o.y < m - 1; o.y++) if (shpfx[o.y + 1 + 2 * m] - l.z > tx) break;
			o.z = tx - (shpfx[o.y + 2 * m] - l.z);
			i = make_uint4(shpfx[o.y], shpfx[o.y + 1], shpfx[o.y + m], shpfx[o.y + m + 1]); // fetch useful data from shared memory
			h = i.z - l.y + j.x;
			i = make_uint4(i.x - l.x, i.z - l.y + j.x * f1.c, i.y - i.x, i.w - i.z);
		}
		// o.y = which of the n groups of this block this thread belongs
		// o.z = index of this thread w.r.t. his group (in this block)
		// i.x = start of input 1 row for this group
		// i.y = start of input 2 row for this group
		// i.z = total number of input 1 rows for this group
		// i.w = total number of input 2 rows for this group

		//shv[j.x + j.y + tx] = shv[i.x + o.z / i.w] + shv[h + o.z % i.w];
		JOINTOPERATION(shv[j.x + j.y + tx], shv[i.x + o.z / i.w], shv[h + o.z % i.w]);
		//printf("[%02u] shv[%u] = shv[%u] + shv[%u] = %f = %f + %f\n", tx, j.x + j.y + tx, i.x + o.z / i.w, h + o.z % i.w, shv[j.x + j.y + tx], shv[i.x + o.z / i.w], shv[h + o.z % i.w]);
		i = make_uint4(i.x + o.z / i.w, i.y + o.z % i.w, f1.m % BITSPERCHUNK, f2.s % BITSPERCHUNK);
		chunk a, b, c, t = i.z ? shd[i.x + j.x * (f1.c - 1)] : 0;
		//printf("[%02u] t=%lu\n",tx, t);
		h = f2.s / BITSPERCHUNK;
		a = shd[i.y + h * j.y];
		for (; h < f2.c; h++) {
			b = h == f2.c - 1 ? 0 : shd[i.y + (h + 1) * j.y];
			c = a >> i.w | b << BITSPERCHUNK - i.w;
			t = t | c << i.z;
			shd[j.x * f1.c + j.y * f2.c + (h - f1.m / BITSPERCHUNK) * j.z + tx] = t;
			//printf("bx=%02u tx=%02u d1=%02u d2=%02u (-%02u) h=%u output[%u]=%llu\n", bx, tx, i.x, i.y, j.x *f1.c, h, j.x * f1.c + j.y * f2.c + h * j.z + tx, t);
			t = c >> BITSPERCHUNK - i.z;
			a = b;
		}

		v3[l.z + o.x + tx] = shv[j.x + j.y + tx];
		//printf("[%02u] v[%u] = shv[%u] = %f\n", tx, l.z + o.x + tx, j.x + j.y + tx, v3[l.z + o.x + tx]);
		for (h = 0; h < f1.m / BITSPERCHUNK; h++) {
			d3[l.z + o.x + h * f3.n + tx] = shd[i.x + h * j.x];
			//printf("[%u] (1) %u <- (%u)\n", tx, l.z + o.x + h * f3.n + tx, i.x + h * j.x);
		}
		for (; h < f3.c; h++) {
			d3[l.z + o.x + h * f3.n + tx] = shd[j.x * f1.c + j.y * f2.c + (h - f1.m / BITSPERCHUNK) * j.z + tx];
			//printf("[%u] (2) %u <- (%u)\n", tx, l.z + o.x + h * f3.n + tx, j.x * f1.c + j.y * f2.c + (h - f1.m / BITSPERCHUNK) * j.z + tx);
		}
	}
}

dim linearbinpacking(func *f1, func *f2, dim *hp, uint3 *o) {

	register dim b, c, i, t, j = 0, k = 0, tb = hp[0];
	register size_t m, mb = MEMORY(0) + 3 * sizeof(dim);

	for (i = 1; i <= f1->hn; i++)
		if ((m = MEMORY(i)) + mb > SHAREDSIZE | (t = hp[i]) + tb > THREADSPERBLOCK || i == f1->hn) {
			c = (m + mb > SHAREDSIZE) ? CEIL(mb, SHAREDSIZE) : CEIL(tb, THREADSPERBLOCK);
			b = c * c;
			do o[j++] = make_uint3(k, c > 1 ? c : 0, c > 1 ? c * c - b : i - k);
			while (--b);
			mb = m + 3 * sizeof(dim);
			tb = t;
			k = i;
		}
		else mb += m, tb += t;

	return j;
}

func jointsum(func *f1, func *f2) {

	#ifdef FUNCTIONCODE
	register id i;

	printf("f1.n = %u;\nf1.m = %u;\n", f1->n, f1->m);
	printf("chunk data1[] = {%lu", f1->data[0]);
	for (i = 1; i < f1->c * f1->n; i++)
		printf(",%lu", f1->data[i]);
	puts("};");
	printf("value v1[] = {%f", f1->v[0]);
	for (i = 1; i < f1->n; i++)
		printf(",%f", f1->v[i]);
	puts("};");
	printf("id vars1[] = {%u", f1->vars[0]);
	for (i = 1; i < f1->m; i++)
		printf(",%u", f1->vars[i]);
	puts("};");

	printf("f2.n = %u;\nf2.m = %u;\n", f2->n, f2->m);
	printf("chunk data2[] = {%lu", f2->data[0]);
        for (i = 1; i < f2->c * f2->n; i++)
                printf(",%lu", f2->data[i]);
        puts("};");
        printf("value v2[] = {%f", f2->v[0]);
        for (i = 1; i < f2->n; i++)
                printf(",%f", f2->v[i]);
        puts("};");
        printf("id vars2[] = {%u", f2->vars[0]);
        for (i = 1; i < f2->m; i++)
                printf(",%u", f2->vars[i]);
        puts("};");
	#endif

	register func f3;
	register chunk *c1 = (chunk *)calloc(f1->c, sizeof(chunk));
	register chunk *c2 = (chunk *)calloc(f2->c, sizeof(chunk));
	sharedmasks(f1, c1, f2, c2);

	f1->mask = f2->mask = f3.mask = (1ULL << (f1->s % BITSPERCHUNK)) - 1;
	#ifdef PRINTINFO
	printf("%u shared variables\n", f1->s);
	#endif
	//if (!f1->s) return 1;
	f3.s = f1->s;

	TIMER_START("Shift & Reorder...");
	shared2least(*f1, c1);
	shared2least(*f2, c2);
	reordershared(*f2, f1->vars);
	TIMER_STOP;

	TIMER_START("Sort...");
	sort(*f1);
	sort(*f2);
	TIMER_STOP;

	f1->hn = uniquecombinations(*f1);
	f2->hn = uniquecombinations(*f2);
	#ifdef PRINTINFO
	printf("%u unique combinations\n", f1->hn);
	printf("%u unique combinations\n", f2->hn);
	#endif
	f1->h = (dim *)calloc(f1->hn, sizeof(dim));
	f2->h = (dim *)calloc(f2->hn, sizeof(dim));

	TIMER_START("Histogram...");
	histogram(*f1);
	histogram(*f2);
	TIMER_STOP;

	TIMER_START("Matching Rows...");
	f1->hmask = (chunk *)calloc(CEIL(f1->hn, BITSPERCHUNK), sizeof(chunk));
	f2->hmask = (chunk *)calloc(CEIL(f2->hn, BITSPERCHUNK), sizeof(chunk));
	dim n1, n2, hn;
	markmatchingrows(*f1, *f2, &n1, &n2, &hn);
	copymatchingrows(f1, f2, n1, n2, hn);
	TIMER_STOP;

	#ifdef PRINTINFO
	printf("%u matching rows\n", f1->n);
	print(*f1);
	printf("%u matching rows\n", f2->n);
	print(*f2);
	#endif

	assert(f1->n && f2->n);

	chunk *d1d, *d2d, *d3d;
	value *v1d, *v2d, *v3d;
	dim *h1d, *h2d, *hpd, *pfxh1d, *pfxh2d, *pfxhpd;
	TIMER_START("Allocating... ");
	cudaMalloc(&d1d, sizeof(chunk) * f1->n * f1->c);
	cudaMalloc(&d2d, sizeof(chunk) * f2->n * f2->c);
	cudaMalloc(&v1d, sizeof(value) * f1->n);
        cudaMalloc(&v2d, sizeof(value) * f2->n);
	cudaMalloc(&h1d, sizeof(dim) * hn);
	cudaMalloc(&h2d, sizeof(dim) * hn);
	cudaMalloc(&hpd, sizeof(dim) * hn);
        cudaMalloc(&pfxh1d, sizeof(dim) * hn);
        cudaMalloc(&pfxh2d, sizeof(dim) * hn);
        cudaMalloc(&pfxhpd, sizeof(dim) * hn);
	TIMER_STOP;

	cudaMemcpy(d1d, f1->data, sizeof(chunk) * f1->n * f1->c, cudaMemcpyHostToDevice);
	cudaMemcpy(d2d, f2->data, sizeof(chunk) * f2->n * f2->c, cudaMemcpyHostToDevice);
        cudaMemcpy(v1d, f1->v, sizeof(value) * f1->n, cudaMemcpyHostToDevice);
        cudaMemcpy(v2d, f2->v, sizeof(value) * f2->n, cudaMemcpyHostToDevice);
	cudaMemcpy(h1d, f1->h, sizeof(dim) * hn, cudaMemcpyHostToDevice);
	cudaMemcpy(h2d, f2->h, sizeof(dim) * hn, cudaMemcpyHostToDevice);

	histogramproductkernel<<<CEIL(hn, THREADSPERBLOCK), THREADSPERBLOCK>>>(h1d, h2d, hpd, hn);
	gpuerrorcheck(cudaPeekAtLastError());
	gpuerrorcheck(cudaDeviceSynchronize());

	// Determine temporary device storage requirements for inclusive prefix sum
	void *ts = NULL;
	size_t tsn = 0;

	cub::DeviceScan::InclusiveSum(ts, tsn, h1d, pfxh1d, hn);
	#ifdef PRINTSIZE
	printf("Temporary storage for prefix sum = %zu bytes\n", tsn);
	#endif
	cudaMalloc(&ts, tsn);
	cub::DeviceScan::InclusiveSum(ts, tsn, h1d, pfxh1d, hn);
	cudaFree(ts);

	ts = NULL;
	tsn = 0;
	cub::DeviceScan::InclusiveSum(ts, tsn, h2d, pfxh2d, hn);
	#ifdef PRINTSIZE
	printf("Temporary storage for prefix sum = %zu bytes\n", tsn);
	#endif
	cudaMalloc(&ts, tsn);
	cub::DeviceScan::InclusiveSum(ts, tsn, h2d, pfxh2d, hn);
	cudaFree(ts);

	ts = NULL;
	tsn = 0;
	cub::DeviceScan::InclusiveSum(ts, tsn, hpd, pfxhpd, hn);
	cudaMalloc(&ts, tsn);
	cub::DeviceScan::InclusiveSum(ts, tsn, hpd, pfxhpd, hn);
	cudaFree(ts);

	cudaMemcpy(&f3.n, pfxhpd + hn - 1, sizeof(dim), cudaMemcpyDeviceToHost);
	f3.m = f1->m + f2->m - f1->s;

	ALLOCFUNC(f3, chunk, id, value);
	#ifdef PRINTSIZE
	printf("Result size = %zu bytes (%u lines)\n", sizeof(chunk) * f3.n * f3.c, f3.n);
	#endif
	cudaMalloc(&d3d, sizeof(chunk) * f3.n * f3.c);
	cudaMalloc(&v3d, sizeof(value) * f3.n);
	memcpy(f3.vars, f1->vars, sizeof(id) * f1->m);
	memcpy(f3.vars + f1->m, f2->vars + f2->s, sizeof(id) * (f2->m - f1->s));

	dim hp[hn], bn;
	uint3 *bh = (uint3 *)malloc(sizeof(uint3) * f3.n);
	cudaMemcpy(hp, hpd, sizeof(dim) * hn, cudaMemcpyDeviceToHost);

	// bn = number of blocks needed
	// each bh[i] stores the information regarding the "i"-th block
	// .x =
	// .y =
	// .z =

	bn = linearbinpacking(f1, f2, hp, bh);
	bh = (uint3 *)realloc(bh, sizeof(uint3) * bn);
	#ifdef PRINTSIZE
	printf("%u blocks needed\n", bn);
	printf("Needed constant memory = %zu bytes (Max = %u bytes)\n", sizeof(uint3) * bn, CONSTANTSIZE);
	#endif
	assert(CONSTANTSIZE > sizeof(uint3) * bn);
	cudaMemcpyToSymbol(bd, bh, sizeof(uint3) * bn);

	//dim i;
	//for (i = 0; i < hn; i++) printf("%u * %u = %u (%zu bytes)\n", f1->h[i], f2->h[i], hp[i], MEMORY(i));
	//for (i = 0; i < bn; i++) printf("%u %u %u\n", bh[i].x, bh[i].y, bh[i].z);

	jointsumkernel<<<bn, THREADSPERBLOCK>>>(*f1, *f2, f3, d1d, d2d, d3d, v1d, v2d, v3d, pfxh1d, pfxh2d, pfxhpd);
	gpuerrorcheck(cudaPeekAtLastError());
	gpuerrorcheck(cudaDeviceSynchronize());

	cudaMemcpy(f3.data, d3d, sizeof(chunk) * f3.n * f3.c, cudaMemcpyDeviceToHost);
	cudaMemcpy(f3.v, v3d, sizeof(value) * f3.n, cudaMemcpyDeviceToHost);

	// Order output table for debugging purposes
	//f3.s = f3.m;
	//f3.mask = (1ULL << (f3.s % BITSPERCHUNK)) - 1;
	//sort(f3);
	//print(f1, NULL);
	//print(f2, NULL);
	//print(f3, NULL);

	#ifdef PRINTCHECKSUM
	puts("Checksum...");
	printf("Checksum Data 1 = %u (size = %zu bytes)\n", crc32(f1->data, sizeof(chunk) * f1->n * f1->c), sizeof(chunk) * f1->n * f1->c);
	printf("Checksum Values 1 = %u (size = %zu bytes)\n", crc32(f1->v, sizeof(value) * f1->n), sizeof(value) * f1->n);
	printf("Checksum Histogram 1 = %u (size = %zu bytes)\n", crc32(f1->h, sizeof(dim) * f1->hn), sizeof(dim) * f1->hn);
	printf("Checksum Data 2 = %u (size = %zu bytes)\n", crc32(f2->data, sizeof(chunk) * f2->n * f2->c), sizeof(chunk) * f2->n * f2->c);
	printf("Checksum Values 2 = %u (size = %zu bytes)\n", crc32(f2->v, sizeof(value) * f2->n), sizeof(value) * f2->n);
	printf("Checksum Histogram 2 = %u (size = %zu bytes)\n", crc32(f2->h, sizeof(dim) * f2->hn), sizeof(dim) * f2->hn);
	printf("Checksum Output Data = %u (size = %zu bytes)\n", crc32(f3.data, sizeof(chunk) * f3.n * f3.c), sizeof(chunk) * f3.n * f3.c);
	printf("Checksum Output Values = %u (size = %zu bytes)\n", crc32(f3.v, sizeof(value) * f3.n), sizeof(value) * f3.n);
	#endif

	cudaFree(d1d);
	cudaFree(d2d);
	cudaFree(d3d);
	cudaFree(v1d);
	cudaFree(v2d);
	cudaFree(v3d);
	cudaFree(h1d);
	cudaFree(h2d);
	cudaFree(hpd);
	cudaFree(pfxh1d);
	cudaFree(pfxh2d);
        cudaFree(pfxhpd);
	free(f1->hmask);
	free(f2->hmask);
	free(f1->h);
	free(f2->h);
	free(c1);
	free(c2);
	free(bh);

	return f3;
}

#ifdef JSCMAIN

int main(int argc, char *argv[]) {

	func f1, f2, f3;
	init_genrand64(SEED);
	srand(SEED);

	f1.n = 1000;
	f1.m = 80;
	f2.n = 3000;
	f2.m = 100;
	ALLOCFUNC(f1, chunk, id, value);
	ALLOCFUNC(f2, chunk, id, value);

	if (!f1.data || !f2.data) {
		printf("Not enough memory!\n");
		return 1;
	}

	printf("Random data... ");
	fflush(stdout);
	gettimeofday(&t1, NULL);
        randomvars(f1);
        randomvars(f2);
        randomdata(f1);
        randomdata(f2);
        randomvalues(f1);
        randomvalues(f2);
	gettimeofday(&t2, NULL);
	printf("%f seconds\n", (double)(t2.tv_usec - t1.tv_usec) / 1e6 + t2.tv_sec - t1.tv_sec);

	f3 = jointsum(&f1, &f2);

	FREEFUNC(f1);
	FREEFUNC(f2);
	FREEFUNC(f3);

	return 0;
}

#endif
