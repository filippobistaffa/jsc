#ifdef JSCMAIN
#include "jsc.h"
#endif

#ifdef PRINTTIME
static struct timeval t1, t2;
#endif

#ifdef __CUDACC__

#include "transpose_kernel.cu"

__constant__ uint4 bdc[CONSTANTSIZE / sizeof(uint4)];

__global__ void histogramproductkernel(dim *h1, dim *h2, dim *hr, dim hn) {

	dim tid = blockIdx.x * THREADSPERBLOCK + threadIdx.x;
	if (tid < hn) hr[tid] = h1[tid] * h2[tid];
}

template <typename type>
__global__ void cudaprintbuf(const type *buf, unsigned n) {

	if (!threadIdx.x) {
		printf("[ ");
		while (n--) printf("%u ", *(buf++));
		printf("]\n");
	}
}

__global__ void jointsumkernel(func f1, func f2, func f3, chunk *d1, chunk *d2, chunk *d3, value *v1, value *v2, value *v3, dim *pfxh1, dim *pfxh2, dim *pfxhp, uint4 *bd) {

	dim bx = blockIdx.x, tx = threadIdx.x;
	uint2 k;
	uint3 j, l;
	uint4 o = bd[bx];
	uint4 i = make_uint4(o.x, o.y, o.w / max(o.z, 1), o.w % max(o.z, 1));
	dim h, m = i.y ? 2 : i.z + 1; // numbers of pfx rows to be read
	__shared__ dim shpfx[SHAREDSIZE / sizeof(dim)];
	chunk *shd = ((chunk *)shpfx) + CEIL(3 * m * sizeof(dim), sizeof(chunk));

	assert(THREADSPERBLOCK >= m);

	if (tx < m && (tx || i.x)) {
		shpfx[tx] = pfxh1[i.x + tx - 1];
		shpfx[tx + m] = pfxh2[i.x + tx - 1];
		shpfx[tx + 2 * m] = pfxhp[i.x + tx - 1];
	}
	if (!i.x) shpfx[0] = shpfx[m] = shpfx[2 * m] = 0;
	__syncthreads();

	l = make_uint3(shpfx[0], shpfx[m], shpfx[2 * m]);
	o.x = 0;

	if (i.y) {
		j = make_uint3(CEIL(o.y = shpfx[1] - l.x, i.y), CEIL(o.w = shpfx[m + 1] - l.y, o.z), 0);
		k = make_uint2(j.x * i.z, j.y * i.w);
	}
	else {
		j = make_uint3(shpfx[i.z] - l.x, shpfx[i.z + m] - l.y, shpfx[i.z + 2 * m] - l.z);
		k = make_uint2(0, 0);
	}

	#ifdef DEBUGKERNEL
	if (!tx) printf("[" YELLOW("%3u") "," GREEN("%3u") "] i = [ .x = %3u .y = %3u .z = %3u .w = %3u ]\n", bx, tx, i.x, i.y, i.z, i.w);
	#endif

	// i.x = group id
	// i.y = bd[bx].y
	// i.z = this block will is assigned to the i.z-th chunk of the i.x-th group of d1
	// i.w = this block will is assigned to the i.w-th chunk of the i.x-th group of d2
	// o.x = offset of first output row of this block w.r.t. to l.z
	// o.y = number of d1 rows for this group
	// o.w = number of d2 rows for this group
	// o.z = bd[bx].z
	// j.x = number of lines for input 1 for this block
	// j.y = number of lines for input 2 for this block
	// l.x = first input line of d1 of this group
	// l.y = first input line of d2 of this group
	// l.z = first output line of d3 of this group
	// k.x = offset between first input line of d1 for this block and first input line of d1 of this group
	// k.y = offset between first input line of d2 for this block and first input line of d2 of this group

	// if this group is divided among multiple blocks...
	if (i.y) o.x = i.z * o.w * j.x;

	// if this block is processing the last chunk of d1
	if (i.y == i.z + 1) j.x = o.y - j.x * (i.y - 1);

	// if this group is divided among multiple blocks...
	if (i.y) o.x += j.x * j.y * i.w;

	// if this block is processing the last chunk of d1
	if (o.z == i.w + 1) j.y = o.w - j.y * (o.z - 1);

	if (i.y) j.z = j.x * j.y;

	// j.z = number of lines for output

	#ifdef DEBUGKERNEL
	if (!tx) printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] j = [ .x = % 3u .y = % 3u .z = % 3u ]\n", bx, tx, j.x, j.y, j.z);
	#endif

	#ifdef DEBUGKERNEL
	if (!tx) printf("[" YELLOW("%3u") "," GREEN("%3u") "] o = [ .x = %3u .y = %3u .z = %3u .w = %3u ]\n", bx, tx, o.x, o.y, o.z, o.w);
	#endif

	assert(THREADSPERBLOCK >= j.z);

	//if (tx < j.x * f1.c) shd[tx] = d1[(tx / j.x) * f1.n + l.x + k.x + tx % j.x];
	//if (tx < j.y * f2.c) shd[j.x * f1.c + tx] = d2[(tx / j.y) * f2.n + l.y + k.y + tx % j.y];
	if (tx < j.x) for (h = 0; h < 2 * f1.c; h++) {
		shd[h * j.x + tx] = d1[h * f1.n + l.x + k.x + tx];
		#ifdef DEBUGKERNEL
		printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] shd[% 3u] <- d1[% 3u] = %lu\n",
		       bx, tx, h * j.x + tx, h * f1.n + l.x + k.x + tx, shd[h * j.x + tx]);
		#endif
	}

	if (tx < j.y) for (h = 0; h < 2 * f2.c; h++) {
		shd[j.x * 2 * f1.c + h * j.y + tx] = d2[h * f2.n + l.y + k.y + tx];
		#ifdef DEBUGKERNEL
		printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] shd[% 3u] <- d2[% 3u] = %lu\n",
		       bx, tx, j.x * 2 * f1.c + h * j.y + tx, h * f2.n + l.y + k.y + tx, shd[j.x * 2 * f1.c + h * j.y + tx]);
		#endif
	}

	value *shv = (value *)(shd + 2 * (j.x * f1.c + j.y * f2.c + j.z * (f3.c - DIVBPC(f1.m))));

	#ifdef DEBUGKERNEL
	if (!tx) {
		printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] %u chunks reserved for d3, starting at %u\n", bx, tx,
		       j.z * 2 * (f3.c - DIVBPC(f1.m)), j.x * 2 * f1.c + j.y * 2 * f2.c);
		printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] j = [ .x = % 3u .y = % 3u .z = % 3u ] f3.c = %u, f1.m = %u, f2.m = %u\n",
		       bx, tx, j.x, j.y, j.z, f3.c, f1.m, f2.m);
	}
	#endif

	if (tx < j.x) {
		shv[tx] = v1[l.x + k.x + tx];
		#ifdef DEBUGKERNEL
		printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] shv[% 3u] <- v1[% 3u] = %f\n", bx, tx, tx, l.x + k.x + tx, v1[l.x + k.x + tx]);
		#endif
	}

	if (tx < j.y) {
		shv[j.x + tx] = v2[l.y + k.y + tx];
		#ifdef DEBUGKERNEL
		printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] shv[% 3u] <- v2[% 3u] = %f\n", bx, tx, j.x + tx, v2[l.y + k.y + tx]);
		#endif
	 }

	__syncthreads();

	if (tx < j.z) {
		o.y = 0;
		if (i.y || i.z == 1) {
			i = make_uint4(0, j.x * 2 * f1.c, j.x, j.y);
			o.z = tx;
			h = j.x;
		} else {
			for (; o.y < m - 1; o.y++) if (shpfx[o.y + 1 + 2 * m] - l.z > tx) break;
			o.z = tx - (shpfx[o.y + 2 * m] - l.z);
			i = make_uint4(shpfx[o.y], shpfx[o.y + 1], shpfx[o.y + m], shpfx[o.y + m + 1]); // fetch useful data from shared memory
			h = i.z - l.y + j.x;
			i = make_uint4(i.x - l.x, i.z - l.y + j.x * 2 * f1.c, i.y - i.x, i.w - i.z);
		}
		// o.y = which of the n groups of this block this thread belongs
		// o.z = index of this thread w.r.t. his group (in this block)
		// i.x = start of input 1 row for this group
		// i.y = start of input 2 row for this group
		// i.z = total number of input 1 rows for this group
		// i.w = total number of input 2 rows for this group

		//shv[j.x + j.y + tx] = shv[i.x + o.z / i.w] + shv[h + o.z % i.w];
		JOINTOPERATION(shv[j.x + j.y + tx], shv[i.x + o.z / i.w], shv[h + o.z % i.w]);
		#ifdef DEBUGKERNEL
		printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] shv[% 3u] = shv[% 3u] + shv[% 3u] = % 2f = % 2f + % 2f\n",
		       bx, tx, j.x + j.y + tx, i.x + o.z / i.w, h + o.z % i.w, shv[j.x + j.y + tx], shv[i.x + o.z / i.w], shv[h + o.z % i.w]);
		#endif
		i = make_uint4(i.x + o.z / i.w, i.y + o.z % i.w, MODBPC(f1.m), MODBPC(f2.s));
		chunk a, b, c;

		// if i.z = 0 (i.e., if f1.m is a multiple of BITSPERCHUNK, I don't have to copy anything from the first table
		chunk t = i.z ? shd[i.x + j.x * 2 * (f1.c - 1)] : 0;
		//#ifdef DEBUGKERNEL
		//printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] i = [ .x = % 3u .y = % 3u .z = % 3u .w = % 3u ] t = %lu\n", bx, tx, i.x, i.y, i.z, i.w, t);
		//printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] o = [ .x = % 3u .y = % 3u .z = % 3u ]\n", bx, tx, o.x, o.y, o.z);
		//#endif

		if (i.z || f2.m - f2.s) {
			h = DIVBPC(f2.s);
			a = shd[i.y + 2 * h * j.y];
			for (m = DIVBPC(f1.m); m < f3.c; m++, h++) {
				b = h == f2.c - 1 ? 0 : shd[i.y + 2 * (h + 1) * j.y];
				// a = current chunk in d2
				// b = next chunk in d2
				// i.z = MODBPC(f1.m)
				// i.w = MODBPC(f2.s)
				c = a >> i.w | b << BITSPERCHUNK - i.w;
				t = t | c << i.z;
				shd[j.x * 2 * f1.c + j.y * 2 * f2.c + 2 * (h - DIVBPC(f2.s)) * j.z + tx] = t;
				#ifdef DEBUGKERNEL
				printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] d1 = % 3u d2 = % 3u (-% 3u) h = %u shd[% 3u] = %lu\n",
				       bx, tx, i.x, i.y, j.x * 2 * f1.c, h, j.x * 2 * f1.c + j.y * f2.c + 2 * (h - DIVBPC(f2.s)) * j.z + tx, t);
				#endif
				t = c >> BITSPERCHUNK - i.z;
				a = b;
			}
		}

		//compute don't cares

		// if i.z = 0 (i.e., if f1.m is a multiple of BITSPERCHUNK, I don't have to copy anything from the first table
		t = i.z ? shd[i.x + j.x * (2 * (f1.c - 1) + 1)] : 0;
		//#ifdef DEBUGKERNEL
		//printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] i = [ .x = % 3u .y = % 3u .z = % 3u .w = % 3u ] t = %lu\n", bx, tx, i.x, i.y, i.z, i.w, t);
		//printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] o = [ .x = % 3u .y = % 3u .z = % 3u ]\n", bx, tx, o.x, o.y, o.z);
		//#endif

		if (i.z || f2.m - f2.s) {
			h = DIVBPC(f2.s);
			a = shd[i.y + (2 * h + 1) * j.y];
			for (m = DIVBPC(f1.m); m < f3.c; m++, h++) {
				b = h == f2.c - 1 ? 0 : shd[i.y + (2 * (h + 1) + 1) * j.y];
				// a = current chunk in d2
				// b = next chunk in d2
				// i.z = MODBPC(f1.m)
				// i.w = MODBPC(f2.s)
				c = a >> i.w | b << BITSPERCHUNK - i.w;
				t = t | c << i.z;
				shd[j.x * 2 * f1.c + j.y * 2 * f2.c + (2 * (h - DIVBPC(f2.s)) + 1) * j.z + tx] = t;
				#ifdef DEBUGKERNEL
				printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] d1 = % 3u d2 = % 3u (-% 3u) h = %u shd[% 3u] = %lu\n",
				       bx, tx, i.x, i.y, j.x * 2 * f1.c, h, j.x * 2 * f1.c + j.y * f2.c + (2 * (h - DIVBPC(f2.s)) + 1) * j.z + tx, t);
				#endif
				t = c >> BITSPERCHUNK - i.z;
				a = b;
			}
		}

		v3[l.z + o.x + tx] = shv[j.x + j.y + tx];
		#ifdef DEBUGKERNEL
		printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] v[% 3u] <- shv[% 3u] = %f\n",
		       bx, tx, l.z + o.x + tx, j.x + j.y + tx, v3[l.z + o.x + tx]);
		#endif

		for (h = 0; h < 2 * DIVBPC(f1.m); h++) {
			d3[l.z + o.x + h * f3.n + tx] = shd[i.x + h * j.x];
			#ifdef DEBUGKERNEL
			printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] (2) d3[% 3u] <- shd[% 3u] = %lu\n",
			       bx, tx, l.z + o.x + h * f3.n + tx, i.x + h * j.x, shd[i.x + h * j.x]);
			#endif
		}

		for (; h < 2 * f3.c; h++) {
			d3[l.z + o.x + h * f3.n + tx] = shd[j.x * 2 * f1.c + j.y * 2 * f2.c + (h - 2 * DIVBPC(f1.m)) * j.z + tx];
			#ifdef DEBUGKERNEL
			printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] (3) h = % 3u d3[% 3u] <- shd[% 3u] = %lu\n",
			       bx, tx, h, l.z + o.x + h * f3.n + tx, j.x * 2 * f1.c + j.y * 2 * f2.c + (h - 2 * DIVBPC(f1.m)) * j.z + tx,
			       d3[l.z + o.x + h * f3.n + tx]);
			#endif
		}
	}
}

/*
 * The best way (i.e., the one that maximises the number of threads) to split the group of input rows of d1 and d2
 */

__attribute__((always_inline)) inline
void solveIP(func *f1, func *f2, dim k, size_t mb, dim tb, unsigned *cx, unsigned *cy) {

	register dim rpmax = 0, i, j;
	register dim h1 = f1->h[k];
	register dim h2 = f2->h[k];
	*cx = *cy = 0;

	for (i = 1; i <= h1; i++)
		for (j = 1; j <= h2; j++) {
			register dim r1 = CEIL(h1, i);
			register dim r2 = CEIL(h2, j);
			register dim rp = r1 * r2;
			// I need enough threads and shared memory to process the rows of input and output data
			if (rp > THREADSPERBLOCK) continue;
			if (MEMORY(r1, r2, rp) > SHAREDSIZE - SHAREDMARGIN) continue;
			if (rp > rpmax) rpmax = rp, *cx = i, *cy = j;
		}

	assert(*cx && *cy);
}

__attribute__((always_inline)) inline
dim linearbinpacking(func *f1, func *f2, dim *hp, uint4 *o) {

	register dim b, i, t, j = 0, k = 0, tb = hp[0];
	register size_t m, mb = MEMORY(f1->h[0], f2->h[0], hp[0]) + 3 * sizeof(dim);
	register uint2 c = make_uint2(0, 0);

	for (i = 1; i <= f1->hn; i++)
		if ((m = MEMORY(f1->h[i], f2->h[i], hp[i])) + mb > SHAREDSIZE - SHAREDMARGIN
		     | (t = hp[i]) + tb > THREADSPERBLOCK || i == f1->hn) {
			solveIP(f1, f2, i - 1, mb, tb, &(c.x), &(c.y));
			b = c.x * c.y;
			do o[j++] = c.x * c.y > 1 ? make_uint4(k, c.x, c.y, c.x * c.y - b) : make_uint4(k, 0, 0, i - k);
			while (--b);
			mb = m + 3 * sizeof(dim);
			tb = t;
			k = i;
		}
		else mb += m, tb += t;

	return j;
}

template <typename type>
__attribute__((always_inline)) inline
void printsourcebuf(const type *buf, unsigned n, const char *name, unsigned id, const char *tname) {

	#include <iostream>
	register unsigned i;
	std::cout << tname << " " << name << id << "[] = {" << *buf;
	for (i = 1; i < n; i++) std::cout << "," << buf[i];
	puts("};");
}

#endif

__attribute__((always_inline)) inline
func jointsum(func *f1, func *f2) {

	#ifdef PRINTFUNCTIONCODE
	printf("f1->n = %u;\nf1->m = %u;\n", f1->n, f1->m);
	puts("ALLOCFUNC(f1);");
	printsourcebuf(f1->data, 2 * f1->n * f1->c, "data", 1, "chunk");
	printsourcebuf(f1->v, f1->n, "v", 1, "value");
	printsourcebuf(f1->vars, f1->m, "vars", 1, "id");
	printf("f2->n = %u;\nf2->m = %u;\n", f2->n, f2->m);
	puts("ALLOCFUNC(f2);");
	printsourcebuf(f2->data, 2 * f2->n * f2->c, "data", 2, "chunk");
	printsourcebuf(f2->v, f2->n, "v", 2, "value");
	fflush(stdout);
	#endif

	register func f3;
	register chunk *c1 = (chunk *)calloc(f1->c, sizeof(chunk));
	register chunk *c2 = (chunk *)calloc(f2->c, sizeof(chunk));
	sharedmasks(f1, c1, f2, c2);

	f1->mask = f2->mask = f3.mask = (1ULL << (f1->s % BITSPERCHUNK)) - 1;
	#ifdef PRINTINFO
	printf(MAGENTA("%u shared variables\n"), f1->s);
	#endif
	f3.s = f1->s;
	f3.mask = f1->mask;
	f3.m = f1->m + f2->m - f1->s;

	//print(f1, "f1", c1);
	//print(f2, "f2", c2);

	TIMER_START(YELLOW("Shift & Reorder..."));
	shared2least(f1, c1);
	shared2least(f2, c2);
	reordershared(f2, f1->vars);
	TIMER_STOP;

	TIMER_START(YELLOW("Instancing Don't Cares..."));
	instanceshared(f1);
	//print(f1, "f1i", cc);
	instanceshared(f2);
	//print(f2, "f2i", cc);
	TIMER_STOP;

	//BREAKPOINT("");

	#ifdef PRINTSIZE
	printf(RED("Table 1 has %u rows and %u variables\n"), f1->n, f1->m);
	printf(RED("Table 2 has %u rows and %u variables\n"), f2->n, f2->m);
	#endif

	TIMER_START(GREEN("Sort..."));
	sort(f1);
	sort(f2);
	TIMER_STOP;

	TIMER_START(YELLOW("Histogram..."));
	f1->hn = uniquecombinations(f1);
	f2->hn = uniquecombinations(f2);
	f1->h = (dim *)calloc(f1->hn, sizeof(dim));
	f2->h = (dim *)calloc(f2->hn, sizeof(dim));
	histogram(f1);
	histogram(f2);
	TIMER_STOP;

	TIMER_START(YELLOW("Matching Rows..."));
	f1->hmask = (chunk *)calloc(CEILBPC(f1->hn), sizeof(chunk));
	f2->hmask = (chunk *)calloc(CEILBPC(f2->hn), sizeof(chunk));
	dim n1, n2, hn;
	markmatchingrows(f1, f2, &n1, &n2, &hn);
	copymatchingrows(f1, f2, n1, n2, hn);
	TIMER_STOP;

	assert(f1->hn == f2->hn);
	//BREAKPOINT("GOOD?");

	#ifdef __CUDACC__
	value *v1d, *v2d, *v3d;
	chunk *d1d, *d2d, *d3d, *d1t, *d2t, *d3t;
	dim *h1d, *h2d, *hpd, *pfxh1d, *pfxh2d, *pfxhpd;

	if (f1->n && f2->n) {

		#ifdef PRINTSIZE
		printf(RED("Will allocate %zu bytes\n"), sizeof(chunk) * 2 * (f1->n * f1->c + f2->n * f2->c) +
							 sizeof(value) * (f1->n + f2->n) + sizeof(dim) * 6 * hn);
		#endif
		TIMER_START(YELLOW("Allocating..."));
		cudaMalloc(&v1d, sizeof(value) * f1->n);
		cudaMalloc(&v2d, sizeof(value) * f2->n);
		cudaMalloc(&h1d, sizeof(dim) * hn);
		cudaMalloc(&h2d, sizeof(dim) * hn);
		cudaMalloc(&hpd, sizeof(dim) * hn);
		cudaMalloc(&pfxh1d, sizeof(dim) * hn);
		cudaMalloc(&pfxh2d, sizeof(dim) * hn);
		cudaMalloc(&pfxhpd, sizeof(dim) * hn);
		TIMER_STOP;

		TIMER_START(YELLOW("Copying..."));
		cudaMemcpy(v1d, f1->v, sizeof(value) * f1->n, cudaMemcpyHostToDevice);
		cudaMemcpy(v2d, f2->v, sizeof(value) * f2->n, cudaMemcpyHostToDevice);
		cudaMemcpy(h1d, f1->h, sizeof(dim) * hn, cudaMemcpyHostToDevice);
		cudaMemcpy(h2d, f2->h, sizeof(dim) * hn, cudaMemcpyHostToDevice);
		TIMER_STOP;

		TIMER_START(GREEN("Transposing First Matrix..."));
		cudaMalloc(&d1d, sizeof(chunk) * 2 * f1->n * f1->c);
		cudaMemcpy(d1d, f1->data, sizeof(chunk) * 2 * f1->n * f1->c, cudaMemcpyHostToDevice);
		cudaMalloc(&d1t, sizeof(chunk) * 2 * f1->n * f1->c);
		dim3 grid1(CEIL(f1->n, BLOCK_DIM), CEIL(2 * f1->c, BLOCK_DIM), 1);
		//dim3 grid1((2 * f1->c) / BLOCK_DIM, f1->n / BLOCK_DIM, 1);
		//printf("%u %u %u %u\n", grid1.x, grid1.y, grid1.z, BLOCK_DIM);
		dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
		transpose<<<grid1,threads>>>(d1t, d1d, 2 * f1->c, f1->n);
		gpuerrorcheck(cudaPeekAtLastError());
		gpuerrorcheck(cudaDeviceSynchronize());
		cudaFree(d1d);
		TIMER_STOP;

		TIMER_START(GREEN("Transposing Second Matrix..."));
		cudaMalloc(&d2d, sizeof(chunk) * 2 * f2->n * f2->c);
		cudaMemcpy(d2d, f2->data, sizeof(chunk) * 2 * f2->n * f2->c, cudaMemcpyHostToDevice);
		cudaMalloc(&d2t, sizeof(chunk) * 2 * f2->n * f2->c);
		dim3 grid2(CEIL(f2->n, BLOCK_DIM), CEIL(2 * f2->c, BLOCK_DIM), 1);
		//dim3 grid2((2 * f2->c) / BLOCK_DIM, f2->n / BLOCK_DIM, 1);
		//printf("%u %u %u %u\n", grid2.x, grid2.y, grid2.z, BLOCK_DIM);
		//dim3 threads2(BLOCK_DIM, BLOCK_DIM, 1);
		transpose<<<grid2,threads>>>(d2t, d2d, 2 * f2->c, f2->n);
		gpuerrorcheck(cudaPeekAtLastError());
		gpuerrorcheck(cudaDeviceSynchronize());
		cudaFree(d2d);
		TIMER_STOP;

		histogramproductkernel<<<CEIL(hn, THREADSPERBLOCK), THREADSPERBLOCK>>>(h1d, h2d, hpd, hn);
		gpuerrorcheck(cudaPeekAtLastError());
		gpuerrorcheck(cudaDeviceSynchronize());

		// Determine temporary device storage requirements for inclusive prefix sum
		void *ts = NULL;
		size_t tsn = 0;

		cub::DeviceScan::InclusiveSum(ts, tsn, h1d, pfxh1d, hn);
		cudaMalloc(&ts, tsn);
		cub::DeviceScan::InclusiveSum(ts, tsn, h1d, pfxh1d, hn);
		cudaFree(ts);

		ts = NULL;
		tsn = 0;
		cub::DeviceScan::InclusiveSum(ts, tsn, h2d, pfxh2d, hn);
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
	}
	else f3.n = 0;

	#ifdef PRINTSIZE
	printf(RED("Result size = %zu bytes (%u lines)\n"), sizeof(chunk) * 2 * f3.n * CEILBPC(f3.m), f3.n);
	#endif

	assert(sizeof(chunk) * 2 * (f1->n * f1->c + f2->n * f2->c + f3.n * CEILBPC(f3.m)) +
	       sizeof(value) * (f1->n + f2->n + f3.n) + sizeof(dim) * 6 * hn < GLOBALSIZE);

	ALLOCFUNC(&f3);
	memcpy(f3.vars, f1->vars, sizeof(id) * f1->m);
	memcpy(f3.vars + f1->m, f2->vars + f2->s, sizeof(id) * (f2->m - f1->s));

	if (f1->n && f2->n) {

		cudaMalloc(&d3t, sizeof(chunk) * 2 * f3.n * f3.c);
		//cudaMemset(d3d, 0, sizeof(chunk) * 2 * f3.n * f3.c);
		cudaMalloc(&v3d, sizeof(value) * f3.n);

		dim hp[hn], pfxhp[hn], bn;
		uint4 *bh = (uint4 *)malloc(sizeof(uint4) * f3.n);
		cudaMemcpy(hp, hpd, sizeof(dim) * hn, cudaMemcpyDeviceToHost);

		// bn = number of blocks needed
		// each bh[i] stores the information regarding the i-th block
		// .x = this block works on the .x-th group
		// .y = the .x-th group is split into .y parts
		// .z = this block is the .z-th among the .y^2 blocks processing the .x-th group
		// notice that if a group is split into n parts, we need n^2 blocks to process it, since both f1 and f2 input
		// data rows are split into n parts, hence we have n^2 combinations

		TIMER_START(YELLOW("Bin packing..."));
		bn = linearbinpacking(f1, f2, hp, bh);
		TIMER_STOP;
		bh = (uint4 *)realloc(bh, sizeof(uint4) * bn);

		uint4 *bd;
		cudaMalloc(&bd, sizeof(uint4) * bn);
		cudaMemcpy(bd, bh, sizeof(uint4) * bn, cudaMemcpyHostToDevice);

		#ifdef PRINTSIZE
		printf(RED("%u block(s) needed\n"), bn);
		printf(RED("Groups splitting information = %zu bytes\n"), sizeof(uint4) * bn);
		#endif

		#ifdef DEBUGKERNEL
		register dim j;
		for (j = 0; j < hn; j++) printf("%u * %u = %u (%zu bytes)\n", f1->h[j], f2->h[j], hp[j], MEMORY(f1->h[j], f2->h[j], hp[j]));
		for (j = 0; j < bn; j++) printf("%3u = %3u %3u %3u %3u\n", j, bh[j].x, bh[j].y, bh[j].z, bh[j].w);
		#endif

		//assert(CONSTANTSIZE > sizeof(uint4) * bn);
		//cudaMemcpyToSymbol(bdc, bh, sizeof(uint4) * bn);

		TIMER_START(GREEN("Joint sum..."));
		jointsumkernel<<<bn, THREADSPERBLOCK>>>(*f1, *f2, f3, d1t, d2t, d3t, v1d, v2d, v3d, pfxh1d, pfxh2d, pfxhpd, bd);
		gpuerrorcheck(cudaPeekAtLastError());
		gpuerrorcheck(cudaDeviceSynchronize());
		cudaFree(d1t);
		cudaFree(d2t);
		TIMER_STOP;

		TIMER_START(GREEN("Transposing Result..."));
		cudaMalloc(&d3d, sizeof(chunk) * 2 * f3.n * f3.c);
                dim3 grid3(CEIL(f3.n, BLOCK_DIM), CEIL(2 * f3.c, BLOCK_DIM), 1);
		dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
                transposeback<<<grid3,threads>>>(d3d, d3t, f3.n, 2 * f3.c);
		gpuerrorcheck(cudaPeekAtLastError());
		gpuerrorcheck(cudaDeviceSynchronize());
		cudaFree(d3t);
                TIMER_STOP;

		cudaMemcpy(f3.data, d3d, sizeof(chunk) * 2 * f3.n * f3.c, cudaMemcpyDeviceToHost);
		cudaMemcpy(f3.v, v3d, sizeof(value) * f3.n, cudaMemcpyDeviceToHost);

		#ifdef PRINTCHECKSUM
		printf("f1i = %u\n", crc32func(f1i));
		printf("f2i = %u\n", crc32func(f2i));
		printf("f3 = %u\n", crc32func(&f3));
		#endif

		cudaFree(pfxh1d);
		cudaFree(pfxh2d);
		cudaFree(pfxhpd);
		cudaFree(d3d);
		cudaFree(v1d);
		cudaFree(v2d);
		cudaFree(v3d);
		cudaFree(h1d);
		cudaFree(h2d);
		cudaFree(hpd);
		cudaFree(bd);
		free(bh);
	}

	//print(&f3, "f3", cc);
	//BREAKPOINT("");

	#endif
	free(f1->hmask);
	free(f2->hmask);
	free(f1->h);
	free(f2->h);
	free(c1);
	free(c2);

	#ifdef __CUDACC__
	return f3;
	#else
	return *f1;
	#endif
}

#ifdef JSCMAIN

int main(int argc, char *argv[]) {

	func sf1, sf2, *f1 = &sf1, *f2 = &sf2;

	#include "functions.i"

	memcpy(f1->data, data1, sizeof(chunk) * f1->c * f1->n);
	memcpy(f1->vars, vars1, sizeof(id) * f1->m);
	memcpy(f1->v, v1, sizeof(value) * f1->n);
	//memcpy(f1->care, care1, sizeof(chunk *) * f1->n);

	memcpy(f2->data, data2, sizeof(chunk) * f2->c * f2->n);
	memcpy(f2->vars, vars2, sizeof(id) * f2->m);
	memcpy(f2->v, v2, sizeof(value) * f2->n);
	//memcpy(f2->care, care2, sizeof(chunk *) * f2->n);

	jointsum(f1, f2);

	FREEFUNC(f1);
	FREEFUNC(f2);

	return 0;
}

#endif
