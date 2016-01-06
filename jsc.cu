#ifdef JSCMAIN
#include "jsc.h"
#endif

static struct timeval t1a, t2a;
static double at = 0;

__attribute__((always_inline)) inline
void joinsumhost(const func *f1, const func *f2, const func *f3,
		 const dim *h1, const dim *hp, const dim *pfxh1, const dim *pfxh2, const dim *pfxhp, dim hn) {

	register const dim dm1 = DIVBPC(f1->m);
	register const dim mm1 = MODBPC(f1->m);
	register const dim ds2 = DIVBPC(f2->s);
	register const dim ms2 = MODBPC(f2->s);

	for (dim b = 0; b < hn; b++) {

		register const dim h1b = h1[b];
		register const dim hpb = hp[b];
		register const dim pfxh1b = pfxh1[b];
		register const dim pfxh2b = pfxh2[b];
		register const dim pfxhpb = pfxhp[b];

		#pragma omp parallel for
		for (dim t = 0; t < hpb; t++) {

			register const dim r1 = pfxh1b + t % h1b;
			register const dim r2 = pfxh2b + t / h1b;
			register const dim r3 = pfxhpb + t;
			JOINOPERATION(f3->v[r3], f1->v[r1], f2->v[r2]);
			memcpy(DATA(f3, r3), DATA(f1, r1), sizeof(chunk) * dm1);
			chunk d = mm1 ? DATA(f1, r1)[f1->c - 1] : 0;

			if (mm1 || f2->m - f2->s) {
				register chunk a = DATA(f2, r2)[ds2], b, c;
				for (dim m = dm1, h = ds2; m < f3->c; m++, h++) {
					b = h == f2->c - 1 ? 0 : DATA(f2, r2)[h + 1];
					c = a >> ms2 | b << BITSPERCHUNK - ms2;
					d = d | c << mm1;
					DATA(f3, r3)[dm1 + h - ds2] = d;
					d = c >> BITSPERCHUNK - mm1;
					a = b;
				}
			}
		}
	}
}

#ifdef __CUDACC__

#ifndef INPLACE
#include "transpose_kernel.cu"
#endif

//__constant__ uint4 bdc[CONSTANTSIZE / sizeof(uint4)];

__global__ void histogramproductkernel(dim *h1, dim *h2, dim *hr, dim hn) {

	dim tid = blockIdx.x * THREADSPERBLOCK + threadIdx.x;
	if (tid < hn) hr[tid] = h1[tid] * h2[tid];
}

/*template <typename type>
__global__ void cudaprintbuf(const type *buf, unsigned n) {

	if (!threadIdx.x) {
		printf("[ ");
		while (n--) printf("%u ", *(buf++));
		printf("]\n");
	}
}*/

__global__ void joinsumkernel(func f1, func f2, func f3, chunk *d1, chunk *d2, chunk *d3, value *v1, value *v2, value *v3,
			       dim f1nr, dim f2nr, dim f3nr, dim *pfxh1, dim *pfxh2, dim *pfxhp, uint4 *bd) {

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
	if (!tx) printf("[" YELLOW("%3u") "," GREEN("%3u") "] i = [ .x = %3u .y = %3u .z = %3u .w = %3u ]\n",
			bx, tx, i.x, i.y, i.z, i.w);
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
	if (!tx) printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] j = [ .x = % 3u .y = % 3u .z = % 3u ]\n",
			bx, tx, j.x, j.y, j.z);
        #endif

        #ifdef DEBUGKERNEL
	if (!tx) printf("[" YELLOW("%3u") "," GREEN("%3u") "] o = [ .x = %3u .y = %3u .z = %3u .w = %3u ]\n",
			bx, tx, o.x, o.y, o.z, o.w);
        #endif

	assert(THREADSPERBLOCK >= j.z);

	//if (tx < j.x * f1.c) shd[tx] = d1[(tx / j.x) * f1nr + l.x + k.x + tx % j.x];
	//if (tx < j.y * f2.c) shd[j.x * f1.c + tx] = d2[(tx / j.y) * f2nr + l.y + k.y + tx % j.y];
	if (tx < j.x) for (h = 0; h < f1.c; h++) {
		shd[h * j.x + tx] = d1[h * f1nr + l.x + k.x + tx];
		#ifdef DEBUGKERNEL
                printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] shd[% 3u] <- d1[% 3u] = %lu\n",
		       bx, tx, h * j.x + tx, h * f1nr + l.x + k.x + tx, shd[h * j.x + tx]);
                #endif
	}

	if (tx < j.y) for (h = 0; h < f2.c; h++) {
		shd[j.x * f1.c + h * j.y + tx] = d2[h * f2nr + l.y + k.y + tx];
		#ifdef DEBUGKERNEL
                printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] shd[% 3u] <- d2[% 3u] = %lu\n",
		       bx, tx, j.x * f1.c + h * j.y + tx, h * f2nr + l.y + k.y + tx, shd[j.x * f1.c + h * j.y + tx]);
                #endif
	}

	value *shv = (value *)(shd + j.x * f1.c + j.y * f2.c + j.z * (f3.c - f1.m / BITSPERCHUNK));

	#ifdef DEBUGKERNEL
	if (!tx) {
		printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] %u chunks reserved for d3, starting at % 3u\n", bx, tx,
		       j.z * (f3.c - f1.m / BITSPERCHUNK), j.x * f1.c + j.y * f2.c);
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
		JOINOPERATION(shv[j.x + j.y + tx], shv[i.x + o.z / i.w], shv[h + o.z % i.w]);
		#ifdef DEBUGKERNEL
		printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] shv[% 3u] = shv[% 3u] + shv[% 3u] = % 2f = % 2f + % 2f\n",
		       bx, tx, j.x + j.y + tx, i.x + o.z / i.w, h + o.z % i.w, shv[j.x + j.y + tx], shv[i.x + o.z / i.w], shv[h + o.z % i.w]);
		#endif
		i = make_uint4(i.x + o.z / i.w, i.y + o.z % i.w, MODBPC(f1.m), MODBPC(f2.s));
		chunk a, b, c;

		// if i.z = 0 (i.e., if f1.m is a multiple of BITSPERCHUNK, I don't have to copy anything from the first table
		chunk t = i.z ? shd[i.x + j.x * (f1.c - 1)] : 0;
		#ifdef DEBUGKERNEL
		printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] i = [ .x = % 3u .y = % 3u .z = % 3u .w = % 3u ] t = %lu\n", bx, tx, i.x, i.y, i.z, i.w, t);
		printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] o = [ .x = % 3u .y = % 3u .z = % 3u ]\n", bx, tx, o.x, o.y, o.z);
		#endif

		if (i.z || f2.m - f2.s) {
			h = DIVBPC(f2.s);
			a = shd[i.y + h * j.y];
			for (m = DIVBPC(f1.m); m < f3.c; m++, h++) {
				b = h == f2.c - 1 ? 0 : shd[i.y + (h + 1) * j.y];
				// a = current chunk in d2
				// b = next chunk in d2
				// i.z = MODBPC(f1.m)
				// i.w = MODBPC(f2.s)
				c = a >> i.w | b << BITSPERCHUNK - i.w;
				t = t | c << i.z;
				shd[j.x * f1.c + j.y * f2.c + (h - DIVBPC(f2.s)) * j.z + tx] = t;
				#ifdef DEBUGKERNEL
				printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] d1 = % 3u d2 = % 3u (-% 3u) h = %u shd[% 3u] = %lu\n",
				       bx, tx, i.x, i.y, j.x * f1.c, h, j.x * f1.c + j.y * f2.c + (h - DIVBPC(f2.s)) * j.z + tx, t);
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

		for (h = 0; h < DIVBPC(f1.m); h++) {
			d3[l.z + o.x + h * f3nr + tx] = shd[i.x + h * j.x];
			#ifdef DEBUGKERNEL
			printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] (2) d3[% 3u] <- shd[% 3u] = %lu\n",
			       bx, tx, l.z + o.x + h * f3nr + tx, i.x + h * j.x, shd[i.x + h * j.x]);
			#endif
		}

		for (; h < f3.c; h++) {
			d3[l.z + o.x + h * f3nr + tx] = shd[j.x * f1.c + j.y * f2.c + (h - DIVBPC(f1.m)) * j.z + tx];
			#ifdef DEBUGKERNEL
			printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] (3) h = % 3u d3[% 3u] <- shd[% 3u] = %lu\n",
			       bx, tx, h, l.z + o.x + h * f3nr + tx, j.x * f1.c + j.y * f2.c + (h - DIVBPC(f1.m)) * j.z + tx,
			       d3[l.z + o.x + h * f3nr + tx]);
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
			if (rp >= THREADSPERBLOCK) continue;
			if (MEMORY(r1, r2, rp) > SHAREDSIZE - SHAREDMARGIN) continue;
			if (rp > rpmax) rpmax = rp, *cx = i, *cy = j;
		}

	assert(*cx && *cy);
}

__attribute__((always_inline)) inline
dim linearbinpacking(func *f1, func *f2, dim *hp, uint4 *o, dim *ho, dim *hi) {

	register dim b, i, t, j = 0, ko = 0, k = 0, tb = hp[0];

	register size_t m, r;
	register size_t mb = MEMORY(f1->h[0], f2->h[0], hp[0]) + 3 * sizeof(dim);
	register size_t gb = mb;
	register size_t rb = RESULTDATA(hp[0]);

	register uint2 c = make_uint2(0, 0);
	register dim idx = 0;

	register size_t free, total;
	cudaMemGetInfo(&free, &total);
	register const size_t ag = free - sizeof(dim) * 6 * f1->hn;

	#ifdef PRINTSIZE
	printf(RED("%zu free bytes / %zu total bytes on the GPU\n"), free, total);
	#endif

	for (i = 1; i <= f1->hn; i++) {

		m = i == f1->hn ? 0 : MEMORY(f1->h[i], f2->h[i], hp[i]);
		r = i == f1->hn ? 0 : RESULTDATA(hp[i]);
		assert(m <= ag);
		assert(r <= ag / TRANSPOSEFACTOR);

		if (m + mb > SHAREDSIZE - SHAREDMARGIN | (t = hp[i]) + tb >= THREADSPERBLOCK || i == f1->hn) {
			solveIP(f1, f2, i - 1, mb, tb, &(c.x), &(c.y));
			b = c.x * c.y;
			do o[j++] = c.x * c.y > 1 ? make_uint4(k - ko, c.x, c.y, c.x * c.y - b) : make_uint4(k - ko, 0, 0, i - k);
			while (--b);
			mb = m + 3 * sizeof(dim);
			tb = t;
			k = i;

			ho[idx] = idx ? j - ho[idx - 1] : j;
			hi[idx] = idx ? i - hi[idx - 1] : i;
			//printbuf(ho, idx + 1, "ho");
			//printbuf(hi, idx + 1, "hi");

			if (m + gb > ag | r + rb > ag / TRANSPOSEFACTOR) {
				gb = m + 3 * sizeof(dim);
				rb = r;
				ko = k;
				idx++;
			} else gb += m, rb += r;
		}
		else mb += m, tb += t, gb += m, rb += r;

	}

	return idx + 1;
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
func joinsum(func *f1, func *f2) {

	assert(f1->n && f2->n);

	#ifdef PRINTFUNCTIONCODE
	printf("f1->n = %u;\nf1->m = %u;\n", f1->n, f1->m);
	puts("ALLOCFUNC(f1);");
	printsourcebuf(f1->data, f1->n * f1->c, "data", 1, "chunk");
	printsourcebuf(f1->v, f1->n, "v", 1, "value");
	printsourcebuf(f1->vars, f1->m, "vars", 1, "id");
	printf("f2->n = %u;\nf2->m = %u;\n", f2->n, f2->m);
	puts("ALLOCFUNC(f2);");
	printsourcebuf(f2->data, f2->n * f2->c, "data", 2, "chunk");
	printsourcebuf(f2->v, f2->n, "v", 2, "value");
	fflush(stdout);
	#endif

	#ifdef PRINTINFO
	printf(MAGENTA("Table 1 has %u rows and %u variables (%zu bytes)\n"), f1->n, f1->m, (sizeof(chunk) * f1->c + sizeof(value)) * f1->n);
	printf(MAGENTA("Table 2 has %u rows and %u variables (%zu bytes)\n"), f2->n, f2->m, (sizeof(chunk) * f2->c + sizeof(value)) * f2->n);
	#endif

	register func f3;
	register chunk *c1 = (chunk *)calloc(f1->c, sizeof(chunk));
	register chunk *c2 = (chunk *)calloc(f2->c, sizeof(chunk));
	sharedmasks(f1, c1, f2, c2);

	f1->mask = f2->mask = f3.mask = (ONE << MODBPC(f1->s)) - 1;
	#ifdef PRINTINFO
	printf(MAGENTA("%u shared variables\n"), f1->s);
	#endif
	f3.s = f1->s;
	f3.mask = f1->mask;
	f3.m = f1->m + f2->m - f1->s;

	#ifdef PRINTDEBUG
	print(f1, "f1", c1);
	print(f2, "f2", c2);
	#endif

	ADDTIME_START;
	TIMER_START(YELLOW("Shift & Reorder..."));
	shared2least(f1, c1);
	shared2least(f2, c2);
	reordershared(f2, f1->vars);
	TIMER_STOP;

	#ifdef PRINTDEBUG
	memset(c1, 0xFF, sizeof(chunk) * DIVBPC(f1->s));
	if (f1->mask) c1[DIVBPC(f1->s)] = f1->mask;
	print(f1, "f1 after shift and reorder", c1);
	print(f2, "f2 after shift and reorder", c1);
	#endif

	sort(f1);
	sort(f2);

	#ifdef PRINTDEBUG
	print(f1, "f1 after sort", c1);
	print(f2, "f2 after sort", c1);
	#endif

	TIMER_START(YELLOW("Histogram..."));
	f1->hn = uniquecombinations(f1);
	f2->hn = uniquecombinations(f2);
	f1->h = (dim *)calloc(f1->hn, sizeof(dim));
	f2->h = (dim *)calloc(f2->hn, sizeof(dim));
	histogram(f1);
	histogram(f2);
	TIMER_STOP;

	#ifdef PRINTDEBUG
	printbuf(f1->h, f1->hn, "f1->h");
	printbuf(f2->h, f2->hn, "f2->h");
	#endif

	TIMER_START(YELLOW("Matching Rows..."));
	f1->hmask = (chunk *)calloc(CEILBPC(f1->hn), sizeof(chunk));
	f2->hmask = (chunk *)calloc(CEILBPC(f2->hn), sizeof(chunk));
	dim n1, n2, hn;
	markmatchingrows(f1, f2, &n1, &n2, &hn);
	copymatchingrows(f1, f2, n1, n2, hn);
	TIMER_STOP;
	ADDTIME_STOP;

	#ifdef PRINTDEBUG
	print(f1, "f1 after matching", c1);
	print(f2, "f2 after matching", c1);
	printbuf(f1->h, hn, "f1->h");
	printbuf(f2->h, hn, "f2->h");
	#endif

	#ifdef PRINTINFO
	printf(MAGENTA("Table 1 has %u rows and %u variables (%zu bytes) after matching\n"), f1->n, f1->m, (sizeof(chunk) * f1->c + sizeof(value)) * f1->n);
	printf(MAGENTA("Table 2 has %u rows and %u variables (%zu bytes) after matching\n"), f2->n, f2->m, (sizeof(chunk) * f2->c + sizeof(value)) * f2->n);
	printf(MAGENTA("%u histogram groups\n"), hn);
	#endif

	if (!hn) {
		printf("Not satisfiable\n");
		printf("Time = %f\n", at - at / SPEEDUP);
		exit(0);
	}

	assert(f1->hn == f2->hn);

	#ifdef __CUDACC__
	dim *h1d, *h2d, *hpd, *pfxh1d, *pfxh2d, *pfxhpd, *f3n;
	void *ts = NULL;
	size_t tsn = 0;

	cudaMalloc(&h1d, sizeof(dim) * hn);
	cudaMalloc(&h2d, sizeof(dim) * hn);
	cudaMalloc(&hpd, sizeof(dim) * hn);
	cudaMalloc(&pfxh1d, sizeof(dim) * hn);
	cudaMalloc(&pfxh2d, sizeof(dim) * hn);
	cudaMalloc(&pfxhpd, sizeof(dim) * hn);
	cudaMemcpy(h1d, f1->h, sizeof(dim) * hn, cudaMemcpyHostToDevice);
	cudaMemcpy(h2d, f2->h, sizeof(dim) * hn, cudaMemcpyHostToDevice);
	dim *hp = (dim *)malloc(sizeof(dim) * hn);

	if (hn > THRESHOLD) {
		histogramproductkernel<<<CEIL(hn, THREADSPERBLOCK), THREADSPERBLOCK>>>(h1d, h2d, hpd, hn);
		cudaMemcpy(hp, hpd, sizeof(dim) * hn, cudaMemcpyDeviceToHost);
		GPUERRORCHECK;
		cudaMalloc(&f3n, sizeof(dim));
		cub::DeviceReduce::Sum(ts, tsn, hpd, f3n, hn);
		cudaMalloc(&ts, tsn);
		cub::DeviceReduce::Sum(ts, tsn, hpd, f3n, hn);
		cudaMemcpy(&f3.n, f3n, sizeof(dim), cudaMemcpyDeviceToHost);
	} else {
		bufproduct(f1->h, f2->h, hp, hn);
		cudaMemcpy(hpd, hp, sizeof(dim) * hn, cudaMemcpyHostToDevice);
		f3.n = sumreduce(hp, hn);
	}

	#ifdef PRINTSIZE
	printf(RED("Total result size = %zu bytes (%u lines)\n"), sizeof(chunk) * f3.n * CEILBPC(f3.m), f3.n);
	#endif

	ADDTIME_START;
	ALLOCFUNC(&f3);
	memcpy(f3.vars, f1->vars, sizeof(id) * f1->m);
	memcpy(f3.vars + f1->m, f2->vars + f2->s, sizeof(id) * (f2->m - f1->s));
	dim *pfxh1 = (dim *)malloc(sizeof(dim) * hn);
	dim *pfxh2 = (dim *)malloc(sizeof(dim) * hn);
	dim *pfxhp = (dim *)malloc(sizeof(dim) * hn);
	exclprefixsum(f1->h, pfxh1, hn);
	exclprefixsum(f2->h, pfxh2, hn);
	exclprefixsum(hp, pfxhp, hn);

	if (hn > THRESHOLD) {

		// bn = number of blocks needed
		// each bh[i] stores the information regarding the i-th block
		// .x = this block works on the .x-th group
		// .y = the .x-th group is split into .y parts
		// .z = this block is the .z-th among the .y^2 blocks processing the .x-th group
		// notice that if a group is split into n parts, we need n^2 blocks to process it, since both f1 and f2 input
		// data rows are split into n parts, hence we have n^2 combinations

		uint4 *bh = (uint4 *)malloc(sizeof(uint4) * f3.n);
		dim *ho = (dim *)malloc(sizeof(dim) * f3.n);
		dim *hi = (dim *)malloc(sizeof(dim) * f3.n);

		//TIMER_START(YELLOW("Bin packing..."));
		register dim runs = linearbinpacking(f1, f2, hp, bh, ho, hi);
		//TIMER_STOP;

		assert(runs <= hn);
		dim *pfxho = (dim *)malloc(sizeof(dim) * runs);
		dim *pfxhi = (dim *)malloc(sizeof(dim) * runs);
		exclprefixsum(ho, pfxho, runs);
		exclprefixsum(hi, pfxhi, runs);
		ADDTIME_STOP;

		for (dim r = 0; r < runs; r++) {

			if (runs > 1) printf(BLUE("Step %u of %u...\n"), r + 1, runs);
			register const dim bn = ho[r];

			uint4 *bd;
			cudaMalloc(&bd, sizeof(uint4) * bn);
			cudaMemcpy(bd, bh + pfxho[r], sizeof(uint4) * bn, cudaMemcpyHostToDevice);

			#ifdef PRINTSIZE
			printf(RED("%u block(s) needed\n"), bn);
			printf(RED("Groups splitting information = %zu bytes\n"), sizeof(uint4) * bn);
			#endif

			ts = NULL;
			tsn = 0;
			cub::DeviceScan::InclusiveSum(ts, tsn, h1d + pfxhi[r], pfxh1d, hi[r]);
			cudaMalloc(&ts, tsn);
			cub::DeviceScan::InclusiveSum(ts, tsn, h1d + pfxhi[r], pfxh1d, hi[r]);
			cudaFree(ts);

			ts = NULL;
			tsn = 0;
			cub::DeviceScan::InclusiveSum(ts, tsn, h2d + pfxhi[r], pfxh2d, hi[r]);
			cudaMalloc(&ts, tsn);
			cub::DeviceScan::InclusiveSum(ts, tsn, h2d + pfxhi[r], pfxh2d, hi[r]);
			cudaFree(ts);

			ts = NULL;
			tsn = 0;
			cub::DeviceScan::InclusiveSum(ts, tsn, hpd + pfxhi[r], pfxhpd, hi[r]);
			cudaMalloc(&ts, tsn);
			cub::DeviceScan::InclusiveSum(ts, tsn, hpd + pfxhi[r], pfxhpd, hi[r]);
			cudaFree(ts);

			value *v1d, *v2d, *v3d; // function values
			chunk *d1d, *d2d, *d3d; // original matrices
			chunk *d1t, *d2t, *d3t; // transposed matrices

			register const dim f1nr = (r == runs - 1 ? f1->n : pfxh1[pfxhi[r + 1]]) - pfxh1[pfxhi[r]];
			register const dim f2nr = (r == runs - 1 ? f2->n : pfxh2[pfxhi[r + 1]]) - pfxh2[pfxhi[r]];
			register const dim f3nr = (r == runs - 1 ? f3.n : pfxhp[pfxhi[r + 1]]) - pfxhp[pfxhi[r]];

			#ifdef PRINTSIZE
			if (runs > 1) printf(RED("Run result size = %zu bytes (%u lines)\n"), sizeof(chunk) * f3nr * CEILBPC(f3.m), f3nr);
			#endif

			cudaMalloc(&v1d, sizeof(value) * f1nr);
			cudaMalloc(&v2d, sizeof(value) * f2nr);
			cudaMemcpy(v1d, f1->v + pfxh1[pfxhi[r]], sizeof(value) * f1nr, cudaMemcpyHostToDevice);
			cudaMemcpy(v2d, f2->v + pfxh2[pfxhi[r]], sizeof(value) * f2nr, cudaMemcpyHostToDevice);

			cudaMalloc(&d1d, sizeof(chunk) * f1nr * f1->c);
			cudaMemcpy(d1d, DATA(f1, pfxh1[pfxhi[r]]), sizeof(chunk) * f1nr * f1->c, cudaMemcpyHostToDevice);

			#ifndef INPLACE
			dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
			#endif

			if (f1->c > 1 && f1nr > 1) {
				#ifdef INPLACE
				TIMER_START(GREEN("Transposing First Matrix Inplace..."));
				inplace::transpose(1, (double *)d1d, f1nr, f1->c);
				d1t = d1d;
				#else
				TIMER_START(GREEN("Transposing First Matrix..."));
				cudaMalloc(&d1t, sizeof(chunk) * f1nr * f1->c);
				dim3 grid1(CEIL(f1nr, BLOCK_DIM), CEIL(f1->c, BLOCK_DIM), 1);
				transpose<<<grid1,threads>>>(d1t, d1d, f1->c, f1nr);
				cudaFree(d1d);
				#endif
				TIMER_STOP;
			} else d1t = d1d;

			cudaMalloc(&d2d, sizeof(chunk) * f2nr * f2->c);
			cudaMemcpy(d2d, DATA(f2, pfxh2[pfxhi[r]]), sizeof(chunk) * f2nr * f2->c, cudaMemcpyHostToDevice);

			if (f2->c > 1 && f2nr > 1) {
				#ifdef INPLACE
				TIMER_START(GREEN("Transposing Second Matrix Inplace..."));
				inplace::transpose(1, (double *)d2d, f2nr, f2->c);
				d2t = d2d;
				#else
				TIMER_START(GREEN("Transposing Second Matrix..."));
				cudaMalloc(&d2t, sizeof(chunk) * f2nr * f2->c);
				dim3 grid2(CEIL(f2nr, BLOCK_DIM), CEIL(f2->c, BLOCK_DIM), 1);
				transpose<<<grid2,threads>>>(d2t, d2d, f2->c, f2nr);
				cudaFree(d2d);
				#endif
				TIMER_STOP;
			} else d2t = d2d;

			cudaMalloc(&d3t, sizeof(chunk) * f3nr * f3.c);
			cudaMalloc(&v3d, sizeof(value) * f3nr);

			TIMER_START(GREEN("Join sum..."));
			joinsumkernel<<<bn, THREADSPERBLOCK>>>(*f1, *f2, f3, d1t, d2t, d3t, v1d, v2d, v3d, f1nr, f2nr, f3nr, pfxh1d, pfxh2d, pfxhpd, bd);
			GPUERRORCHECK;
			cudaFree(d1t);
			cudaFree(d2t);
			cudaFree(v1d);
			cudaFree(v2d);
			TIMER_STOP;

			cudaMemcpy(f3.v + pfxhp[pfxhi[r]], v3d, sizeof(value) * f3nr, cudaMemcpyDeviceToHost);
			cudaFree(v3d);
			cudaFree(bd);

			if (f3.c > 1 && f3nr > 1) {
				#ifdef INPLACE
				TIMER_START(GREEN("Transposing Result Inplace..."));
				inplace::transpose(0, (double *)d3t, f3nr, f3.c);
				d3d = d3t;
				#else
				TIMER_START(GREEN("Transposing Result..."));
				cudaMalloc(&d3d, sizeof(chunk) * f3nr * f3.c);
				dim3 grid3(CEIL(f3nr, BLOCK_DIM), CEIL(f3.c, BLOCK_DIM), 1);
				transposeback<<<grid3,threads>>>(d3d, d3t, f3nr, f3.c);
				cudaFree(d3t);
				#endif
				TIMER_STOP;
			} else d3d = d3t;

			cudaMemcpy(DATA(&f3, pfxhp[pfxhi[r]]), d3d, sizeof(chunk) * f3nr * f3.c, cudaMemcpyDeviceToHost);
			cudaFree(d3d);
		}

		cudaFree(pfxh1d);
		cudaFree(pfxh2d);
		cudaFree(pfxhpd);
		cudaFree(h1d);
		cudaFree(h2d);
		cudaFree(hpd);
		free(pfxho);
		free(pfxhi);
		free(ho);
		free(hi);
		free(bh);

	} else {
		TIMER_START(YELLOW("Join sum..."));
		joinsumhost(f1, f2, &f3, f1->h, hp, pfxh1, pfxh2, pfxhp, hn);
		TIMER_STOP;
		ADDTIME_STOP;
	}

	free(pfxh1);
	free(pfxh2);
	free(pfxhp);
	free(hp);

	#ifdef PRINTDEBUG
	print(&f3, "join sum result", c1);
	#endif

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

	memcpy(f2->data, data2, sizeof(chunk) * f2->c * f2->n);
	memcpy(f2->vars, vars2, sizeof(id) * f2->m);
	memcpy(f2->v, v2, sizeof(value) * f2->n);

	joinsum(f1, f2);

	FREEFUNC(f1);
	FREEFUNC(f2);

	return 0;
}

#endif
