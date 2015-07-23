#ifdef JSCMAIN
#include "jsc.h"
#endif

#ifdef PRINTTIME
static struct timeval t1, t2;
#endif

#ifdef __CUDACC__

__constant__ uint4 bdc[CONSTANTSIZE / sizeof(uint4)];

__global__ void histogramproductkernel(dim *h1, dim *h2, dim *hr, dim hn) {

	dim tid = blockIdx.x * THREADSPERBLOCK + threadIdx.x;
	if (tid < hn) hr[tid] = h1[tid] * h2[tid];
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

	//if (tx < j.x * f1.c) shd[tx] = d1[(tx / j.x) * f1.n + l.x + k.x + tx % j.x];
	//if (tx < j.y * f2.c) shd[j.x * f1.c + tx] = d2[(tx / j.y) * f2.n + l.y + k.y + tx % j.y];
	if (tx < j.x) for (h = 0; h < f1.c; h++) {
		shd[h * j.x + tx] = d1[h * f1.n + l.x + k.x + tx];
		#ifdef DEBUGKERNEL
		printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] shd[% 3u] <- d1[% 3u] = %lu\n",
		       bx, tx, h * j.x + tx, h * f1.n + l.x + k.x + tx, shd[h * j.x + tx]);
		#endif
	}

	if (tx < j.y) for (h = 0; h < f2.c; h++) {
		shd[j.x * f1.c + h * j.y + tx] = d2[h * f2.n + l.y + k.y + tx];
		#ifdef DEBUGKERNEL
		printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] shd[% 3u] <- d2[% 3u] = %lu\n",
		       bx, tx, j.x * f1.c + h * j.y + tx, h * f2.n + l.y + k.y + tx, shd[j.x * f1.c + h * j.y + tx]);
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
		JOINTOPERATION(shv[j.x + j.y + tx], shv[i.x + o.z / i.w], shv[h + o.z % i.w]);
		#ifdef DEBUGKERNEL
		printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] shv[% 3u] = shv[% 3u] + shv[% 3u] = % 2f = % 2f + % 2f\n",
		       bx, tx, j.x + j.y + tx, i.x + o.z / i.w, h + o.z % i.w, shv[j.x + j.y + tx], shv[i.x + o.z / i.w], shv[h + o.z % i.w]);
		#endif
		i = make_uint4(i.x + o.z / i.w, i.y + o.z % i.w, f1.m % BITSPERCHUNK, f2.s % BITSPERCHUNK);
		chunk a, b, c;

		// if i.z = 0 (i.e., if f.m is a multiple of BITSPERCHUNK, I don't have to copy anything from the first table
		chunk t = i.z ? shd[i.x + j.x * (f1.c - 1)] : 0;
		//#ifdef DEBUGKERNEL
		//printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] i = [ .x = % 3u .y = % 3u .z = % 3u .w = % 3u ] t = %lu\n", bx, tx, i.x, i.y, i.z, i.w, t);
		//printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] o = [ .x = % 3u .y = % 3u .z = % 3u ]\n", bx, tx, o.x, o.y, o.z);
		//#endif

		if (i.z || f2.m - f2.s) {
			h = f2.s / BITSPERCHUNK;
			a = shd[i.y + h * j.y];
			for (m = DIVBPC(f1.m); m < f3.c; m++, h++) {
				b = h == f2.c - 1 ? 0 : shd[i.y + (h + 1) * j.y];
				// a = current chunk in d2
				// b = next chunk in d2
				// i.z = MODBPC(f1.m)
				// i.w = MODBPC(f2.s)
				c = a >> i.w | b << BITSPERCHUNK - i.w;
				t = t | c << i.z;
				shd[j.x * f1.c + j.y * f2.c + (h - f2.s / BITSPERCHUNK) * j.z + tx] = t;
				#ifdef DEBUGKERNEL
				printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] d1 = % 3u d2 = % 3u (-% 3u) h = %u shd[% 3u] = %lu\n",
				       bx, tx, i.x, i.y, j.x * f1.c, h, j.x * f1.c + j.y * f2.c + (h - f2.s / BITSPERCHUNK) * j.z + tx, t);
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

		for (h = 0; h < f1.m / BITSPERCHUNK; h++) {
			d3[l.z + o.x + h * f3.n + tx] = shd[i.x + h * j.x];
			#ifdef DEBUGKERNEL
			printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] (2) d3[% 3u] <- shd[% 3u] = %lu\n",
			       bx, tx, l.z + o.x + h * f3.n + tx, i.x + h * j.x, shd[i.x + h * j.x]);
			#endif
		}

		for (; h < f3.c; h++) {
			d3[l.z + o.x + h * f3.n + tx] = shd[j.x * f1.c + j.y * f2.c + (h - f1.m / BITSPERCHUNK) * j.z + tx];
			#ifdef DEBUGKERNEL
			printf("[" YELLOW("% 3u") "," GREEN("% 3u") "] (3) h = % 3u d3[% 3u] <- shd[% 3u] = %lu\n",
			       bx, tx, h, l.z + o.x + h * f3.n + tx, j.x * f1.c + j.y * f2.c + (h - f1.m / BITSPERCHUNK) * j.z + tx,
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

__attribute__((always_inline)) inline
void copyfunc(const func *f1, const func *f2, dim idx) {

	register dim i;

	for (i = 0; i < f1->c; i++)
		memcpy(f1->data + i * f1->n + idx, f2->data + i * f2->n, sizeof(chunk) * f2->n);

	memcpy(f1->v + idx, f2->v, sizeof(value) * f2->n);

	for (i = 0; i < f2->n; i++)
		if (f2->care[i]) {
			f1->care[idx + i] = (chunk *)malloc(sizeof(chunk) * f1->c);
			memcpy(f1->care[idx + i], f2->care[i], sizeof(chunk) * f1->c);
		}
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

__attribute__((always_inline)) inline
void removedefaults(func *f) {

	register dim idx = 0;
	while (!f->v[idx]) idx++;
	//#ifdef PRINTINFO
	printf(MAGENTA("Reduced to %.2f%%\n"), 100.0 * (f->n - idx) / f->n);
	//#endif
	transpose(f->data, f->c, f->n);
	memmove(f->data, f->data + idx * f->c, sizeof(chunk) * f->c * (f->n - idx));
	memmove(f->care, f->care + idx, sizeof(chunk *) * (f->n - idx));
	memmove(f->v, f->v + idx, sizeof(value) * (f->n - idx));
	f->n -= idx;
	f->data = (chunk *)realloc(f->data, sizeof(chunk) * f->n * f->c);
	transpose(f->data, f->n, f->c);
	f->care = (chunk **)realloc(f->care, sizeof(chunk *) * f->n);
	f->v = (value *)realloc(f->v, sizeof(value) * f->n);
}

#define DIFFERSONEBIT(F, I, J, TMP) ({ register int ret = -1; if ((!(F)->care[I] && !(F)->care[J]) || \
				       ((F)->care[I] && (F)->care[J] && !memcmp((F)->care[I], (F)->care[J], sizeof(chunk) * (F)->c))) { \
				       MASKXOR((F)->data + (I) * (F)->c, (F)->data + (J) * (F)->c, TMP, (F)->c); \
				       if (MASKPOPCNT(TMP, (F)->c) == 1) ret = MASKFFS(TMP, (F)->c); } ret; })

__attribute__((always_inline)) inline
void collapsethreshold(func *f) {

	#define THRESHOLD 50

	register dim idx = 0;
	while (f->v[idx] < THRESHOLD) idx++;

	sort(f, idx);
	//print(f, "f sorted");
	f->hn = uniquecombinations(f, idx);
	f->h = (dim *)malloc(sizeof(dim) * f->hn);
	histogram(f, idx);
	//printbuf(f->h, f->hn, "f->h");
	dim pfx[f->hn];
	exclprefixsum(f->h, pfx, f->hn);

	register const dim cn = CEILBPC(f->n);
	register chunk *keep = (chunk *)malloc(sizeof(chunk) * cn);
	ONES(keep, f->n, cn);
	register chunk *tmp = (chunk *)malloc(sizeof(chunk) * f->c);
	transpose(f->data, f->c, f->n);

	register dim h;

	#pragma omp parallel for private(h)
	for (h = 0; h < f->hn; h++) {

		register bool collapsed = true;

		while (collapsed) {

			register const dim n = (h == f->hn - 1) ? f->n : pfx[h + 1];
			register dim i, j;
			register int b;
			collapsed = false;

			for (i = idx + pfx[h]; i < n; i++) if (GET(keep, i))
				for (j = i + 1; j < n; j++) if (GET(keep, j))
					if ((b = DIFFERSONEBIT(f, i, j, tmp)) != -1) {
						//printf("%u matches %u on bit %d\n", i, j, b);
						collapsed = true;
						CLEAR(keep, j);
						CLEAR(f->data + i * f->c, b);
						if (!f->care[i]) { f->care[i] = (chunk *)malloc(sizeof(chunk) * f->c); ONES(f->care[i], f->m, f->c); }
						CLEAR(f->care[i], b);
						if (f->care[j]) { free(f->care[j]); f->care[j] = NULL; }
						break;
					}
			//puts("");
		}
	}

	register const dim pk = MASKPOPCNT(keep, cn);
	printf(MAGENTA("Reduced to %.2f%%\n"), 100.0 * pk / f->n);
	register chunk *data = (chunk *)malloc(sizeof(chunk) * pk * f->c);
	register chunk **care = (chunk **)malloc(sizeof(chunk *) * pk);
	register value *v = (value *)malloc(sizeof(value) * pk);
	register dim i, j;

	for (i = 0, j = MASKFFS(keep, cn); i < pk; i++, j = MASKCLEARANDFFS(keep, j, cn)) {
		memcpy(data + i * f->c, f->data + j * f->c, sizeof(chunk) * f->c);
		care[i] = f->care[j];
		v[i] = f->v[j];
	}

	free(f->data);
	f->data = data;
	free(f->care);
	f->care = care;
	free(f->v);
	f->v = v;
	free(f->h);
	f->n = pk;
	transpose(data, pk, f->c);
	free(keep);
	free(tmp);
	//print(f);
}

#endif

__attribute__((always_inline)) inline
func jointsum(func *f1, func *f2) {

	#ifdef PRINTFUNCTIONCODE
	register id i;

	printf("f1->n = %u;\nf1->m = %u;\n", f1->n, f1->m);
	puts("ALLOCFUNC(f1);");
	printsourcebuf(f1->data, f1->n * f1->c, "data", 1, "chunk");
	printsourcebuf(f1->v, f1->n, "v", 1, "value");
	printsourcebuf(f1->vars, f1->m, "vars", 1, "id");
	printf("chunk *care1[%u] = { 0 };\n", f1->n);

	for (i = 0; i < f1->n; i++) if (f1->care[i]) {
		printsourcebuf(f1->care[i], f1->c, "f1care", i, "chunk");
		printf("care1[%u] = (chunk *)malloc(sizeof(chunk) * f1->c);\n", i);
		printf("memcpy(care1[%u], f1care%u, sizeof(chunk) * f1->c);\n", i, i);
	}

	printf("f2->n = %u;\nf2->m = %u;\n", f2->n, f2->m);
	puts("ALLOCFUNC(f2);");
	printsourcebuf(f2->data, f2->n * f2->c, "data", 2, "chunk");
	printsourcebuf(f2->v, f2->n, "v", 2, "value");
	printsourcebuf(f2->vars, f2->m, "vars", 2, "id");
	printf("chunk *care2[%u] = { 0 };\n", f2->n);

	for (i = 0; i < f2->n; i++) if (f2->care[i]) {
		printsourcebuf(f2->care[i], f2->c, "f2care", i, "chunk");
		printf("care2[%u] = (chunk *)malloc(sizeof(chunk) * f2->c);\n", i);
		printf("memcpy(care2[%u], f2care%u, sizeof(chunk) * f2->c);\n", i, i);
	}

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
	f3.d = f1->d + f2->d;
	f3.m = f1->m + f2->m - f1->s;

	//register const dim cs12 = CEILBPC(MAX(f1->m, f2->m));
	//chunk cc[cs12];
	//ONES(cc, f1->s, cs12);
	//print(f1, "f1", c1);
	//print(f2, "f2", c2);

	TIMER_START(YELLOW("Shift & Reorder..."));
	shared2least(f1, c1);
	shared2least(f2, c2);
	reordershared(f2, f1->vars);
	TIMER_STOP;

	//print(f1, "f1", cc);
	//print(f2, "f2", cc);

	TIMER_START(YELLOW("Sort..."));
	sort(f1);
	sort(f2);
	TIMER_STOP;

	f1->hn = uniquecombinations(f1);
	f2->hn = uniquecombinations(f2);
	f1->h = (dim *)calloc(f1->hn, sizeof(dim));
	f2->h = (dim *)calloc(f2->hn, sizeof(dim));

	histogram(f1);
	histogram(f2);
	//printbuf(f1->h, f1->hn, "f1->h");
	//printbuf(f2->h, f2->hn, "f2->h");

	TIMER_START(YELLOW("Prefix Sum..."));
	register dim *pfxh1, *pfxh2;
	pfxh1 = (dim *)malloc(sizeof(dim) * f1->hn);
	pfxh2 = (dim *)malloc(sizeof(dim) * f2->hn);
	exclprefixsum(f1->h, pfxh1, f1->hn);
	exclprefixsum(f2->h, pfxh2, f2->hn);
	TIMER_STOP;

	//print(f1, "f1", cc);
	//print(f2, "f2", cc);
	TIMER_START(YELLOW("Instancing don't cares..."));
	register func sf1i, sf2i, *f1i = &sf1i, *f2i = &sf2i;
	register func sf1d, sf2d, *f1d = &sf1d, *f2d = &sf2d;
        instancedontcare(f1, f2, f3.m, 0, pfxh1, pfxh2, f1i, f1d);
        instancedontcare(f2, f1, f3.m, f1->m - f1->s, pfxh2, pfxh1, f2i, f2d);
	TIMER_STOP;

	TIMER_START(YELLOW("Sorting..."));
	sort(f1i);
	sort(f2i);
	TIMER_STOP;

	/*print(f1, "f1", cc);
	print(f1i, "f1i", cc);
	print(f1d, "f1d", cc);
	print(f2, "f2", cc);
	print(f2i, "f2i", cc);
	print(f2d, "f2d", cc);*/

	#ifdef __CUDACC__
	*f1 = sf1i;
	*f2 = sf2i;

	f1->hn = intuniquecombinations(f1);
	f2->hn = intuniquecombinations(f2);
	//printf("f1->hn = %u\n", f1->hn);

	#ifdef PRINTINFO
	printf(MAGENTA("%u unique combinations\n"), f1->hn);
	printf(MAGENTA("%u unique combinations\n"), f2->hn);
	#endif

	f1->h = (dim *)calloc(f1->hn, sizeof(dim));
	f2->h = (dim *)calloc(f2->hn, sizeof(dim));

	TIMER_START(YELLOW("Histogram..."));
	inthistogram(f1);
	inthistogram(f2);
	TIMER_STOP;

	//printf("%u %u\n", f1->hn, f2->hn);
	assert(f1->hn == f2->hn);

	pfxh1 = (dim *)realloc(pfxh1, sizeof(dim) * f1->hn);
	pfxh2 = (dim *)realloc(pfxh2, sizeof(dim) * f2->hn);

	exclprefixsum(f1->h, pfxh1, f1->hn);
	exclprefixsum(f2->h, pfxh2, f2->hn);

	TIMER_START(YELLOW("Instancing defaults..."));
	instancedefaults(f1, pfxh1);
	instancedefaults(f2, pfxh2);
	TIMER_STOP;

	//printf("f1 = %u\n", crc32func(f1));
	//printf("f2 = %u\n", crc32func(f2));

	sort(f1);
	sort(f2);

	//print(f1, "f1", cc);
	//print(f2, "f2", cc);
	//BREAKPOINT("with defaults");
	//printbuf(f1->h, f1->hn, "f1->h");
	//printbuf(f2->h, f2->hn, "f2->h");

	exclprefixsum(f1->h, pfxh1, f1->hn);
	exclprefixsum(f2->h, pfxh2, f2->hn);

	register const dim hn = f1->hn;
	value *v1d, *v2d, *v3d;
	chunk *d1d, *d2d, *d3d;
	dim *h1d, *h2d, *hpd, *pfxh1d, *pfxh2d, *pfxhpd;

	if (f1->n && f2->n) {

		#ifdef PRINTSIZE
		printf(RED("Will allocate %zu bytes\n"), sizeof(chunk) * (f1->n * f1->c + f2->n * f2->c) +
							 sizeof(value) * (f1->n + f2->n) + sizeof(dim) * 6 * hn);
		#endif
		TIMER_START(YELLOW("Allocating... "));
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

		TIMER_START(YELLOW("Copying... "));
		cudaMemcpy(d1d, f1->data, sizeof(chunk) * f1->n * f1->c, cudaMemcpyHostToDevice);
		cudaMemcpy(d2d, f2->data, sizeof(chunk) * f2->n * f2->c, cudaMemcpyHostToDevice);
		cudaMemcpy(v1d, f1->v, sizeof(value) * f1->n, cudaMemcpyHostToDevice);
		cudaMemcpy(v2d, f2->v, sizeof(value) * f2->n, cudaMemcpyHostToDevice);
		cudaMemcpy(h1d, f1->h, sizeof(dim) * hn, cudaMemcpyHostToDevice);
		cudaMemcpy(h2d, f2->h, sizeof(dim) * hn, cudaMemcpyHostToDevice);
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

	f3.n += f1d->n + f2d->n;

	#ifdef PRINTSIZE
	printf(RED("Result size = %zu bytes (%u lines)\n"), sizeof(chunk) * f3.n * CEILBPC(f3.m), f3.n);
	#endif

	assert(sizeof(chunk) * (f1->n * f1->c + f2->n * f2->c + f3.n * CEILBPC(f3.m)) +
	       sizeof(value) * (f1->n + f2->n + f3.n) + sizeof(dim) * 6 * hn < GLOBALSIZE);

	ALLOCFUNC(&f3);
	memcpy(f3.vars, f1->vars, sizeof(id) * f1->m);
	memcpy(f3.vars + f1->m, f2->vars + f2->s, sizeof(id) * (f2->m - f1->s));

	if (f1->n && f2->n) {

		cudaMalloc(&d3d, sizeof(chunk) * f3.n * f3.c);
		cudaMemset(d3d, 0, sizeof(chunk) * f3.n * f3.c);
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
		jointsumkernel<<<bn, THREADSPERBLOCK>>>(*f1, *f2, f3, d1d, d2d, d3d, v1d, v2d, v3d, pfxh1d, pfxh2d, pfxhpd, bd);
		TIMER_STOP;
		gpuerrorcheck(cudaPeekAtLastError());
		gpuerrorcheck(cudaDeviceSynchronize());

		cudaMemcpy(f3.data, d3d, sizeof(chunk) * f3.n * f3.c, cudaMemcpyDeviceToHost);
		cudaMemcpy(f3.v, v3d, sizeof(value) * f3.n, cudaMemcpyDeviceToHost);

		exclprefixsum(f1->h, pfxh1, f1->hn);
		exclprefixsum(f2->h, pfxh2, f2->hn);
		exclprefixsum(hp, pfxhp, hn);

		register dim i, j, k;
		// could be parallelised
		for (i = 0; i < hn; i++)
			for (j = 0; j < hp[i]; j++) {
				register const dim pfxj1 = pfxh1[i] + j / f2->h[i];
				register const dim pfxj2 = pfxh2[i] + j % f2->h[i];
				//printf("%u %u\n", pfxj1, pfxj2);
				f3.care[pfxhp[i] + j] = (chunk *)calloc(f3.c, sizeof(chunk));
				if (f1->care[pfxj1]) memcpy(f3.care[pfxhp[i] + j], f1->care[pfxj1], sizeof(chunk) * f1->c);
				else ONES(f3.care[pfxhp[i] + j], f1->m, f1->c);
				for (k = 0; k < f2->m - f2->s; k++)
					if (!f2->care[pfxj2] || GET(f2->care[pfxj2], f2->s + k)) SET(f3.care[pfxhp[i] + j], f1->m + k);
				if (MASKPOPCNT(f3.care[pfxhp[i] + j], f3.c) == f3.m) { free(f3.care[pfxhp[i] + j]); f3.care[pfxhp[i] + j] = NULL; }
			}

		#ifdef PRINTCHECKSUM
		printf("f1i = %u\n", crc32func(f1i));
		printf("f1d = %u\n", crc32func(f1d));
		printf("f2i = %u\n", crc32func(f2i));
		printf("f2d = %u\n", crc32func(f2d));
		printf("f3 = %u\n", crc32func(&f3));
		#endif

		cudaFree(pfxh1d);
		cudaFree(pfxh2d);
		cudaFree(pfxhpd);
		cudaFree(d1d);
		cudaFree(d2d);
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

	copyfunc(&f3, f1d, f3.n - f1d->n - f2d->n);
	copyfunc(&f3, f2d, f3.n - f2d->n);
	sort<true>(&f3);
	//print(&f3, "f3", cc);
	TIMER_START(YELLOW("Removing defaults..."));
	removedefaults(&f3);
	TIMER_STOP;
	//print(&f3, "f3", cc);
	//TIMER_START(YELLOW("Collapsing..."));
	//collapsethreshold(&f3);
	//TIMER_STOP;
	//print(&f3, "f3", cc);
	//BREAKPOINT("Good?");
	#endif
	free(f1->h);
	free(f2->h);
	free(pfxh1);
	free(pfxh2);
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
