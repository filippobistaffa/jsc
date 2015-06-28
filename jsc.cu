#ifdef PRINTTIME
static struct timeval t1, t2;
#endif

#ifdef __CUDACC__

__constant__ uint4 bdc[CONSTANTSIZE / sizeof(uint4)];

__global__ void histogramproductkernel(dim *h1, dim *h2, dim *hr, dim hn) {

	dim tid = blockIdx.x * THREADSPERBLOCK + threadIdx.x;
	if (tid < hn) hr[tid] = h1[tid] * h2[tid];
}

template <bool care = false>
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
void copyfunc(func f1, func f2, dim idx) {

	register dim i;

	for (i = 0; i < f1.c; i++)
		memcpy(f1.data + i * f1.n + idx, f2.data + i * f2.n, sizeof(chunk) * f2.n);

	memcpy(f1.v + idx, f2.v, sizeof(value) * f2.n);
}

template <char mode = 0>
__attribute__((always_inline)) inline
func jointsum(func *f1, func *f2) {

	#ifdef PRINTFUNCTIONCODE
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

	if (mode == 2) {
		instancedontcare(f1, c1);
		instancedontcare(f1, c2);
	}

	f1->mask = f2->mask = f3.mask = (1ULL << (f1->s % BITSPERCHUNK)) - 1;
	#ifdef PRINTINFO
	printf(MAGENTA("%u shared variables\n"), f1->s);
	#endif
	//if (!f1->s) return 1;
	f3.s = f1->s;

	TIMER_START(YELLOW("Shift & Reorder..."));
	shared2least(*f1, c1);
	shared2least(*f2, c2);
	reordershared(*f2, f1->vars);
	TIMER_STOP;

	TIMER_START(YELLOW("Sort..."));
	sort(*f1);
	sort(*f2);
	TIMER_STOP;

	f1->hn = uniquecombinations(*f1);
	f2->hn = uniquecombinations(*f2);
	#ifdef PRINTINFO
	printf(MAGENTA("%u unique combinations\n"), f1->hn);
	printf(MAGENTA("%u unique combinations\n"), f2->hn);
	#endif
	f1->h = (dim *)calloc(f1->hn, sizeof(dim));
	f2->h = (dim *)calloc(f2->hn, sizeof(dim));

	TIMER_START(YELLOW("Histogram..."));
	histogram(*f1);
	histogram(*f2);
	TIMER_STOP;

	TIMER_START(YELLOW("Matching Rows..."));
	f1->hmask = (chunk *)calloc(CEIL(f1->hn, BITSPERCHUNK), sizeof(chunk));
	f2->hmask = (chunk *)calloc(CEIL(f2->hn, BITSPERCHUNK), sizeof(chunk));
	dim n1, n2, hn;
	markmatchingrows(*f1, *f2, &n1, &n2, &hn);
	func fn1, fn2;

	if (mode == 1) {
		f3.d = f1->d + f2->d;
		fn1.m = f1->m;
		fn2.m = f2->m;
		fn1.s = f1->s;
		fn2.s = f2->s;
		fn1.n = f1->n - n1;
		fn2.n = f2->n - n2;
		fn1.mask = f1->mask;
		fn2.mask = f2->mask;
		fn1.hn = f1->hn - hn;
		fn2.hn = f2->hn - hn;
		fn1.h = (dim *)malloc(sizeof(dim) * fn1.hn);
		fn2.h = (dim *)malloc(sizeof(dim) * fn2.hn);
		ALLOCFUNC(fn1, chunk, id, value);
		ALLOCFUNC(fn2, chunk, id, value);
		memcpy(fn1.vars, f1->vars, sizeof(id) * f1->m);
		memcpy(fn2.vars, f2->vars, sizeof(id) * f2->m);
		copymatchingrows<true>(f1, f2, n1, n2, hn, &fn1, &fn2);
		if (fn1.n) { puts("\nNon matching 1"); print(fn1); }
		if (fn2.n) { puts("\nNon matching 2"); print(fn2); }
	} else copymatchingrows(f1, f2, n1, n2, hn);
	TIMER_STOP;

	#ifdef PRINTINFO
	printf(MAGENTA("%u matching rows\n"), f1->n);
	#endif
	#ifdef PRINTTABLES
	print(*f1);
	#endif
	#ifdef PRINTINFO
	printf(MAGENTA("%u matching rows\n"), f2->n);
	#endif
	#ifdef PRINTTABLES
	print(*f2);
	#endif

	f3.m = f1->m + f2->m - f1->s;
	value *v1d, *v2d, *v3d;
	chunk *d1d, *d2d, *d3d, **c1d, **c2d, **c3d;
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
		if (mode == 2) {
			cudaMalloc(&c1d, sizeof(chunk *) * f1->n);
			cudaMalloc(&c2d, sizeof(chunk *) * f2->n);
		}
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

	//if (mode == 1) f3.n += (n1 = fn1.n * (1ULL << (f2->m - f2->s))) + (n2 = fn2.n * (1ULL << (f1->m - f1->s)));

	#ifdef PRINTSIZE
	printf(RED("Result size = %zu bytes (%u lines)\n"), sizeof(chunk) * f3.n * CEIL(f3.m, BITSPERCHUNK), f3.n);
	#endif

	assert(sizeof(chunk) * (f1->n * f1->c + f2->n * f2->c + f3.n * CEIL(f3.m, BITSPERCHUNK)) +
	       sizeof(value) * (f1->n + f2->n + f3.n) + sizeof(dim) * 6 * hn < GLOBALSIZE);

	ALLOCFUNC(f3, chunk, id, value);
	memcpy(f3.vars, f1->vars, sizeof(id) * f1->m);
	memcpy(f3.vars + f1->m, f2->vars + f2->s, sizeof(id) * (f2->m - f1->s));

	if (f1->n && f2->n) {

		cudaMalloc(&d3d, sizeof(chunk) * f3.n * f3.c);
		cudaMemset(d3d, 0, sizeof(chunk) * f3.n * f3.c);
		cudaMalloc(&v3d, sizeof(value) * f3.n);
		if (mode == 2) cudaMalloc(&c3d, sizeof(chunk *) * f3.n);

		dim hp[hn], bn;
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
		dim j;
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

		cudaFree(pfxh1d);
		cudaFree(pfxh2d);
		cudaFree(pfxhpd);
		cudaFree(d1d);
		cudaFree(d2d);
		cudaFree(d3d);
		cudaFree(v1d);
		cudaFree(v2d);
		cudaFree(v3d);
		if (mode == 2) {
			cudaFree(c1d);
			cudaFree(c2d);
			cudaFree(c3d);
		}
		cudaFree(h1d);
		cudaFree(h2d);
		cudaFree(hpd);
		cudaFree(bd);
		free(bh);
	}

	if (mode == 1) {

		register dim i, k, g, h;

		if (fn1.n) {

			register func fa2;
			fa2.m = f2->m;
			fa2.n = fn1.hn;
			ALLOCFUNC(fa2, chunk, id, value);
			memcpy(fa2.vars, f2->vars, sizeof(id) * f2->m);
			g = h = 0;

			// g = current line in fn1.data and fn1.v
			// h = current line in fn1.h

			for (i = 0; i < fn1.hn; i++) {

				for (k = 0; k < DIVBPC(f2->s); k++)
					fa2.data[k * fa2.n + i] = fn1.data[k * fn1.n + g];

				if (fn1.mask) fa2.data[k * fa2.n + i] = fn1.data[k * fn1.n + g] & fn1.mask;

				fa2.care[i] = (chunk *)calloc(fa2.c, sizeof(chunk));
				memset(fa2.care[i], 0xFF, sizeof(chunk) * DIVBPC(fn1.s));
				if (fn1.mask) fa2.care[i][DIVBPC(fn1.s)] = fn1.mask;
				fa2.v[i] = f2->d;
				g += fn1.h[h++];
			}

			puts("fa2");
			print(fa2);
			//func fn1fa2 = jointsum<2>(&fn1, &fa2);
			//puts("fn1fa2");
			//print(fn1fa2);
			FREEFUNC(fn1);
			FREEFUNC(fa2);
			//copyfunc(f3, fn1fa2, f3.n - n1 - n2);
			//FREEFUNC(fn1fa2);
		}

		if (fn2.n) {

			register func fa1;
			fa1.m = f1->m;
			fa1.n = fn2.hn;
			ALLOCFUNC(fa1, chunk, id, value);
			memcpy(fa1.vars, f1->vars, sizeof(id) * f1->m);
			g = h = 0;

			for (i = 0; i < fn2.hn; i++) {

				for (k = 0; k < DIVBPC(f1->s); k++)
					fa1.data[k * fa1.n + i] = fn2.data[k * fn2.n + g];

				if (fn2.mask) fa1.data[k * fa1.n + i] = fn2.data[k * fn2.n + g] & fn2.mask;

				fa1.care[i] = (chunk *)calloc(fa1.c, sizeof(chunk));
				memset(fa1.care[i], 0xFF, sizeof(chunk) * DIVBPC(fn2.s));
				if (fn2.mask) fa1.care[i][DIVBPC(fn2.s)] = fn2.mask;
				fa1.v[i] = f1->d;
				g += fn2.h[h++];
			}

			puts("fa1");
			print(fa1);
			func fn2fa1 = jointsum<2>(&fn2, &fa1);
			puts("fn2fa1");
			print(fn2fa1);
			FREEFUNC(fn2);
			FREEFUNC(fa1);
			//copyfunc(f3, fn2fa1, f3.n - n2);
			//FREEFUNC(fn2fa1);
		}
	}

	free(f1->hmask);
	free(f2->hmask);
	free(f1->h);
	free(f2->h);
	free(c1);
	free(c2);

	return f3;
}

#endif

#ifdef JSCMAIN

#include "jsc.h"

int main(int argc, char *argv[]) {

	/*func f1, f2, f3;

	#include "functions.i"

	ALLOCFUNC(f1, chunk, id, value);
	ALLOCFUNC(f2, chunk, id, value);

	memcpy(f1.data, data1, sizeof(chunk) * f1.c * f1.n);
	memcpy(f1.vars, vars1, sizeof(id) * f1.m);
	memcpy(f1.v, v1, sizeof(value) * f1.n);
	memcpy(f2.data, data2, sizeof(chunk) * f2.c * f2.n);
	memcpy(f2.vars, vars2, sizeof(id) * f2.m);
	memcpy(f2.v, v2, sizeof(value) * f2.n);*/

	func f1, f2;
	f1.m = 25;
	f2.m = 22;
	f1.n = f2.n = 2;

	ALLOCFUNC(f1, true);
	ALLOCFUNC(f2, true);

	id vars1[] = { 77,84,83,59,58,55,54,82,86,85,91,90,89,88,80,79,57,56,38,37,76,75,87,78,92 };
	id vars2[] = { 79,86,23,26,25,24,80,82,83,84,88,89,92,87,90,91,75,76,81,77,78,85 };
	memcpy(f1.vars, vars1, sizeof(id) * f1.m);
	memcpy(f2.vars, vars2, sizeof(id) * f2.m);

	f1.v[0] = 1300000;
	f1.v[1] = 2;
	f2.v[0] = 400000;
	f2.v[1] = 1;

	f1.care[0] = (chunk *)malloc(sizeof(chunk) * f1.c);
	f1.care[1] = (chunk *)malloc(sizeof(chunk) * f1.c);
	f2.care[0] = (chunk *)malloc(sizeof(chunk) * f2.c);
	f2.care[1] = (chunk *)malloc(sizeof(chunk) * f2.c);
	f1.care[0][0] = 1ULL << 0 | 1ULL << 2 | 1ULL << 4 | 1ULL << 6 | 1ULL << 7 | 1ULL << 9 | 1ULL << 11 | 1ULL << 13 | 1ULL << 15 | 1ULL << 17 | 1ULL << 19 | 1ULL << 21 | 1ULL << 22 | 1ULL << 24;
	f1.care[1][0] = f1.data[1] = 1ULL << 24;
	f2.care[0][0] = ((1ULL << 4) - 1) << 2;
	f2.care[1][0] = ((1ULL << 18) - 1) << 2;
	f2.data[1] = 1ULL << 2 | 1ULL << 5 | 1ULL << 11 | 1ULL << 13 | 1ULL << 16 | 1ULL << 18 | 1ULL << 7;

	print(f1);
	print(f2);

	#ifdef __CUDACC__
	jointsum(&f1, &f2, &f3);
	#endif

	FREEFUNC(f1, true);
	FREEFUNC(f2, true);

	return 0;
}

#endif
