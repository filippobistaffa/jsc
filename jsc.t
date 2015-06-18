#ifdef PRINTTIME
static struct timeval t1, t2;
#endif

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

template <bool def = 0>
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

	if (def) {
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
		copymatchingrows<def>(f1, f2, n1, n2, hn, &fn1, &fn2);
		//if (fn1.n) { puts("Non matching 1"); print(fn1); }
		//if (fn2.n) { puts("Non matching 2"); print(fn2); }
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
	chunk *d1d, *d2d, *d3d;
	value *v1d, *v2d, *v3d;
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

	if (def) f3.n += (n1 = fn1.n * (1ULL << (f2->m - f2->s))) + (n2 = fn2.n * (1ULL << (f1->m - f1->s)));

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
		cudaFree(h1d);
		cudaFree(h2d);
		cudaFree(hpd);
		cudaFree(bd);
		free(bh);
	}

	if (def) {

		register dim i, j, k, g, h, l;

		if (fn1.n) {

			register func fa2;
			fa2.m = f2->m;
			fa2.n = fn1.hn * (1ULL << (f2->m - f2->s));
			ALLOCFUNC(fa2, chunk, id, value);
			memcpy(fa2.vars, f2->vars, sizeof(id) * f2->m);
			g = h = l = 0;

			// g = current line in fn1.data and fn1.v
			// h = current line in fn1.h
			// l = current line in fa2.data and fa2.v

			for (i = 0; i < fn1.hn; i++) {
				for (j = 0; j < (1ULL << (f2->m - f2->s)); j++) {
					for (k = 0; k < DIVBPC(f2->s); k++)
						fa2.data[k * fa2.n + l] = fn1.data[k * fn1.n + g];
					if (fn1.mask) fa2.data[k * fa2.n + l] = fn1.data[k * fn1.n + g] & fn1.mask;

					for (k = 0; k < f2->m - f2->s; k++) if ((j >> k) & 1)
						fa2.data[DIVBPC(f2->s + k) * fa2.n + l] |= 1ULL << MODBPC(f2->s + k);

					fa2.v[l++] = f2->d;
				}
				g += fn1.h[h++];
			}

			//puts("fa2");
			//print(fa2);
			func fn1fa2 = jointsum(&fn1, &fa2);
			//puts("fn1fa2");
			//print(fn1fa2);
			FREEFUNC(fn1);
			FREEFUNC(fa2);
			copyfunc(f3, fn1fa2, f3.n - n1 - n2);
			FREEFUNC(fn1fa2);
		}

		if (fn2.n) {

			register func fa1;
			fa1.m = f1->m;
			fa1.n = fn2.hn * (1ULL << (f1->m - f1->s));
			ALLOCFUNC(fa1, chunk, id, value);
			memcpy(fa1.vars, f1->vars, sizeof(id) * f1->m);
			g = h = l = 0;

			for (i = 0; i < fn2.hn; i++) {
				for (j = 0; j < (1ULL << (f1->m - f1->s)); j++) {
					for (k = 0; k < DIVBPC(f1->s); k++)
						fa1.data[k * fa1.n + l] = fn2.data[k * fn2.n + g];
					if (fn2.mask) fa1.data[k * fa1.n + l] = fn2.data[k * fn2.n + g] & fn2.mask;

					for (k = 0; k < f1->m - f1->s; k++) if ((j >> k) & 1)
						fa1.data[DIVBPC(f1->s + k) * fa1.n + l] |= 1ULL << MODBPC(f1->s + k);

					fa1.v[l++] = f1->d;
				}
				g += fn2.h[h++];
			}

			//puts("fa1");
			//print(fa1);
			func fn2fa1 = jointsum(&fn2, &fa1);
			//puts("fn2fa1");
			//print(fn2fa1);
			FREEFUNC(fn2);
			FREEFUNC(fa1);
			copyfunc(f3, fn2fa1, f3.n - n2);
			FREEFUNC(fn2fa1);
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
