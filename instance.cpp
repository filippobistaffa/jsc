__attribute__((always_inline)) inline
void printmask(const chunk *m, dim n) {

	register dim i;
	for (i = 0; i < n; i++) printf(GET(m, i) ? "1" : "0");
	printf("\n");
}

template <typename type>
__attribute__((always_inline)) inline
void printbuf(const type *buf, unsigned n, const char *name) {

	printf("%s = [ ", name);
	while (n--) printf("%u ", *(buf++));
	printf("]\n");
}

template <typename type>
__attribute__((always_inline)) inline
char duplicate(const type *buf, unsigned n) {

	if (n < 2) return 0;
	//printbuf(buf, n, "unsorted");
	#include "../cucop/iqsort.h"
	#define LT(_a, _b) (*(_a)) < (*(_b))
	type tmp[n];
	memcpy(tmp, buf, sizeof(type) * n);
	QSORT(type, tmp, n, LT);
	//printbuf(tmp, n, "sorted");
	n--;
	register type *p = tmp + 1;
	do { if (*p == *(p - 1)) return 1; p++; }
	while (--n);
	return 0;
}

template <bool compute = false> inline
void nestedloop(dim *bits, chunk *masks, dim *idx, const dim *initbits, const chunk *initmasks, const dim *popcnt,
		dim l, dim n, dim s, dim c, dim *j, dim offs = 0, dim hoffs = 0, dim moffs = 0, dim i = 0,
		const func *f1 = NULL, const func *f2 = NULL, const dim *pfxh1 = NULL, const dim *pfxh2 = NULL, const func *fd = NULL, const dim *map = NULL,
		const func *fi = NULL) {

	if (l == n) {
		if (!duplicate(bits, n)) {
			//printbuf(idx, n, "idx");
			//printbuf(bits, n, "bits");
			if (compute) {
				register const dim ds = DIVBPC(f1->s);
				//if (fi) print(fi, "fi before");
				//printf("hoffs = %u j = %u\n", hoffs, *j);
				//fd->h[hoffs + (*j)] = f1->h[i];
				//print(fi, "fi after");

				register dim h;
				for (h = 0; h < f1->h[i]; h++) {
					//puts("computation");
					register dim k, e = offs + (*j) * f1->h[i] + h;
					fd->care[e] = (chunk *)malloc(sizeof(chunk) * fd->c);
					if (moffs) {
						for (k = 0; k < ds; k++) fd->data[k * fd->n + e] = f1->data[k * f1->n + pfxh1[i] + h];
						memcpy(fd->care[e], f1->care[pfxh1[i] + h], sizeof(chunk) * ds);
						if (f1->mask) {
							fd->data[ds * fd->n + e] = f1->data[ds * f1->n + pfxh1[i] + h] & f1->mask;
							fd->care[e][ds] = f1->care[pfxh1[i] + h][ds] & f1->mask;
						}
						for (k = 0; k < f1->m - f1->s; k++) {
							if (GET(f1->data + pfxh1[i] + h, f1->s + k, f1->n))
								fd->data[DIVBPC(f1->s + moffs + k) * fd->n + e] |= 1ULL << MODBPC(f1->s + moffs + k);
							if (GET(f1->care[pfxh1[i] + h], f1->s + k)) SET(fd->care[e], f1->s + moffs + k);
						}
					} else {
						for (k = 0; k < fd->c; k++) fd->data[k * fd->n + e] = f1->data[k * f1->n + pfxh1[i] + h];
						memcpy(fd->care[e], f1->care[pfxh1[i] + h], sizeof(chunk) * fd->c);
					}
					fd->v[e] = f1->v[pfxh1[i] + h] + f2->d;

					for (k = 0; k < n; k++) {
						chunk tmp[c];
						memset(tmp, 0, sizeof(chunk) * c);
						MASKANDNOT(initmasks + k * c, masks + k * c, tmp, c);
						//printf("tmp %u\n", k);
						//printmask(tmp, s);
						MASKOR(fd->care[e], tmp, fd->care[e], c);
						register dim j, b, ntmp = MASKPOPCNT(tmp, c);
						for (j = 0, b = MASKFFS(tmp, c); j < ntmp; j++, b = MASKCLEARANDFFS(tmp, b, c))
							if (GET(f2->data + map[k], b, f2->n)) fd->data[DIVBPC(b) * fd->n + e] |= 1ULL << MODBPC(b);
						SET(fd->care[e], bits[k]);
						if (!GET(f2->data + map[k], bits[k], f2->n)) fd->data[DIVBPC(bits[k]) * fd->n + e] |= 1ULL << MODBPC(bits[k]);
					}

					if (MASKPOPCNT(fd->care[e], fd->c) == fd->m) { free(fd->care[e]); fd->care[e] = NULL; }
				}
			}
			(*j)++;
		}
	} else {
		for (bits[l] = initbits[l], memcpy(masks + l * c, initmasks + l * c, sizeof(chunk) * c), idx[l] = 0;
		     idx[l] < popcnt[l]; idx[l]++, bits[l] = MASKCLEARANDFFS(masks + l * c, bits[l], c)) {
			//printmask(masks + l * c, s);
			nestedloop<compute>(bits, masks, idx, initbits, initmasks, popcnt, l + 1, n, s, c, j, offs, hoffs, moffs, i, f1, f2, pfxh1, pfxh2, fd, map, fi);
		}
	}
}

// DIFFERENCE is true if I minus J is not empty

#define ONESIFNULL(P) ((P) ? (P) : ones)

#define DIFFERENCE(F1, I, F2, J, CS, TMP) ({ register char res = 0; if ((F1)->care[I]) { \
					     MASKNOTAND((F1)->care[I], ONESIFNULL((F2)->care[J]), TMP, CS); \
					     if (MASKPOPCNT(TMP, CS)) res = 1; } /*printf("%u\n", res);*/ res; })

#define ONES(V, I, C) do { register dim _i; register const dim _mi = MODBPC(I); for (_i = 0; _i < (C); _i++) (V)[_i] = ~0ULL; \
			   if (_mi) (V)[(C) - 1] = (1ULL << _mi) - 1; } while (0)

// fi = intersection between f1 and f2
// fd = difference between f1 and f2

template <bool joint  = true>
__attribute__((always_inline)) inline
void instancedontcare(const func *f1, const func *f2, dim f3m, dim moffs, const dim *pfxh1, const dim *pfxh2, func *fi, func *fd) {

	register const dim cs = CEILBPC(f1->s);
	register const dim cn = CEILBPC(f1->hn);
	register const dim ds = DIVBPC(f1->s);
	register const dim ms = MODBPC(f1->s);
	register const dim f1nf2n = f1->hn * f2->hn;

	register dim i, j, tni = 0, tnd = 0, tnhi = 0, tnhd = 0;
	register chunk *tmpmask = (chunk *)calloc(cn, sizeof(chunk));
	register chunk *nodifference = (chunk *)calloc(cn, sizeof(chunk));
	register chunk *nointersection = (chunk *)calloc(cn, sizeof(chunk));
	ONES(nointersection, f1->hn, cn);
	ONES(nodifference, f1->hn, cn);

	register chunk *notandmasks = (chunk *)calloc(f1nf2n * cs, sizeof(chunk));
	register chunk *masks = (chunk *)malloc(sizeof(chunk) * f1nf2n * cs);

	register dim popcnt[f1nf2n];
	register dim bits[f1nf2n];
	register dim init[f1nf2n];
	register dim idx[f1nf2n];
	register dim map[f1nf2n];

	register dim ni[f1->hn];
	register dim pfxni[f1->hn];
	register dim nd[f1->hn];
	register dim pfxnd[f1->hn];

	register dim nhi[f1->hn];
	register dim pfxnhi[f1->hn];
	register dim nhd[f1->hn];
	register dim pfxnhd[f1->hn];
	memset(ni, 0, sizeof(dim) * f1->hn);
	memset(nd, 0, sizeof(dim) * f1->hn);
	memset(nhi, 0, sizeof(dim) * f1->hn);
	memset(nhd, 0, sizeof(dim) * f1->hn);

	register chunk ones[cs];
	ONES(ones, f1->s, cs);

	//printbuf(pfxh1, f1->hn, "pfxh1");
	//printbuf(pfxh2, f2->hn, "pfxh2");

	// TODO: Could be parallelised, with reduction!
	for (i = 0; i < f1->hn; i++) {

		register const dim if2n = i * f2->hn;
		register chunk *const inotandmasks = notandmasks + if2n;
		register chunk *const imasks = masks + if2n;
		register dim *const ipopcnt = popcnt + if2n;
		register dim *const ibits = bits + if2n;
		register dim *const iinit = init + if2n;
		register dim *const iidx = idx + if2n;
		register dim *const imap = map + if2n;
		register chunk tmp[cs];

		for (j = 0; j < f2->hn; j++)
			if (INTERSECT(f1, pfxh1[i], f2, pfxh2[j])) {
				if (!joint && i == j && f1->h[i] == 1) continue;
				CLEAR(nointersection, i);
				if (DIFFERENCE(f1, pfxh1[i], f2, pfxh2[j], cs, tmp)) {
					MASKNOTAND(f1->care[pfxh1[i]], ONESIFNULL(f2->care[pfxh2[j]]), inotandmasks + nhi[i] * cs, cs);
					if (ms) inotandmasks[nhi[i] * cs + cs - 1] &= ones[cs - 1];
					ipopcnt[nhi[i]] = MASKPOPCNT(inotandmasks + nhi[i] * cs, cs);
					iinit[nhi[i]] = MASKFFS(inotandmasks + nhi[i] * cs, cs);
					CLEAR(nodifference, i);
					imap[nhi[i]] = pfxh2[j];
					nhi[i]++;
				}
			}

		if (joint && nhi[i]) {
			//printf("Row %u\n", i);
			//printbuf(init, m, "init");
			//printbuf(popcnt, m, "popcnt");
			//printbuf(imap, nhi[i], "imap");
			nestedloop<false>(ibits, imasks, iidx, iinit, inotandmasks, ipopcnt, 0, nhi[i], f1->s, cs, nhd + i);
		}

		// reduction over these
		tnhi += nhi[i];
		tnhd += nhd[i];
		tni += (ni[i] = nhi[i] * f1->h[i]);
		tnd += (nd[i] = nhd[i] * f1->h[i]);
	}

	MASKANDNOT(nodifference, nointersection, nodifference, cn);

	fd->m = f3m;
	fi->m = f1->m;
	//fi->s = fd->s = f1->s;
	//fi->d = fd->d = f1->d;
	//fi->mask = fd->mask = f1->mask;

	//printmask(nodifference, f1->hn);
	//printmask(nointersection, f1->hn);
	register const dim popnd = MASKPOPCNT(nodifference, cn);
	register const dim popni = MASKPOPCNT(nointersection, cn);
	register dim nnd = 0, nni = 0;

	memcpy(tmpmask, nodifference, sizeof(chunk) * cn);
	for (i = 0, j = MASKFFS(tmpmask, cn); i < popnd; i++, j = MASKCLEARANDFFS(tmpmask, j, cn)) nnd += f1->h[j];
	memcpy(tmpmask, nointersection, sizeof(chunk) * cn);
	for (i = 0, j = MASKFFS(tmpmask, cn); i < popni; i++, j = MASKCLEARANDFFS(tmpmask, j, cn)) nni += f1->h[j];

	fi->n = tni + nnd;
	fd->n = tnd + nni;
	//fi->hn = tnhi + popnd;
	//fd->hn = tnhd + popni;
	/*printf("fi->hn = %u\n", fi->hn);
	printf("fd->hn = %u\n", fd->hn);
	printf("popnd = %u\n", popnd);
	printf("popnin = %u\n", popni);*/
	ALLOCFUNC(fi);
	ALLOCFUNC(fd);
	COPYFIELDS(fi, f1);
	COPYFIELDS(fd, f1);
	memcpy(fi->vars, f1->vars, sizeof(id) * f1->m);
	memset(fd->vars, 0, sizeof(id) * f3m);
	//fi->h = (dim *)malloc(sizeof(dim) * fi->hn);
	//fd->h = (dim *)malloc(sizeof(dim) * fd->hn);

	register dim l, h, k;

	for (i = 0, l = 0, j = MASKFFS(nodifference, cn); i < popnd; i++, l += f1->h[j], j = MASKCLEARANDFFS(nodifference, j, cn)) {
		for (h = 0; h < f1->h[j]; h++) {
			register const dim lh = l + h;
			register const dim pfxh = pfxh1[j] + h;
			for (k = 0; k < f1->c; k++) fi->data[k * fi->n + lh] = f1->data[k * f1->n + pfxh];
			fi->v[lh] = f1->v[pfxh];
			if (f1->care[pfxh]) {
				fi->care[lh] = (chunk *)malloc(sizeof(chunk) * f1->c);
				memcpy(fi->care[lh], f1->care[pfxh], sizeof(chunk) * f1->c);
			}
		}
		//fi->h[i] = f1->h[j];
	}

	for (i = 0, l = 0, j = MASKFFS(nointersection, cn); i < popni; i++, l += f1->h[j], j = MASKCLEARANDFFS(nointersection, j, cn)) {
		for (h = 0; h < f1->h[j]; h++) {
			register const dim lh = l + h;
			register const dim pfxh = pfxh1[j] + h;
			fd->v[lh] = f1->v[pfxh];

			if (moffs) {
				for (k = 0; k < ds; k++) fd->data[k * fd->n + lh] = f1->data[k * f1->n + pfxh];
				fd->care[lh] = (chunk *)malloc(sizeof(chunk) * fd->c);
				memcpy(fd->care[lh], ONESIFNULL(f1->care[pfxh]), sizeof(chunk) * ds);
				if (f1->mask) {
					fd->data[ds * fd->n + lh] = f1->data[ds * f1->n + pfxh] & f1->mask;
					fd->care[lh][ds] = ONESIFNULL(f1->care[pfxh])[ds] & f1->mask;
				}
				for (k = 0; k < f1->m - f1->s; k++) {
					if (GET(f1->data + pfxh, f1->s + k, f1->n))
						fd->data[DIVBPC(f1->s + moffs + k) * fd->n + lh] |= 1ULL << MODBPC(f1->s + moffs + k);
					if (!f1->care[pfxh] || GET(f1->care[pfxh], f1->s + k)) SET(fd->care[lh], f1->s + moffs + k);
				}
			} else {
				for (k = 0; k < fd->c; k++) fd->data[k * fd->n + lh] = f1->data[k * f1->n + pfxh];
				fd->care[lh] = (chunk *)malloc(sizeof(chunk) * fd->c);
				if (f1->care[pfxh]) memcpy(fd->care[lh], f1->care[pfxh], sizeof(chunk) * fd->c);
				else ONES(fd->care[lh], f1->m, fd->c);
			}
			if (MASKPOPCNT(fd->care[lh], fd->c) == fd->m) { free(fd->care[lh]); fd->care[lh] = NULL; }
		}
		//fd->h[i] = f1->h[j];
	}

	exclprefixsum(ni, pfxni, f1->hn);
	exclprefixsum(nd, pfxnd, f1->hn);
	exclprefixsum(nhi, pfxnhi, f1->hn);
	exclprefixsum(nhd, pfxnhd, f1->hn);
	/*printbuf(nhi, f1->hn, "nhi");
	printbuf(nhd, f1->hn, "nhd");
	printbuf(pfxnhi, f1->hn, "pfxnhi");
	printbuf(pfxnhd, f1->hn, "pfxnhd");
	printbuf(ni, f1->hn, "ni");
	printbuf(nd, f1->hn, "nd");
	printbuf(pfxni, f1->hn, "pfxni");
	printbuf(pfxnd, f1->hn, "pfxnd");*/

	// TODO: Could be parallelised
	for (i = 0, j = 0; i < f1->hn; i++, j = 0) {

		if (nhi[i]) {

			register dim a, b;
			register const dim if2n = i * f2->hn;
			register chunk *const inotandmasks = notandmasks + if2n;
			register chunk *const imasks = masks + if2n;
			register dim *const ipopcnt = popcnt + if2n;
			register dim *const ibits = bits + if2n;
			register dim *const iinit = init + if2n;
			register dim *const iidx = idx + if2n;
			register dim *const imap = map + if2n;

			if (joint) nestedloop<true>(ibits, imasks, iidx, iinit, inotandmasks, ipopcnt, 0, nhi[i], f1->s, cs, &j,
						    nni + pfxnd[i], popni + pfxnhd[i], moffs, i, f1, f2, pfxh1, pfxh2, fd, imap, fi);

			for (a = 0, j = nnd + pfxni[i]; a < nhi[i]; a++, j += f1->h[i]) {

				//fi->h[popnd + pfxnhi[i] + a] = f1->h[i];
				register const dim im = imap[a];
				register dim h, k;

				for (b = 0; b < f1->h[i]; b++) {

					register const dim jb = j + b;
					register const dim pfxb = pfxh1[i] + b;
					fi->v[jb] = f1->v[pfxb];
					for (k = 0; k < f1->c; k++) fi->data[k * fi->n + jb] = f1->data[k * f1->n + pfxb];

					fi->care[jb] = (chunk *)malloc(sizeof(chunk) * fi->c);
					memcpy(fi->care[jb], f1->care[pfxb], sizeof(chunk) * fi->c);
					MASKOR(ONESIFNULL(f1->care[pfxb]), ONESIFNULL(f2->care[im]), fi->care[jb], ds);

					if (ms) {
						fi->care[jb][ds] &= ~f1->mask;
						fi->care[jb][ds] |= (ONESIFNULL(f1->care[pfxb])[ds] | ONESIFNULL(f2->care[im])[ds]) & f1->mask;
					}

					if (MASKPOPCNT(fi->care[jb], fi->c) == fi->m) { free(fi->care[jb]); fi->care[jb] = NULL; }

					chunk tmp[cs];
					memset(tmp, 0, sizeof(chunk) * cs);
					MASKANDNOT(ONESIFNULL(f2->care[im]), ONESIFNULL(f1->care[pfxb]), tmp, cs);
					//printmask(tmp, f1->s);
					if (ms) tmp[ds] &= f1->mask;
					register const dim poptmp = MASKPOPCNT(tmp, cs);
					//printf("poptmp = %u\n", poptmp);

					for (h = 0, k = MASKFFS(tmp, cs); h < poptmp; h++, k = MASKCLEARANDFFS(tmp, k, cs)) {
						if GET(f2->data + im, k, f2->hn) fi->data[DIVBPC(k) * fi->n + jb] |= 1ULL << MODBPC(k);
						//printf("k = %u\n", k);
					}
				}
			}
		}
	}

	//printbuf(fi->h, fi->hn, "fi->h");
	//printbuf(fd->h, fd->hn, "fd->h");
	free(nointersection);
	free(nodifference);
	free(notandmasks);
	free(tmpmask);
	free(masks);
}
