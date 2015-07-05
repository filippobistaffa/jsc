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

template <typename type>
__attribute__((always_inline)) inline
void exclprefixsum(type *hi, type *ho, unsigned hn) {

        register unsigned i;
        ho[0] = 0;
        for (i = 1; i < hn; i++) ho[i] = hi[i - 1] + ho[i - 1];
}

template <bool compute = false> inline
void nestedloop(dim *bits, chunk *masks, dim *idx, const dim *initbits, const chunk *initmasks, const dim *popcnt,
		dim l, dim n, dim s, dim c, dim *j, dim offs = 0, dim i = 0,
		const func *f1 = NULL, const func *f2 = NULL, const dim *pfxh1 = NULL, const dim *pfxh2 = NULL, func *fd = NULL, const dim *map = NULL) {

	if (l == n) {
		if (!duplicate(bits, n)) {
			//printbuf(idx, n, "idx");
			//printbuf(bits, n, "bits");
			if (compute) {
				register dim h;
				for (h = 0; h < f1->h[i]; h++) {
					//puts("computation");
					register dim k, e = offs + (*j) * f1->h[i] + h;
					for (k = 0; k < fd->c; k++) fd->data[k * fd->n + e] = f1->data[k * f1->n + pfxh1[i] + h];
					fd->care[e] = (chunk *)malloc(sizeof(chunk) * fd->c);
					memcpy(fd->care[e], f1->care[pfxh1[i] + h], sizeof(chunk) * fd->c);
					fd->v[e] = f1->v[pfxh1[i] + h];

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
			nestedloop<compute>(bits, masks, idx, initbits, initmasks, popcnt, l + 1, n, s, c, j, offs, i, f1, f2, pfxh1, pfxh2, fd, map);
		}
	}
}

#define INTERSECT(F1, I, F2, J) ({ register dim _i; register char cmp = 0; register chunk mask;\
				   register chunk * const ca = (F1)->care[I]; register chunk * const cb = (F2)->care[J]; \
                                   if (!ca && !cb) cmp = DATACOMPARE((F1)->data + (I), (F2)->data + (J), F1, F2); \
                                   else { for (_i = 0; _i < DIVBPC((F1)->s); _i++) { \
                                   mask = (F1)->mask & ((ca && cb) ? ca[_i] & cb[_i] : (ca ? ca[_i] : cb[_i])); \
                                   if ((cmp = CMP((F1)->data[_i * (F1)->n + (I)] & mask, (F2)->data[_i * (F2)->n + (J)] & mask))) break; } \
                                   if (!cmp) { if ((F1)->mask) mask = (F1)->mask & ((ca && cb) ? ca[_i] & cb[_i] : (ca ? ca[_i] : cb[_i])); \
                                   cmp = ((F1)->mask ? CMP(mask & (F1)->data[(DIVBPC((F1)->s)) * (F1)->n + (I)], \
                                                           mask & (F2)->data[(DIVBPC((F1)->s)) * (F2)->n + (J)]) : 0); } } !cmp; })

// DIFFERENCE is true if I minus J is not empty

#define DIFFERENCE(F1, I, F2, J, CS, ONES) ({ register char res = 0; if ((F1)->care[I]) { chunk tmp[CS]; \
                                              MASKNOTAND((F1)->care[I], (F2)->care[J] ? (F2)->care[J] : ONES, tmp, CS); \
                                              if (MASKPOPCNT(tmp, CS)) res = 1; } res; })

#define ONES(V, I, C) do { register dim _i; register const dim _mi = MODBPC(I); for (_i = 0; _i < (C); _i++) (V)[_i] = ~0ULL; \
			   if (_mi) (V)[(C) - 1] = (1ULL << _mi) - 1; } while (0)

// fi = intersection between f1 and f2
// fd = difference between f1 and f2

__attribute__((always_inline)) inline
void instancedontcare(const func *f1, const func *f2, const dim *pfxh1, const dim *pfxh2, func *fi, func *fd) {

	register dim i, j, tni = 0, tnd = 0;
	register const dim cs = CEILBPC(f1->s);
	register const dim cn = CEILBPC(f1->hn);
	register const dim ds = DIVBPC(f1->s);
	register const dim ms = MODBPC(f1->s);
	register const dim f1nf2n = f1->hn * f2->hn;

	register chunk *tmpmask = (chunk *)calloc(cn, sizeof(chunk));
	register chunk *nodifference = (chunk *)calloc(cn, sizeof(chunk));
	register chunk *nointersection = (chunk *)calloc(cn, sizeof(chunk));
	ONES(nointersection, f1->hn, cn);

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
	register dim nhd[f1->hn];
	memset(ni, 0, sizeof(dim) * f1->hn);
	memset(nd, 0, sizeof(dim) * f1->hn);
	memset(nhi, 0, sizeof(dim) * f1->hn);
	memset(nhd, 0, sizeof(dim) * f1->hn);

	register chunk ones[cs];
	ONES(ones, f1->s, cs);

	// TODO: Could be parallelised, reduction over t[ and tnd
	for (i = 0; i < f1->hn; i++) {

		register const dim if2n = i * f2->hn;
		register chunk *const inotandmasks = notandmasks + if2n;
		register chunk *const imasks = masks + if2n;
		register dim *const ipopcnt = popcnt + if2n;
		register dim *const ibits = bits + if2n;
		register dim *const iinit = init + if2n;
		register dim *const iidx = idx + if2n;
		register dim *const imap = map + if2n;

		for (j = 0; j < f2->hn; j++)
			if (INTERSECT(f1, pfxh1[i], f2, pfxh2[j])) {
				CLEAR(nointersection, i);
				if (DIFFERENCE(f1, pfxh2[j], f2, pfxh2[j], cs, ones)) {
					MASKNOTAND(f1->care[pfxh2[j]], f2->care[pfxh2[j]] ? f2->care[pfxh2[j]] : ones, inotandmasks + nhi[i] * cs, cs);
					if (ms) inotandmasks[nhi[i] * cs + cs - 1] &= ones[cs - 1];
					ipopcnt[nhi[i]] = MASKPOPCNT(inotandmasks + nhi[i] * cs, cs);
					iinit[nhi[i]] = MASKFFS(inotandmasks + nhi[i] * cs, cs);
					CLEAR(nodifference, i);
					imap[nhi[i]] = pfxh2[j];
					nhi[i]++;
				}
			}

		if (nhi[i]) {
			//printf("Row %u\n", i);
			//printbuf(init, m, "init");
			//printbuf(popcnt, m, "popcnt");
			//printbuf(imap, nhi[i], "imap");
			nestedloop<false>(ibits, imasks, iidx, iinit, inotandmasks, ipopcnt, 0, nhi[i], f1->s, cs, nhd + i);
		}

		tni += (ni[i] = nhi[i] * f1->h[i]);
		tnd += (nd[i] = nhd[i] * f1->h[i]);
	}

	fi->m = fd->m = f1->m;
	fi->s = fd->s = f1->s;
	fi->d = fd->d = f1->d;
	fi->mask = fd->mask = f1->mask;

	printmask(nodifference, f1->hn);
	printmask(nointersection, f1->hn);
	register const dim popnd = MASKPOPCNT(nodifference, cn);
	register const dim popni = MASKPOPCNT(nointersection, cn);
	register dim nnd = 0, nni = 0;

	memcpy(tmpmask, nodifference, sizeof(chunk) * cn);
	for (i = 0, j = MASKFFS(tmpmask, cn); i < popnd; i++, j = MASKCLEARANDFFS(tmpmask, j, cn)) nnd += f1->h[j];
	memcpy(tmpmask, nointersection, sizeof(chunk) * cn);
	for (i = 0, j = MASKFFS(tmpmask, cn); i < popni; i++, j = MASKCLEARANDFFS(tmpmask, j, cn)) nni += f1->h[j];

	fi->n = tni + nnd;
	fd->n = tnd + nni;
	ALLOCFUNC(fi);
	ALLOCFUNC(fd);
	memcpy(fi->vars, f1->vars, sizeof(id) * f1->m);
	memcpy(fd->vars, f1->vars, sizeof(id) * f1->m);

	register dim l, h, k;

	for (i = 0, l = 0, j = MASKFFS(nodifference, cn); i < popnd; i++, l += f1->h[j], j = MASKCLEARANDFFS(nodifference, j, cn)) {
		for (h = 0; h < f1->h[j]; h++) {
			register const dim pfxl = pfxh1[j] + l;
			for (k = 0; k < f1->c; k++) fi->data[k * fi->n + l] = f1->data[k * f1->hn + pfxl];
			fi->v[i] = f1->v[pfxl];
			if (f1->care[pfxl]) {
				fi->care[l] = (chunk *)malloc(sizeof(chunk) * f1->c);
				memcpy(fi->care[l], f1->care[pfxl], sizeof(chunk) * f1->c);
			}
		}
	}

	for (i = 0, l = 0, j = MASKFFS(nointersection, cn); i < popni; i++, l += f1->h[j], j = MASKCLEARANDFFS(nointersection, j, cn)) {
		for (h = 0; h < f1->h[j]; h++) {
			register const dim pfxl = pfxh1[j] + l;
			for (k = 0; k < f1->c; k++) fd->data[k * fd->n + l] = f1->data[k * f1->hn + pfxl];
			fd->v[i] = f1->v[pfxl];
			if (f1->care[pfxl]) {
				fd->care[l] = (chunk *)malloc(sizeof(chunk) * f1->c);
				memcpy(fd->care[l], f1->care[pfxl], sizeof(chunk) * f1->c);
			}
		}
	}

	exclprefixsum(ni, pfxni, f1->hn);
	exclprefixsum(nd, pfxnd, f1->hn);
	printbuf(nhi, f1->hn, "nhi");
	printbuf(nhd, f1->hn, "nhd");
	printbuf(ni, f1->hn, "ni");
	printbuf(nd, f1->hn, "nd");
	printbuf(pfxni, f1->hn, "pfxni");
	printbuf(pfxnd, f1->hn, "pfxnd");

	// TODO: Could be parallelised
	for (i = 0, j = 0; i < f1->hn; i++) {

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

			nestedloop<true>(ibits, imasks, iidx, iinit, inotandmasks, ipopcnt, 0, nhi[i], f1->s, cs, &j, nni + pfxnd[i], i, f1, f2, pfxh1, pfxh2, fd, imap);

			for (a = 0, j = nnd + pfxni[i]; a < nhi[i]; a++, j += f1->h[i]) {

				register const dim im = imap[a];
				register dim h, k;

				for (b = 0; b < f1->h[i]; b++) {

					register const dim jb = j + b;
					register const dim pfxb = pfxh1[i] + b;
					fi->v[jb] = f1->v[pfxb];
					for (k = 0; k < f1->c; k++) fi->data[k * fi->n + jb] = f1->data[k * f1->n + pfxb];

					fi->care[jb] = (chunk *)malloc(sizeof(chunk) * fi->c);
					memcpy(fi->care[jb], f1->care[pfxb], sizeof(chunk) * fi->c);
					MASKOR(f1->care[pfxb] ? f1->care[pfxb] : ones, f2->care[im] ? f2->care[im] : ones, fi->care[jb], ds);

					if (ms) {
						fi->care[jb][ds] &= ~f1->mask;
						fi->care[jb][ds] |= (f1->care[pfxb][ds] | f2->care[im][ds]) & f1->mask;
					}

					if (MASKPOPCNT(fi->care[jb], fi->c) == fi->m) { free(fi->care[jb]); fi->care[jb] = NULL; }

					chunk tmp[cs];
					memset(tmp, 0, sizeof(chunk) * cs);
					MASKANDNOT(f2->care[im], f1->care[pfxb], tmp, cs);
					printmask(tmp, f1->s);
					if (ms) tmp[ds] &= f1->mask;
					register const dim poptmp = MASKPOPCNT(tmp, cs);
					printf("poptmp = %u\n", poptmp);

					for (h = 0, k = MASKFFS(tmp, cs); h < poptmp; h++, k = MASKCLEARANDFFS(tmp, k, cs)) {
						if GET(f2->data + im, k, f2->hn) fi->data[DIVBPC(k) * fi->n + jb] |= 1ULL << MODBPC(k);
						printf("k = %u\n", k);
					}
				}
			}

		}
	}

	free(nointersection);
	free(nodifference);
	free(notandmasks);
	free(tmpmask);
	free(masks);
}
