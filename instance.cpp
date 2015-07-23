#include "../cucop/uthash/src/utarray.h"

//static const UT_icd dim_icd = {sizeof(dim), NULL, NULL, NULL };
static const UT_icd chunk_icd = {sizeof(chunk), NULL, NULL, NULL };
static const UT_icd care_icd = {sizeof(chunk *), NULL, NULL, NULL };
static const UT_icd value_icd = {sizeof(value), NULL, NULL, NULL };

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
bool duplicate(const type *buf, unsigned n) {

	if (n < 2) return false;
	//printbuf(buf, n, "unsorted");
	#include "../cucop/iqsort.h"
	#define LT(_a, _b) (*(_a)) < (*(_b))
	type tmp[n];
	memcpy(tmp, buf, sizeof(type) * n);
	QSORT(type, tmp, n, LT);
	//printbuf(tmp, n, "sorted");
	n--;
	register type *p = tmp + 1;
	do { if (*p == *(p - 1)) return true; p++; }
	while (--n);
	return false;
}

__attribute__((always_inline)) inline
void removeduplicates(func *f) {

	if (f->n < 2) return;
	register const dim olds = f->s;
	register const dim oldmask = f->mask;
	f->s = f->m;
	f->mask = (1ULL << MODBPC(f->s)) - 1;
	sort(f);
	f->hn = uniquecombinations(f);

	if (f->hn != f->n) {
		//print(f, "with duplicates");
		register func sof, *of = &sof;
		of->n = f->hn;
		of->m = f->m;
		of->d = f->d;
		ALLOCFUNC(of);
		memcpy(of->vars, f->vars, sizeof(id) * f->m);
		f->h = (dim *)calloc(f->hn, sizeof(dim));
		histogram(f);
		dim pfx[f->hn];
		exclprefixsum(f->h, pfx, f->hn);

		register dim t;
		for (t = 0; t < f->hn; t++) {
			register const dim pfxt = pfx[t];
			of->v[t] = f->v[pfxt];
			if (f->care[pfxt]) {
				of->care[t] = (chunk *)malloc(sizeof(chunk) * of->c);
				memcpy(of->care[t], f->care[pfxt], sizeof(chunk) * of->c);
			}
			register dim k;
			for (k = 0; k < f->c; k++) of->data[k * of->n + t] = f->data[k * f->n + pfxt];
		}

		//print(of, "duplicates removed");
		*f = sof;
	}

	f->mask = oldmask;
	f->s = olds;
}

__attribute__((always_inline)) inline
void instancealldontcare(func *f, chunk* m) {

	register dim i, ext = 0;
	register dim *its = (dim *)calloc(f->n, sizeof(dim));
	register chunk *mask = (chunk *)malloc(sizeof(mask) * f->c * f->n);

	for (i = 0; i < f->n; i++) if (f->care[i]) {
		MASKANDNOT(m, f->care[i], mask + i * f->c, f->c);
		if ((its[i] = MASKPOPCNT(mask + i * f->c, f->c))) ext += (1ULL << its[i]) - 1;
	}

	if (ext) {

		register chunk *extcare[ext];
		register chunk *extdata = (chunk *)malloc(sizeof(chunk) * ext * f->c);
		register value *extv = (value *)malloc(sizeof(value) * ext);
		register dim l = 0;

		for (i = 0; i < f->n; i++) if (its[i]) {

			MASKOR(m, f->care[i], f->care[i], f->c);
			register dim j, k;

			for (j = 1; j < 1ULL << its[i]; j++) {

				register dim b = 0;
				extcare[l] = (chunk *)malloc(sizeof(chunk) * f->c);
				memcpy(extcare[l], f->care[i], sizeof(chunk) * f->c);
				chunk tmp[f->c];
				memcpy(tmp, mask + i * f->c, sizeof(chunk) * f->c);

				for (k = 0; k < f->c; k++) {

					register chunk d = f->data[k * f->n + i];

					while (tmp[k]) {
						register dim idx = __builtin_ffsll(tmp[k]) - 1;
						if (GETBIT(j, b++)) d |= 1ULL << idx;
						tmp[k] ^= 1ULL << idx;
					}

					extdata[k * ext + l] = d;
				}
				extv[l++] = f->v[i];
			}
		}

		f->data = (chunk *)realloc(f->data, sizeof(chunk) * f->c * (f->n + ext));
		f->care = (chunk **)realloc(f->care, sizeof(chunk *) * (f->n + ext));
		f->v = (value *)realloc(f->v, sizeof(value) * (f->n + ext));

		for (i = 0; i < f->c; i++) {
			memmove(f->data + (f->n + ext) * (f->c - i - 1), f->data + f->n * (f->c - i - 1), sizeof(chunk) * f->n);
			memcpy(f->data + (f->n + ext) * (f->c - i - 1) + f->n, extdata + ext * (f->c - i - 1), sizeof(chunk) * ext);
		}

		memcpy(f->care + f->n, extcare, sizeof(chunk *) * ext);
		memcpy(f->v + f->n, extv, sizeof(value) * ext);
		f->n += ext;
		free(extdata);
		free(extv);
	}

	free(mask);
	free(its);
}

/*template <typename type>
__attribute__((always_inline)) inline
bool sorted(const type *buf, unsigned n) {

	if (n < 2) return true;
	n--;
	buf++;
	do { if (*buf < *(buf - 1)) return false; buf++; }
	while (--n);
	return true;
}

__attribute__((always_inline)) inline
void computefd(const dim *bitsUT, dim n, const chunk *masksUT, dim c, dim s, dim i, dim nhdi, const chunk *initmasks,
	       const func *f1, const func *f2, const dim *pfxh1, const dim *pfxh2,
	       const func *fd, const dim *map, dim offs, dim hoffs, dim moffs) {

	register const dim ds = DIVBPC(f1->s);
	register dim j;

	// could be parallelised
	for (j = 0; j < nhdi; j++) {

		register const dim *bits = bitsUT + j * n;
		register const chunk *masks = masksUT + j * n * c;
		printbuf(bits, n, "bits");
		register dim h;

		for (h = 0; h < f1->h[i]; h++) {

			register dim k, e = offs + j * f1->h[i] + h;
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

			chunk maskor[c];
			memset(maskor, 0, sizeof(chunk) * c);
			for (k = 0; k < n; k++) {
				chunk tmp[c];
				memset(tmp, 0, sizeof(chunk) * c);
				MASKANDNOT(initmasks + k * c, masks + k * c, tmp, c);
				printf("original %u\n", k);
				printmask(tmp, s);
				printf("maskor\n");
				printmask(maskor, s);
				MASKANDNOT(tmp, maskor, tmp, c);
				MASKOR(tmp, maskor, maskor, c);
				printf("final %u\n", k);
				printmask(tmp, s);
				MASKOR(fd->care[e], tmp, fd->care[e], c);
				register dim j, b, ntmp = MASKPOPCNT(tmp, c);
				for (j = 0, b = MASKFFS(tmp, c); j < ntmp; j++, b = MASKCLEARANDFFS(tmp, b, c))
					if (GET(f2->data + map[k], b, f2->n)) fd->data[DIVBPC(b) * fd->n + e] |= 1ULL << MODBPC(b);
				SET(fd->care[e], bits[k]);
				if (!GET(f2->data + map[k], bits[k], f2->n)) fd->data[DIVBPC(bits[k]) * fd->n + e] |= 1ULL << MODBPC(bits[k]);
			}

			if (MASKPOPCNT(fd->care[e], fd->c) == fd->m) { free(fd->care[e]); fd->care[e] = NULL; }
			printrow(fd, e);
		}
	}
}

inline void nestedloop(dim *bits, chunk *masks, dim *idx, const dim *initbits, const chunk *initmasks, const dim *popcnt,
		       dim l, dim n, dim s, dim c, dim *j, UT_array *bitsUT, UT_array *masksUT) {

	if (l == n) {
		if (!duplicate(bits, n) && sorted(bits, n)) {
			//printbuf(idx, n, "idx");
			//printbuf(bits, n, "bits");
			register dim k;
			for (k = 0; k < n; k++) utarray_push_back(bitsUT, bits + k);
			for (k = 0; k < c * n; k++) utarray_push_back(masksUT, masks + k);
			(*j)++;
		}
	} else {
		for (bits[l] = initbits[l], memcpy(masks + l * c, initmasks + l * c, sizeof(chunk) * c), idx[l] = 0;
		     idx[l] < popcnt[l]; idx[l]++, bits[l] = MASKCLEARANDFFS(masks + l * c, bits[l], c)) {
			//printmask(masks + l * c, s);
			nestedloop(bits, masks, idx, initbits, initmasks, popcnt, l + 1, n, s, c, j, bitsUT, masksUT);
		}
	}
}*/

#define INTERSECTMASK(F1, I, F2, J, S, MASK) ({ register dim _i; register char cmp = 0; register const dim ds = DIVBPC(S); \
						register chunk * const ca = (F1)->care[I]; register chunk * const cb = (F2)->care[J]; \
						if (!ca && !cb) cmp = DATACOMPARE((F1)->data + (I), (F2)->data + (J), F1, F2); \
						else { register chunk mask; for (_i = 0; _i < ds; _i++) { \
						mask = (MASK) & ((ca && cb) ? ca[_i] & cb[_i] : (ca ? ca[_i] : cb[_i])); \
						if ((cmp = CMP((F1)->data[_i * (F1)->n + (I)] & mask, (F2)->data[_i * (F2)->n + (J)] & mask))) break; } \
						if (!cmp) { if (MASK) mask = (MASK) & ((ca && cb) ? (ca[ds] & cb[ds]) : (ca ? ca[ds] : cb[ds])); \
						cmp = ((MASK) ? CMP(mask & (F1)->data[ds * (F1)->n + (I)], \
								    mask & (F2)->data[ds * (F2)->n + (J)]) : 0); } } !cmp; })

#define INTERSECT(F1, I, F2, J) INTERSECTMASK(F1, I, F2, J, (F1)->s, (F1)->mask)

inline dim computefi(chunk *maskstack, func *fstack, dim popcnt, dim i, dim cm, const func *fi, const dim *pfxh, UT_array *dataUT, UT_array *careUT) {

	if (!popcnt) return 0;

	register const chunk mmask = (1ULL << MODBPC(fi->m)) - 1;
	register func *const f = fstack + 1;
	register chunk *const m = maskstack + cm;
	memcpy(m, maskstack, sizeof(chunk) * cm);
	*f = *fstack;
	ALLOCFUNC(f);
	COPYFIELDS(f, fstack);
	memcpy(f->data, fstack->data, sizeof(chunk) * fi->c);
	f->care[0] = (chunk *)malloc(sizeof(chunk) * fi->c);
	memcpy(f->care[0], fstack->care[0], sizeof(chunk) * fi->c);

	register const dim idx = fi->s + MASKFFSANDCLEAR(m, cm);
	register dim j, ret = 0;
	register bool in;

	// try 0

	SET(f->care[0], idx);
	in = false;

	//#pragma omp parallel for private(j) reduction(||:in)
	for (j = 0; j < fi->h[i]; j++) if (INTERSECTMASK(f, 0, fi, pfxh[i] + j, fi->m, mmask)) { in = true; break; }

	if (!in) {

		utarray_reserve(dataUT, fi->c);
		utarray_reserve(careUT, 1);
		register chunk *const tmpdata = ((chunk *)dataUT->d) + utarray_len(dataUT);
		register chunk **const tmpcare = ((chunk **)careUT->d) + utarray_len(careUT);
		dataUT->i++;
		careUT->i++;
		ret++;
		memcpy(tmpdata, f->data, sizeof(chunk) * fi->c);

		if (MASKPOPCNT(f->care[0], fi->c) != fi->m) {
			tmpcare[0] = (chunk *)malloc(sizeof(chunk) * fi->c);
			memcpy(tmpcare[0], f->care[0], sizeof(chunk) * fi->c);
		} else tmpcare[0] = NULL;

	} else ret += computefi(m, f, popcnt - 1, i, cm, fi, pfxh, dataUT, careUT);

	// try 1

	SET(f->data, idx);
	in = false;

	//#pragma omp parallel for private(j) reduction(||:in)
	for (j = 0; j < fi->h[i]; j++) if (INTERSECTMASK(f, 0, fi, pfxh[i] + j, fi->m, mmask)) { in = true; break; }

	if (!in) {

		utarray_reserve(dataUT, fi->c);
		utarray_reserve(careUT, 1);
		register chunk *const tmpdata = ((chunk *)dataUT->d) + utarray_len(dataUT);
		register chunk **const tmpcare = ((chunk **)careUT->d) + utarray_len(careUT);
		dataUT->i++;
		careUT->i++;
		ret++;
		memcpy(tmpdata, f->data, sizeof(chunk) * fi->c);

		if (MASKPOPCNT(f->care[0], fi->c) != fi->m) {
			tmpcare[0] = (chunk *)malloc(sizeof(chunk) * fi->c);
			memcpy(tmpcare[0], f->care[0], sizeof(chunk) * fi->c);
		} else tmpcare[0] = NULL;

	} else ret += computefi(m, f, popcnt - 1, i, cm, fi, pfxh, dataUT, careUT);

	return ret;
}

inline dim computefd(chunk *maskstack, func *fstack, dim popcnt, dim c, dim fdm, dim fdc, dim i, const dim *imap, dim nhi,
		     const func *f1, const dim *pfxh1, const func *f2, const dim *pfxh2, dim moffs, UT_array *dataUT, UT_array *careUT, UT_array *valueUT) { 

	if (!popcnt) return 0;

	register func *const f = fstack + 1;
	register chunk *const m = maskstack + c;
	memcpy(m, maskstack, sizeof(chunk) * c);
	*f = *fstack;
	ALLOCFUNC(f);
	COPYFIELDS(f, fstack);
	memcpy(f->data, fstack->data, sizeof(chunk) * c);
	f->care[0] = (chunk *)malloc(sizeof(chunk) * c);
	memcpy(f->care[0], fstack->care[0], sizeof(chunk) * c);

	register const dim idx = MASKFFSANDCLEAR(m, c);
	register dim j, h, ret = 0;
	register bool in;

	// try 0

	SET(f->care[0], idx);
	in = false;

	for (j = 0; j < nhi; j++) if (INTERSECT(f, 0, f2, imap[j])) { in = true; break; }

	if (!in) {

		ret += f1->h[i];
		utarray_reserve(dataUT, f1->h[i] * fdc);
		utarray_reserve(careUT, f1->h[i]);
		utarray_reserve(valueUT, f1->h[i]);
		register chunk *const tmpdata = ((chunk *)dataUT->d) + utarray_len(dataUT);
		register chunk **const tmpcare = ((chunk **)careUT->d) + utarray_len(careUT);
		register value *const tmpvalue = ((value *)valueUT->d) + utarray_len(valueUT);
		dataUT->i += f1->h[i];
		careUT->i += f1->h[i];
		valueUT->i += f1->h[i];

		for (h = 0; h < f1->h[i]; h++) {

			register const dim pfxh = pfxh1[i] + h;
			tmpvalue[h] = f1->v[pfxh];
			memcpy(tmpdata + h * fdc, f->data, sizeof(chunk) * c);
			tmpcare[h] = (chunk *)malloc(sizeof(chunk) * fdc);
			memcpy(tmpcare[h], f->care[0], sizeof(chunk) * c);

			for (j = 0; j < f1->m - f1->s; j++) {
				if (GET(f1->data + pfxh, f1->s + j, f1->n))
					tmpdata[h * fdc + DIVBPC(f1->s + moffs + j)] |= 1ULL << MODBPC(f1->s + moffs + j);
				if (!f1->care[pfxh] || GET(f1->care[pfxh], f1->s + j)) SET(tmpcare[h], f1->s + moffs + j);
			}

			if (MASKPOPCNT(tmpcare[h], fdc) == fdm) { free(tmpcare[h]); tmpcare[h] = NULL; }
		}

	} else ret += computefd(m, f, popcnt - 1, c, fdm, fdc, i, imap, nhi, f1, pfxh1, f2, pfxh2, moffs, dataUT, careUT, valueUT);

	// try 1

	SET(f->data, idx);
	in = false;

	for (j = 0; j < nhi; j++) if (INTERSECT(f, 0, f2, imap[j])) { in = true; break; }

	if (!in) {

		ret += f1->h[i];
		utarray_reserve(dataUT, f1->h[i] * fdc);
		utarray_reserve(careUT, f1->h[i]);
		utarray_reserve(valueUT, f1->h[i]);
		register chunk *const tmpdata = ((chunk *)dataUT->d) + utarray_len(dataUT);
		register chunk **const tmpcare = ((chunk **)careUT->d) + utarray_len(careUT);
		register value *const tmpvalue = ((value *)valueUT->d) + utarray_len(valueUT);
		dataUT->i += f1->h[i];
		careUT->i += f1->h[i];
		valueUT->i += f1->h[i];

		for (h = 0; h < f1->h[i]; h++) {

			register const dim pfxh = pfxh1[i] + h;
			tmpvalue[h] = f1->v[pfxh];
			memcpy(tmpdata + h * fdc, f->data, sizeof(chunk) * c);
			tmpcare[h] = (chunk *)malloc(sizeof(chunk) * fdc);
			memcpy(tmpcare[h], f->care[0], sizeof(chunk) * c);

			for (j = 0; j < f1->m - f1->s; j++) {
				if (GET(f1->data + pfxh, f1->s + j, f1->n))
					tmpdata[h * fdc + DIVBPC(f1->s + moffs + j)] |= 1ULL << MODBPC(f1->s + moffs + j);
				if (!f1->care[pfxh] || GET(f1->care[pfxh], f1->s + j)) SET(tmpcare[h], f1->s + moffs + j);
			}

			if (MASKPOPCNT(tmpcare[h], fdc) == fdm) { free(tmpcare[h]); tmpcare[h] = NULL; }
		}

	} else ret += computefd(m, f, popcnt - 1, c, fdm, fdc, i, imap, nhi, f1, pfxh1, f2, pfxh2, moffs, dataUT, careUT, valueUT);

	return ret;
}

// DIFFERENCE is true if I minus J is not empty

#define DIFFERENCE(F1, I, F2, J, CS, TMP) ({ register char res = 0; if ((F1)->care[I]) { \
					     MASKNOTAND((F1)->care[I], ONESIFNULL((F2)->care[J]), TMP, CS); \
					     MASKAND(TMP, ones, TMP, CS); if (MASKPOPCNT(TMP, CS)) res = 1; } res; })

// fi = intersection between f1 and f2
// fd = difference between f1 and f2

__attribute__((always_inline)) inline
void instancedefaults(func *fi, const dim *pfxh) {

	register dim i, ei, ext = 0;
	register const dim popcnt = fi->m - fi->s;
	register const dim cp = CEILBPC(popcnt);
	register const dim cs = CEILBPC(fi->s);
	register const dim ds = DIVBPC(fi->s);
	register chunk ones[cs];
	ONES(ones, fi->s, cs);

	register UT_array *dataUT, *careUT, *valueUT;
	utarray_new(dataUT, &chunk_icd);
	utarray_new(careUT, &care_icd);
	utarray_new(valueUT, &value_icd);

	for (i = 0; i < fi->hn; i++) {

		chunk maskstack[(popcnt + 1) * cp];
		memset(maskstack, 0, sizeof(chunk) * cp);
		ONES(maskstack, popcnt, cp);
		func fstack[popcnt + 1];
		fstack->n = 1;
		fstack->m = fi->m;
		ALLOCFUNC(fstack);
		COPYFIELDS(fstack, fi);

		register dim j;
		for (j = 0; j < cs; j++) fstack->data[j] = fi->data[j * fi->n + pfxh[i]];
		fstack->care[0] = (chunk *)calloc(fi->c, sizeof(chunk));
		memcpy(fstack->care[0], ONESIFNULL(fi->care[pfxh[i]]), sizeof(chunk) * cs);
		if (fi->mask) {
			fstack->data[ds] &= fi->mask;
			fstack->care[0][ds] &= fi->mask;
		}

		//print(fstack, "fstack");
		printf("popcnt = %u h[i] = %u\n", popcnt, fi->h[i]);
		ei = computefi(maskstack, fstack, popcnt, i, cp, fi, pfxh, dataUT, careUT);
		fi->h[i] += ei;
		ext += ei;
	}

	if (ext) {
		printf("ext = %u\n", ext);
		transpose(fi->data, fi->c, fi->n);
		fi->data = (chunk *)realloc(fi->data, sizeof(chunk) * fi->c * (fi->n + ext));
		fi->care = (chunk **)realloc(fi->care, sizeof(chunk *) * (fi->n + ext));
		fi->v = (value *)realloc(fi->v, sizeof(value) * (fi->n + ext));
		memcpy(fi->data + fi->c * fi->n, dataUT->d, sizeof(chunk) * ext * fi->c);
		memcpy(fi->care + fi->n, careUT->d, sizeof(chunk *) * ext);
		for (i = 0; i < ext; i++) fi->v[fi->n + i] = fi->d;
		fi->n += ext;
		transpose(fi->data, fi->n, fi->c);
	}

	utarray_free(dataUT);
	utarray_free(careUT);
	utarray_free(valueUT);
}

template <bool joint = true>
__attribute__((always_inline)) inline
void instancedontcare(func *f1, func *f2, dim f3m, dim moffs, const dim *pfxh1, const dim *pfxh2, func *fi, func *fd) {

	register const dim cs = CEILBPC(f1->s);
	register const dim cn = CEILBPC(f1->hn);
	register const dim ds = DIVBPC(f1->s);
	register const dim ms = MODBPC(f1->s);
	register const dim f1nf2n = f1->hn * f2->hn;

	register dim i, j, tni = 0, tnhi = 0, tnd = 0;
	register chunk *tmpmask = (chunk *)calloc(cn, sizeof(chunk));
	register chunk *nodifference = (chunk *)calloc(cn, sizeof(chunk));
	register chunk *nointersection = (chunk *)calloc(cn, sizeof(chunk));
	ONES(nointersection, f1->hn, cn);
	ONES(nodifference, f1->hn, cn);

	register chunk *mask = (chunk *)calloc(f1->n * cs, sizeof(chunk));
	//register chunk *masks = (chunk *)malloc(sizeof(chunk) * f1nf2n * cs);

	/*register dim popcnt[f1nf2n];
	register dim bits[f1nf2n];
	register dim init[f1nf2n];
	register dim idx[f1nf2n];*/
	register dim map[f1nf2n];

	register dim ni[f1->hn];
	register dim pfxni[f1->hn];
	//register dim nd[f1->hn];
	//register dim pfxnd[f1->hn];

	register dim nhi[f1->hn];
	register dim pfxnhi[f1->hn];
	//register dim nhd[f1->hn];
	//register dim pfxnhd[f1->hn];
	memset(ni, 0, sizeof(dim) * f1->hn);
	//memset(nd, 0, sizeof(dim) * f1->hn);
	memset(nhi, 0, sizeof(dim) * f1->hn);
	//memset(nhd, 0, sizeof(dim) * f1->hn);

	//register UT_array *bitsUT[f1->hn];
	//register UT_array *masksUT[f1->hn];

	register chunk ones[cs];
	ONES(ones, f1->s, cs);

	//printbuf(pfxh1, f1->hn, "pfxh1");
	//printbuf(pfxh2, f2->hn, "pfxh2");

	register UT_array *dataUT, *careUT, *valueUT;
	utarray_new(dataUT, &chunk_icd);
	utarray_new(careUT, &care_icd);
	utarray_new(valueUT, &value_icd);

	for (i = 0; i < f1->hn; i++) {

		//register const dim if2n = i * f2->hn;
		register chunk *const imask = mask + i * cs;
		/*register chunk *const imasks = masks + if2n;
		register dim *const ipopcnt = popcnt + if2n;
		register dim *const ibits = bits + if2n;
		register dim *const iinit = init + if2n;
		register dim *const iidx = idx + if2n;*/
		register dim *const imap = map + i * f2->hn;
		register chunk tmp[cs];

		for (j = 0; j < f2->hn; j++)
			if (INTERSECT(f1, pfxh1[i], f2, pfxh2[j])) {
				if (!joint && i == j && f1->h[i] == 1) continue;
				CLEAR(nointersection, i);
				if (DIFFERENCE(f1, pfxh1[i], f2, pfxh2[j], cs, tmp)) {
					MASKNOTAND(f1->care[pfxh1[i]], ONESIFNULL(f2->care[pfxh2[j]]), tmp, cs);
					if (ms) tmp[cs - 1] &= ones[cs - 1];
					MASKOR(tmp, imask, imask, cs);
					/*ipopcnt[nhi[i]] = MASKPOPCNT(inotandmasks + nhi[i] * cs, cs);
					iinit[nhi[i]] = MASKFFS(inotandmasks + nhi[i] * cs, cs);*/
					CLEAR(nodifference, i);
					imap[nhi[i]] = pfxh2[j];
					nhi[i]++;
				}
			}

		//printmask(imask, f1->s);
		//printf("%u %u\n", i, nhi[i]);

		if (joint && nhi[i]) {
			//printf("Row %u\n", i);
			//printbuf(init, m, "init");
			//printbuf(popcnt, m, "popcnt");
			//printbuf(imap, nhi[i], "imap");
			//utarray_new(bitsUT[i], &dim_icd);
			//utarray_new(masksUT[i], &chunk_icd);
			//nestedloop(ibits, imasks, iidx, iinit, inotandmasks, ipopcnt, 0, nhi[i], f1->s, cs, nhd + i, bitsUT[i], masksUT[i]);
			register const dim popcnt = MASKPOPCNT(imask, cs);
			chunk maskstack[(popcnt + 1) * cs];
			memcpy(maskstack, imask, sizeof(chunk) * cs);
			func fstack[popcnt + 1];
			fstack->n = 1;
			fstack->m = f1->s;
			ALLOCFUNC(fstack);
			COPYFIELDS(fstack, f1);
			memset(fstack->vars, 0, sizeof(id) * f1->s);
			for (j = 0; j < cs; j++) fstack->data[j] = f1->data[j * f1->n + pfxh1[i]];
			fstack->care[0] = (chunk *)calloc(cs, sizeof(chunk));
			memcpy(fstack->care[0], f1->care[pfxh1[i]], sizeof(chunk) * cs);
			if (f1->mask) {
				fstack->data[ds] &= f1->mask;
				fstack->care[0][ds] &= f1->mask;
			}
			//print(fstack, "fstack");
			//BREAKPOINT("pre computefd");
			tnd += computefd(maskstack, fstack, popcnt, cs, f3m, CEILBPC(f3m), i, imap, nhi[i], f1, pfxh1, f2, pfxh2, moffs, dataUT, careUT, valueUT);
			//BREAKPOINT("after computefd");

		}

		tnhi += nhi[i];
		//tnhd += nhd[i];
		tni += (ni[i] = nhi[i] * f1->h[i]);
		//tnd += (nd[i] = nhd[i] * f1->h[i]);
	}

	MASKANDNOT(nodifference, nointersection, nodifference, cn);

	fd->m = f3m;
	fi->m = f1->m;

	//puts("nodifference"); printmask(nodifference, f1->hn);
	//puts("nointersection"); printmask(nointersection, f1->hn);
	register const dim popnd = MASKPOPCNT(nodifference, cn);
	register const dim popni = MASKPOPCNT(nointersection, cn);
	//printf("popnd = %u\n", popnd);
	//printf("popnin = %u\n", popni);
	register dim nnd = 0, nni = 0;

	memcpy(tmpmask, nodifference, sizeof(chunk) * cn);
	for (i = 0, j = MASKFFS(tmpmask, cn); i < popnd; i++, j = MASKCLEARANDFFS(tmpmask, j, cn)) nnd += f1->h[j];
	memcpy(tmpmask, nointersection, sizeof(chunk) * cn);
	for (i = 0, j = MASKFFS(tmpmask, cn); i < popni; i++, j = MASKCLEARANDFFS(tmpmask, j, cn)) nni += f1->h[j];

	fi->n = tni + nnd;
	fd->n = tnd + nni;
	ALLOCFUNC(fi);
	ALLOCFUNC(fd);
	COPYFIELDS(fi, f1);
	COPYFIELDS(fd, f1);
	memcpy(fi->vars, f1->vars, sizeof(id) * f1->m);
	memset(fd->vars, 0, sizeof(id) * f3m);

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
	}

	memcpy(fd->data + nni * fd->c, dataUT->d, sizeof(chunk) * tnd * fd->c);
	transpose(fd->data + nni * fd->c, tnd, fd->c);
	memcpy(fd->care + nni, careUT->d, sizeof(chunk *) * tnd);
	memcpy(fd->v + nni, valueUT->d, sizeof(value) * tnd);
	utarray_free(dataUT);
	utarray_free(careUT);
	utarray_free(valueUT);

	exclprefixsum(ni, pfxni, f1->hn);
	//exclprefixsum(nd, pfxnd, f1->hn);
	exclprefixsum(nhi, pfxnhi, f1->hn);
	//exclprefixsum(nhd, pfxnhd, f1->hn);
	//printbuf(nhi, f1->hn, "nhi");
	//printbuf(nhd, f1->hn, "nhd");
	//printbuf(pfxnhi, f1->hn, "pfxnhi");
	//printbuf(pfxnhd, f1->hn, "pfxnhd");
	//printbuf(ni, f1->hn, "ni");
	//printbuf(nd, f1->hn, "nd");
	//printbuf(pfxni, f1->hn, "pfxni");
	//printbuf(pfxnd, f1->hn, "pfxnd");

	//BREAKPOINT("preloop");
	// TODO: Could be parallelised
	for (i = 0, j = 0; i < f1->hn; i++, j = 0) {

		if (nhi[i]) {

			register dim a, b;
			register const dim if2n = i * f2->hn;
			//register chunk *const inotandmasks = notandmasks + if2n;
			//register chunk *const imasks = masks + if2n;
			//register dim *const ipopcnt = popcnt + if2n;
			//register dim *const ibits = bits + if2n;
			//register dim *const iinit = init + if2n;
			//register dim *const iidx = idx + if2n;
			register dim *const imap = map + if2n;

			for (a = 0, j = nnd + pfxni[i]; a < nhi[i]; a++, j += f1->h[i]) {

				//fi->h[popnd + pfxnhi[i] + a] = f1->h[i];
				register const dim im = imap[a];
				register dim h, k;

				for (b = 0; b < f1->h[i]; b++) {

					register const dim jb = j + b;
					register const dim pfxb = pfxh1[i] + b;
					//printf("pfxb = %u\n", pfxb);
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
	//BREAKPOINT("postloop");

	free(nointersection);
	free(nodifference);
	free(tmpmask);
	free(mask);
	//free(masks);
	removeduplicates(fi);
	/*instancealldontcare(f1, ones);
	print(f1, "entire", ones);
	print(fd, "original fd", ones);
	instancealldontcare(fd, ones);
	print(fd, "fd instantianted", ones);*/
}
