#include "../cucop/uthash/src/utarray.h"

//static const UT_icd dim_icd = {sizeof(dim), NULL, NULL, NULL };
static const UT_icd chunk_icd = {sizeof(chunk), NULL, NULL, NULL };
static const UT_icd care_icd = {sizeof(chunk *), NULL, NULL, NULL };
static const UT_icd value_icd = {sizeof(value), NULL, NULL, NULL };

__attribute__((always_inline)) inline
void printmask(const chunk *m, dim n) {

	register dim i;
	for (i = 0; i < n; i++) printf(GET(m, i) ? "1 " : "0 ");
	printf("\n");
}

#include <iostream>
template <typename type>
__attribute__((always_inline)) inline
void printbuf(const type *buf, unsigned n, const char *name) {

	printf("%s = [ ", name);
	while (n--) std::cout << *(buf++) << " ";
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

/*__attribute__((always_inline)) inline
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
}*/

__attribute__((always_inline)) inline
void instanceshared(func *f) {

	register dim text = 0;
	register const dim cs = CEILBPC(f->s);
	chunk ones[cs];
	ONES(ones, f->s, cs);

	dim popc[f->n], ext[f->n], pfxext[f->n];
	memset(ext, 0, sizeof(dim) * f->n);
	register chunk *mask = (chunk *)malloc(sizeof(mask) * cs * f->n);

	for (dim i = 0; i < f->n; i++) {
		MASKANDNOT(ones, CARE(f, i), mask + i * cs, cs);
		//printmask(mask + i * cs, f->s);
		popc[i] = MASKPOPCNT(mask + i * cs, cs);
		if (popc[i]) text += (ext[i] = (1ULL << popc[i]) - 1);
	}

	exclprefixsum(ext, pfxext, f->n);
	//printbuf(ext, f->n, "ext");
	//printbuf(pfxext, f->n, "pfxext");

	if (text) {
		f->data = (chunk *)realloc(f->data, sizeof(chunk) * 2 * f->c * (f->n + text));
		f->v = (value *)realloc(f->v, sizeof(value) * (f->n + text));
		//#pragma omp parallel for schedule(dynamic) private(i)
		for (dim i = 0; i < f->n; i++) if (ext[i]) {
			MASKOR(ones, CARE(f, i), CARE(f, i), cs);
			for (dim j = 1; j < 1ULL << popc[i]; j++) {
				register const dim idx = f->n + pfxext[i] + (j - 1);
				//printf("idx = %u\n", idx);
				memcpy(DATA(f, idx), DATA(f, i), sizeof(chunk) * 2 * f->c);
				f->v[idx] = f->v[i];
				chunk tmp[cs];
				memcpy(tmp, mask + i * cs, sizeof(chunk) * cs);
				for (dim a = 0, b = MASKFFS(tmp, cs); a < popc[i]; a++, b = MASKCLEARANDFFS(tmp, b, cs))
					if (GETBIT(j, a)) SET(DATA(f, idx), b);
			}
		}
	}

	f->n += text;
	free(mask);
}

inline dim computerows(chunk *maskstack, dim cm, dim popcnt, func *fstack, const func *f, dim s,
		       const dim *rowmap, const dim *pfxh, dim nmap, UT_array *dataUT, dim hi = 1, dim pfxi = 0);

__attribute__((always_inline)) inline
dim checkandcopy(chunk *m, dim cm, dim popcnt, func *fs, const func *f, dim s, const dim *rowmap, const dim *pfxh,
		 dim nmap, UT_array *dataUT, dim hi, dim pfxi) {

	register const chunk smask = (1ULL << MODBPC(s)) - 1;
	chunk tmp1[f->c], tmp2[f->c];
	register bool in = false;
	register dim ret = 0;

	//#pragma omp parallel for private(j) reduction(||:in)
	for (dim j = 0; j < nmap; j++) if (INTERSECTMASK(fs, 0, f, pfxh[rowmap[j]], s, smask, tmp1, tmp2)) { in = true; break; }

	if (!in) {

		//printmask(m, s);
		//puts("matching");
		//printrow(fs, 0);
		//printf("pfxi = %u hi = %u\n", pfxi, hi);
		utarray_reserve(dataUT, 2 * f->c * hi);
		register chunk *const ptr = ((chunk *)dataUT->d) + utarray_len(dataUT);
		for (dim h = 0; h < hi; h++) {
			memcpy(ptr + h * 2 * f->c, DATA(fs, 0), sizeof(chunk) * 2 * f->c);
			//TODO: could be optimised
			for (dim j = s; j < f->m; j++) {
				//printf("data = %u care = %u\n", h * 2 * f->c, (h * 2 + 1) * f->c
				if GET(DATA(f, pfxi + h), j) SET(ptr + h * 2 * f->c, j); else CLEAR(ptr + h * 2 * f->c, j);
				if GET(CARE(f, pfxi + h), j) SET(ptr + (h * 2 + 1) * f->c, j); else CLEAR(ptr + (h * 2 + 1) * f->c, j);
			}
		}
		dataUT->i += 2 * f->c * hi;
		ret += hi;

	} else ret += computerows(m, cm, popcnt - 1, fs, f, s, rowmap, pfxh, nmap, dataUT, hi, pfxi);

	return ret;
}

inline dim computerows(chunk *maskstack, dim cm, dim popcnt, func *fstack, const func *f, dim s,
		       const dim *rowmap, const dim *pfxh, dim nmap, UT_array *dataUT, dim hi, dim pfxi) {

	if (!popcnt) return 0;

	register func *const fs = fstack + 1;
	register chunk *const m = maskstack + cm;
	memcpy(m, maskstack, sizeof(chunk) * cm);
	*fs = *fstack;
	ALLOCFUNC(fs);
	memcpy(DATA(fs, 0), DATA(fstack,0), sizeof(chunk) * 2 * f->c);

	register const dim idx = MASKFFSANDCLEAR(m, cm);
	register dim ret = 0;

	// try 0
	SET(CARE(fs, 0), idx);
	ret += checkandcopy(m, cm, popcnt, fs, f, s, rowmap, pfxh, nmap, dataUT, hi, pfxi);

	// try 1
	SET(DATA(fs, 0), idx);
	ret += checkandcopy(m, cm, popcnt, fs, f, s, rowmap, pfxh, nmap, dataUT, hi, pfxi);

	return ret;
}

__attribute__((always_inline)) inline
void instancezeros(func *f) {

	register UT_array *dataUT;
	utarray_new(dataUT, &chunk_icd);

	chunk maskstack[(f->m + 1) * f->c];
	memset(maskstack, 0, sizeof(chunk) * f->c);
	ONES(maskstack, f->m, f->c);
	func fstack[f->m + 1];
	fstack->n = 1;
	fstack->m = f->m;
	ALLOCFUNC(fstack);
	dim rowmap[f->n];
	for (dim i = 0; i < f->n; i++) rowmap[i] = i;
	register const dim ext = computerows(maskstack, f->c, f->m, fstack, f, f->m, rowmap, rowmap, f->n, dataUT);
	//printf("ext = %u\n", ext);

	if (ext) {
		f->data = (chunk *)realloc(f->data, sizeof(chunk) * 2 * f->c * (f->n + ext));
		f->v = (value *)realloc(f->v, sizeof(value) * (f->n + ext));
		memcpy(f->data + 2 * f->c * f->n, dataUT->d, sizeof(chunk) * 2 * f->c * ext);
		memset(f->v + f->n, 0, sizeof(value) * ext);
		f->n += ext;
	}

	utarray_free(dataUT);
}
