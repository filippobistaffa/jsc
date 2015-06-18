#include "jsc.h"

void randomdata(func f) { // assumes BITSPERCHUNK == 64

	register dim i, j;

	for (i = 0; i < f.n; i++) {
		for (j = 0; j < f.m / BITSPERCHUNK; j++) f.data[j * f.n + i] = genrand64_int64();
		if (f.m % BITSPERCHUNK) f.data[(f.m / BITSPERCHUNK) * f.n + i] = genrand64_int64() & ((1ULL << (f.m % BITSPERCHUNK)) - 1);
	}
}

void print(func f, chunk *s) {

	register dim i, j, k;

	#define WIDTH "2"
	#define FORMAT "%" WIDTH "u"
	#define BITFORMAT "%" WIDTH "zu"

	for (i = 0; i < f.m; i++)
		printf(i & 1 ? FORMAT : WHITE(FORMAT), i);
	printf("\n");

	for (i = 0; i < f.m; i++)
		if (s) {
			if ((s[i / BITSPERCHUNK] >> (i % BITSPERCHUNK)) & 1) printf(i & 1 ? DARKGREEN(FORMAT) : GREEN(FORMAT), f.vars[i]);
			else printf(i & 1 ? DARKRED(FORMAT) : RED(FORMAT), f.vars[i]);
		} else printf(i & 1 ? DARKCYAN(FORMAT) : CYAN(FORMAT), f.vars[i]);
	printf("\n");

	for (i = 0; i < f.n; i++) {
		for (j = 0; j < f.m / BITSPERCHUNK; j++)
			for (k = 0; k < BITSPERCHUNK; k++)
				printf(k & 1 ? BITFORMAT : WHITE(BITFORMAT), (f.data[j * f.n + i] >> k) & 1);
		for (k = 0; k < f.m % BITSPERCHUNK; k++)
			printf(k & 1 ? BITFORMAT : WHITE(BITFORMAT), (f.data[(f.m / BITSPERCHUNK) * f.n + i] >> k) & 1);
		printf(" = %f\n", f.v[i]);
	}
}

void shared2least(func f, chunk* m) {

	register dim x, y, i, n = 0;
	register id t;
	chunk s[f.c], a[f.c], o[f.c];
	memset(s, 0, sizeof(chunk) * f.c);

	for (i = 0; i < f.s / BITSPERCHUNK; i++) s[i] = ~(0ULL);
	if (f.s % BITSPERCHUNK) s[f.s / BITSPERCHUNK] = f.mask;

	for (i = 0; i < f.c; i++) {
		a[i] = s[i] & ~m[i];
		o[i] = m[i] & ~s[i];
		n += __builtin_popcountll(o[i]);
	}

	if (!n) return;
	memcpy(m, s, sizeof(chunk) * f.c);

	do {
		i = x = y = 0;
		while (!o[i++]) x += BITSPERCHUNK;
		x += __builtin_ctzll(o[i - 1]);
		i = 0;
		while (!a[i++]) y += BITSPERCHUNK;
		y += __builtin_ctzll(a[i - 1]);
		t = f.vars[x];
		f.vars[x] = f.vars[y];
		f.vars[y] = t;
		#pragma omp parallel for private(i)
		for (i = 0; i < f.n; i++) SWAP(f.data + i, x, y, f.n);
		o[x / BITSPERCHUNK] ^= 1ULL << (x % BITSPERCHUNK);
		a[y / BITSPERCHUNK] ^= 1ULL << (y % BITSPERCHUNK);
	} while (--n);
}

/*

void shared2least(func f, chunk* m) {

	register dim x, y, i;
	register id t;
	chunk* s = (chunk *)calloc(f.c, sizeof(chunk));
	chunk* z = (chunk *)calloc(f.c, sizeof(chunk));
	chunk* a = (chunk *)malloc(sizeof(chunk) * f.c);
	chunk* o = (chunk *)malloc(sizeof(chunk) * f.c);

	for (i = 0; i < f.s / BITSPERCHUNK; i++) s[i] = ~(0ULL);
	if (f.s % BITSPERCHUNK) s[f.s / BITSPERCHUNK] = f.mask;

	for (i = 0; i < f.c; i++) {
		a[i] = s[i] & ~m[i];
		o[i] = m[i] & ~s[i];
	}

	memcpy(m, s, sizeof(chunk) * f.c);

	while (memcmp(o, z, f.c * sizeof(chunk))) {
		i = x = y = 0;
		while (!o[i++]) x += BITSPERCHUNK;
		x += __builtin_ctzll(o[i - 1]);
		i = 0;
		while (!a[i++]) y += BITSPERCHUNK;
		y += __builtin_ctzll(a[i - 1]);
		t = f.vars[x];
		f.vars[x] = f.vars[y];
		f.vars[y] = t;
		#pragma omp parallel for private(i)
		for (i = 0; i < f.n; i++) SWAP(f.data + i, x, y, f.n);
		o[x / BITSPERCHUNK] ^= 1ULL << (x % BITSPERCHUNK);
		a[y / BITSPERCHUNK] ^= 1ULL << (y % BITSPERCHUNK);
	}

	free(s);
	free(z);
	free(a);
	free(o);
}

*/

void reordershared(func f, id *vars) {

	chunk s[CEIL(f.s, BITSPERCHUNK)];
	register dim i, j;
	id *v = (id *)malloc(sizeof(id) * MAXVAR);

	for (i = 0; i < f.s; i++) v[vars[i]] = i;
	#pragma omp parallel for private(i, s)
	for (i = 0; i < f.n; i++) {
		memset(s, 0, sizeof(chunk) * CEIL(f.s, BITSPERCHUNK));
		for (j = 0; j < f.s; j++) if GET(f.data + i, j, f.n) SET(s, v[f.vars[j]]);
		for (j = 0; j < f.s / BITSPERCHUNK; j++) f.data[j * f.n + i] = s[j];
		if (f.mask) {
			f.data[(f.s / BITSPERCHUNK) * f.n + i] &= ~f.mask;
			f.data[(f.s / BITSPERCHUNK) * f.n + i] |= s[f.s / BITSPERCHUNK];
		}
	}

	memcpy(f.vars, vars, sizeof(id) * f.s);
	free(v);
}

dim uniquecombinations(func f) {

	register dim i, j, u = 1;

	for (i = 1; i < f.n; i++) {
		for (j = 0; j < DIVBPC(f.s); j++)
			if (f.data[j * f.n + i] != f.data[j * f.n + i - 1]) { u++; goto next; }
		if (f.mask & (f.data[DIVBPC(f.s) * f.n + i] ^ f.data[DIVBPC(f.s) * f.n + i - 1])) u++;
		next:;
	}

	return u;
}

dim invuniquecombinations(func f) {

	register dim i, j, u = 1;

	for (i = 1; i < f.n; i++) {
		if (f.mask && (f.data[DIVBPC(f.s) * f.n + i] >> MODBPC(f.s)) ^
		    (f.data[DIVBPC(f.s) * f.n + i - 1] >> MODBPC(f.s))) { u++; continue; }
		for (j = DIVBPC(f.s) + (f.mask ? 1 : 0); j < f.c; j++)
			if (f.data[j * f.n + i] != f.data[j * f.n + i - 1]) { u++; break; }
	}

	return u;
}

void histogram(func f) {

	register dim i, j, k;
	f.h[0] = 1;

	for (i = 1, k = 0; i < f.n; i++) {
		for (j = 0; j < DIVBPC(f.s); j++)
			if (f.data[j * f.n + i] != f.data[j * f.n + i - 1]) { k++; goto next; }
		if (f.mask & (f.data[DIVBPC(f.s) * f.n + i] ^ f.data[DIVBPC(f.s) * f.n + i - 1])) k++;
		next:
		f.h[k]++;
	}
}

void invhistogram(func f) {

        register dim i, j, k;
        f.h[0] = 1;

        for (i = 1, k = 0; i < f.n; i++) {
		if (f.mask && (f.data[DIVBPC(f.s) * f.n + i] >> MODBPC(f.s)) ^
                    (f.data[DIVBPC(f.s) * f.n + i - 1] >> MODBPC(f.s))) { k++; goto next; }
                for (j = DIVBPC(f.s) + (f.mask ? 1 : 0); j < f.c; j++)
                        if (f.data[j * f.n + i] != f.data[j * f.n + i - 1]) { k++; break; }
                next:
                f.h[k]++;
        }
}

void markmatchingrows(func f1, func f2, dim *n1, dim *n2, dim *hn) {

	register dim i1, i2, j1, j2;
	i1 = i2 = j1 = j2 = *n1 = *n2 = *hn = 0;
	register char cmp;

	while (i1 != f1.n && i2 != f2.n)
		if ((cmp = COMPARE(0, f1.data + i1, f2.data + i2, f1, f2)))
			if (cmp < 0) i1 += f1.h[j1++];
			else i2 += f2.h[j2++];
		else {
			//for (i = i1; i < i1 + f1.h[j1]; i++) SET(f1.rmask, i);
			//for (i = i2; i < i2 + f2.h[j2]; i++) SET(f2.rmask, i);
			SET(f1.hmask, j1);
			SET(f2.hmask, j2);
			(*n1) += f1.h[j1];
			(*n2) += f2.h[j2];
			i1 += f1.h[j1++];
			i2 += f2.h[j2++];
			(*hn)++;
		}
}

void removenonmatchingrows(func *f1, func *f2) {

	register dim i, i1, i2, j, j1, j2;
	i1 = i2 = j1 = j2 = 0;
	register char cmp;
	register func *f;
	register int k;

	while (i1 != f1->n && i2 != f2->n)
		if ((cmp = COMPARE(0, f1->data + i1, f2->data + i2, *f1, *f2))) {
			if (cmp < 0) f = f1, i = i1, j = j1;
			else f = f2, i = i2, j = j2;
			for (k = f->c - 1; k >= 0; k--)
			memmove(f->data + i + k * f->n, f->data + i + k * f->n + f->h[j], sizeof(chunk) * ((f->c - k) * (f->n - f->h[j]) - i));
			//printf("%u %u\n", i1, i2);
			//for (k = 0; k < f->c; k++) {
			//	memmove(f->data + i + (f->c - k - 1) * f->n, f->data + i + (f->c - k - 1) * f->n + f->h[j],
			//	sizeof(chunk) * (k * (f->n - f->h[j]) + f->n - i - f->h[j]));
			//	sizeof(chunk) * ((k + 1) * (f->n - f->h[j]) - i));
			//}
			f->n -= f->h[j];
			memmove(f->h + j, f->h + j + 1, sizeof(dim) * (f->hn - j - 1));
			(f->hn)--;
		}
		else {
			i1 += f1->h[j1++];
			i2 += f2->h[j2++];
		}

	if (i1 != f1->n) f = f1, i = i1, j = j1;
	else f = f2, i = i2, j = j2;
	for (k = f->c - 2; k >= 0; k--)
	memmove(f->data + i + k * f->n, f->data + (k + 1) * f->n, sizeof(chunk) * i * (f->c - k - 1));
	f->hn = j;
	f->n = i;
	f1->data = (chunk *)realloc(f1->data, sizeof(chunk) * f1->n * f1->c);
	f2->data = (chunk *)realloc(f2->data, sizeof(chunk) * f2->n * f2->c);
	f1->h = (dim *)realloc(f1->h, sizeof(dim) * f1->hn);
	f2->h = (dim *)realloc(f2->h, sizeof(dim) * f2->hn);
}
