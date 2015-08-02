#include "jsc.h"

/*void randomdata(func *f) { // assumes BITSPERCHUNK == 64

	register dim i, j;

	for (i = 0; i < f->n; i++) {
		for (j = 0; j < DIVBPC(f->m); j++) f->data[j * f->n + i] = genrand64_int64();
		if (MODBPC(f->m)) f->data[DIVBPC(f->m) * f->n + i] = genrand64_int64() & ((ONE << MODBPC(f->m)) - 1);
	}
}*/

#define WIDTH "2"
#define FORMAT "%" WIDTH "u"
#define BITFORMAT "%" WIDTH "zu"
#include <iostream>

void printrow(const func *f, dim i) {

	register dim j;

	for (j = 0; j < f->m; j++)
		printf(j & 1 ? BITFORMAT : WHITE(BITFORMAT), GET(DATA(f, i), j));

	std::cout << " = " << f->v[i] << std::endl;
}

void print(const func *f, const char *title, const chunk *s) {

	if (title) printf("%s\n", title);
	register dim i;

	for (i = 0; i < f->m; i++) printf(i & 1 ? FORMAT : WHITE(FORMAT), i);
	printf("\n");

	for (i = 0; i < f->m; i++)
		if (s) {
			if ((s[DIVBPC(i)] >> MODBPC(i)) & 1) printf(i & 1 ? DARKGREEN(FORMAT) : GREEN(FORMAT), f->vars[i]);
			else printf(i & 1 ? DARKRED(FORMAT) : RED(FORMAT), f->vars[i]);
		} else printf(i & 1 ? DARKCYAN(FORMAT) : CYAN(FORMAT), f->vars[i]);
	printf("\n");

	for (i = 0; i < f->n; i++) printrow(f, i);
}

#define SWAP(V, I, J) do { register chunk d = GET(V, I) ^ GET(V, J); (V)[DIVBPC(I)] ^= d << MODBPC(I); (V)[DIVBPC(J)] ^= d << MODBPC(J); } while (0)

void shared2least(const func *f, chunk* m) {

	register dim x, y, i, n = 0;
	register id t;
	chunk s[f->c], a[f->c], o[f->c];
	memset(s, 0, sizeof(chunk) * f->c);

	for (i = 0; i < DIVBPC(f->s); i++) s[i] = ~(0ULL);
	if (MODBPC(f->s)) s[DIVBPC(f->s)] = f->mask;

	for (i = 0; i < f->c; i++) {
		a[i] = s[i] & ~m[i];
		o[i] = m[i] & ~s[i];
		n += __builtin_popcountll(o[i]);
	}

	if (!n) return;
	memcpy(m, s, sizeof(chunk) * f->c);

	do {
		i = x = y = 0;
		while (!o[i++]) x += BITSPERCHUNK;
		x += __builtin_ctzll(o[i - 1]);
		i = 0;
		while (!a[i++]) y += BITSPERCHUNK;
		y += __builtin_ctzll(a[i - 1]);
		t = f->vars[x];
		f->vars[x] = f->vars[y];
		f->vars[y] = t;
		#pragma omp parallel for private(i)
		for (i = 0; i < f->n; i++) SWAP(DATA(f, i), x, y);
		o[DIVBPC(x)] ^= ONE << MODBPC(x);
		a[DIVBPC(y)] ^= ONE << MODBPC(y);
	} while (--n);
}

void reordershared(const func *f, id *vars) {

	if (!memcmp(f->vars, vars, sizeof(id) * f->s)) return;

	register dim i;
	register const dim cs = CEILBPC(f->s);
	id *v = (id *)malloc(sizeof(id) * MAXVAR);
	for (i = 0; i < f->s; i++) v[vars[i]] = i;

	#pragma omp parallel for private(i)
	for (i = 0; i < f->n; i++) {
		chunk s[cs];
		register dim j;
		memset(s, 0, sizeof(chunk) * cs);
		for (j = 0; j < f->s; j++) if GET(DATA(f, i), j) SET(s, v[f->vars[j]]);
		for (; j < f->c * BITSPERCHUNK; j++) if GET(DATA(f, i), j) SET(s, j);
		memcpy(DATA(f, i), s, sizeof(chunk) * cs);
	}

	memcpy(f->vars, vars, sizeof(id) * f->s);
	free(v);
}

dim uniquecombinations(const func *f, dim idx) {

	if (!f->n) return 0;

	register dim u = 1;

	for (dim i = 1 + idx; i < f->n; i++)
		if (COMPARE(DATA(f, i - 1), DATA(f, i), f->s, f->mask)) u++;

	return u;
}

void histogram(const func *f, dim idx) {

	if (f->n - idx && f->h) {

		f->h[0] = 1;

		for (dim i = 1 + idx, k = 0; i < f->n; i++) {
			if (COMPARE(DATA(f, i - 1), DATA(f, i), f->s, f->mask)) k++;
			f->h[k]++;
		}
	}
}

void markmatchingrows(const func *f1, const func *f2, dim *n1, dim *n2, dim *hn) {

	register dim i1, i2, j1, j2;
	i1 = i2 = j1 = j2 = *n1 = *n2 = *hn = 0;
	register char cmp;

	while (i1 != f1->n && i2 != f2->n)
		if ((cmp = COMPARE(DATA(f1, i1), DATA(f2, i2), f1->s, f1->mask)))
			if (cmp < 0) i1 += f1->h[j1++];
			else i2 += f2->h[j2++];
		else {
			//for (i = i1; i < i1 + f1->h[j1]; i++) SET(f1->rmask, i);
			//for (i = i2; i < i2 + f2->h[j2]; i++) SET(f2->rmask, i);
			SET(f1->hmask, j1);
			SET(f2->hmask, j2);
			(*n1) += f1->h[j1];
			(*n2) += f2->h[j2];
			i1 += f1->h[j1++];
			i2 += f2->h[j2++];
			(*hn)++;
		}
}

void copymatchingrows(func *f1, func *f2, dim n1, dim n2, dim hn) {

        register dim i1, i2, i3, i4, j1, j2, j3, j4;
        i1 = i2 = i3 = i4 = j1 = j2 = j3 = j4 = 0;
	register char cmp;

        chunk *d1 = (chunk *)malloc(sizeof(chunk) * n1 * f1->c);
        chunk *d2 = (chunk *)malloc(sizeof(chunk) * n2 * f2->c);
	value *v1 = (value *)malloc(sizeof(value) * n1);
	value *v2 = (value *)malloc(sizeof(value) * n2);
        dim *h1 = (dim *)malloc(sizeof(dim) * hn);
        dim *h2 = (dim *)malloc(sizeof(dim) * hn);

	// i1 and i2: current row in f1->data and f2->data, f1->v and f2->v
	// i3 and i4: current row in d1 and d2, v1 and v2

        while (i1 != f1->n && i2 != f2->n)
                if ((cmp = GET(f1->hmask, j1)) & GET(f2->hmask, j2)) {
                	memcpy(d1 + i3 * f1->c, DATA(f1, i1), sizeof(chunk) * f1->h[j1] * f1->c);
                	memcpy(d2 + i4 * f2->c, DATA(f2, i2), sizeof(chunk) * f2->h[j2] * f2->c);
			memcpy(v1 + i3, f1->v + i1, sizeof(value) * f1->h[j1]);
			memcpy(v2 + i4, f2->v + i2, sizeof(value) * f2->h[j2]);
                	h1[j3++] = f1->h[j1];
                	h2[j4++] = f2->h[j2];
                        i1 += f1->h[j1];
                        i2 += f2->h[j2];
                        i3 += f1->h[j1++];
                        i4 += f2->h[j2++];
		} else
			if (cmp) i2 += f2->h[j2++];
			else i1 += f1->h[j1++];

	free(f1->data); f1->data = d1;
	free(f2->data); f2->data = d2;
	free(f1->v); f1->v = v1;
	free(f2->v); f2->v = v2;
	free(f1->h); f1->h = h1;
	free(f2->h); f2->h = h2;
	f1->hn = f2->hn = hn;
	f1->n = n1;
	f2->n = n2;
}
