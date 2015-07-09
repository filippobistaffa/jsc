#include "jsc.h"

void randomdata(func *f) { // assumes BITSPERCHUNK == 64

	register dim i, j;

	for (i = 0; i < f->n; i++) {
		for (j = 0; j < DIVBPC(f->m); j++) f->data[j * f->n + i] = genrand64_int64();
		if (MODBPC(f->m)) f->data[DIVBPC(f->m) * f->n + i] = genrand64_int64() & ((1ULL << MODBPC(f->m)) - 1);
	}
}

#define WIDTH "2"
#define FORMAT "%" WIDTH "u"
#define BITFORMAT "%" WIDTH "zu"

void printrow(const func *f, dim i) {

	register dim j, k;

	for (j = 0; j < DIVBPC(f->m); j++)
		for (k = 0; k < BITSPERCHUNK; k++)
			if (!f->care[i] || ((f->care[i][j] >> k) & 1))
				printf(k & 1 ? BITFORMAT : WHITE(BITFORMAT), (f->data[j * f->n + i] >> k) & 1);
			else printf(k & 1 ? "%" WIDTH "s" : WHITE("%" WIDTH "s"), "*");

	for (k = 0; k < MODBPC(f->m); k++)
		if (!f->care[i] || ((f->care[i][j] >> k) & 1))
			printf(k & 1 ? BITFORMAT : WHITE(BITFORMAT), (f->data[DIVBPC(f->m) * f->n + i] >> k) & 1);
		else printf(k & 1 ? "%" WIDTH "s" : WHITE("%" WIDTH "s"), "*");

	printf(" = %u (%p)\n", f->v[i], f->care[i]);
}

void print(const func *f, const char *title, const chunk *s) {

	if (title) printf("%s\n", title);
	register dim i;

	for (i = 0; i < f->m; i++)
		printf(i & 1 ? FORMAT : WHITE(FORMAT), i);
	printf("\n");

	for (i = 0; i < f->m; i++)
		if (s) {
			if ((s[DIVBPC(i)] >> MODBPC(i)) & 1) printf(i & 1 ? DARKGREEN(FORMAT) : GREEN(FORMAT), f->vars[i]);
			else printf(i & 1 ? DARKRED(FORMAT) : RED(FORMAT), f->vars[i]);
		} else printf(i & 1 ? DARKCYAN(FORMAT) : CYAN(FORMAT), f->vars[i]);
	printf("\n");

	for (i = 0; i < f->n; i++) printrow(f, i);
}

#define SWAP(V, CARE, I, J, N) do { register chunk d = GET(V, I, N) ^ GET(V, J, N); (V)[DIVBPC(I) * (N)] ^= d << MODBPC(I); (V)[DIVBPC(J) * (N)] ^= d << MODBPC(J); \
				    if (CARE) { d = GET(CARE, I) ^ GET(CARE, J); (CARE)[DIVBPC(I)] ^= d << MODBPC(I); (CARE)[DIVBPC(J)] ^= d << MODBPC(J); } } while (0)

void shared2least(func *f, chunk* m) {

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
		for (i = 0; i < f->n; i++) SWAP(f->data + i, f->care[i], x, y, f->n);
		o[DIVBPC(x)] ^= 1ULL << MODBPC(x);
		a[DIVBPC(y)] ^= 1ULL << MODBPC(y);
	} while (--n);
}

void reordershared(func *f, id *vars) {

	if (!memcmp(f->vars, vars, sizeof(id) * f->s)) return;

	register dim i, j;
	register const dim ds = DIVBPC(f->s);
	register const dim cs = CEIL(f->s, BITSPERCHUNK);
	id *v = (id *)malloc(sizeof(id) * MAXVAR);
	for (i = 0; i < f->s; i++) v[vars[i]] = i;

	#pragma omp parallel for private(i)
	for (i = 0; i < f->n; i++) {
		chunk s[cs];
		memset(s, 0, sizeof(chunk) * cs);
		for (j = 0; j < f->s; j++) if GET(f->data + i, j, f->n) SET(s, v[f->vars[j]]);
		for (j = 0; j < ds; j++) f->data[j * f->n + i] = s[j];

		if (f->mask) {
			f->data[ds * f->n + i] &= ~f->mask;
			f->data[ds * f->n + i] |= s[ds];
		}

		if (f->care[i]) {
			chunk c[cs];
			memset(c, 0, sizeof(chunk) * cs);
			for (j = 0; j < f->s; j++) if GET(f->care[i], j) SET(c, v[f->vars[j]]);
			for (j = 0; j < ds; j++) f->care[i][j] = c[j];
			if (f->mask) {
				f->care[i][ds] &= ~f->mask;
				f->care[i][ds] |= c[ds];
			}
		}
	}

	memcpy(f->vars, vars, sizeof(id) * f->s);
	free(v);
}

dim uniquecombinations(const func *f) {

	register dim i, u = 1;

	for (i = 1; i < f->n; i++)
		if (COMPARE(f->data + i, f->data + i - 1, f, f)) u++;

	return u;
}

void histogram(const func *f) {

	register dim i, k;
	f->h[0] = 1;

	for (i = 1, k = 0; i < f->n; i++) {
		if (COMPARE(f->data + i, f->data + i - 1, f, f)) k++;
		f->h[k]++;
	}
}

dim intuniquecombinations(const func *f) {

	register dim i, u = 1;

	for (i = 1; i < f->n; i++)
		if (!INTERSECT(f, i, f, i - 1)) u++;

	return u;
}

void inthistogram(const func *f) {

	register dim i, k;
	f->h[0] = 1;

	for (i = 1, k = 0; i < f->n; i++) {
		if (!INTERSECT(f, i, f, i - 1)) k++;
		f->h[k]++;
	}
}
