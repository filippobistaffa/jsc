template <bool def = false>
__attribute__((always_inline)) inline
void copymatchingrows(func *f1, func *f2, dim n1, dim n2, dim hn, func *fn1 = NULL, func *fn2 = NULL) {

        register dim i, i1, i2, i3, i4, j1, j2, j3, j4;
        i1 = i2 = i3 = i4 = j1 = j2 = j3 = j4 = 0;
	register dim i5, i6, j5, j6;
	i5 = i6 = j5 = j6 = 0;
	register char cmp1, cmp2;

        chunk *d1 = (chunk *)malloc(sizeof(chunk) * n1 * f1->c);
        chunk *d2 = (chunk *)malloc(sizeof(chunk) * n2 * f2->c);
	value *v1 = (value *)malloc(sizeof(value) * n1);
	value *v2 = (value *)malloc(sizeof(value) * n2);
        dim *h1 = (dim *)malloc(sizeof(dim) * hn);
        dim *h2 = (dim *)malloc(sizeof(dim) * hn);

	// i1 and i2: current row in f1->data and f2->data, f1->v and f2->v
	// i3 and i4: current row in d1 and d2, v1 and v2

        while (i1 != f1->n || i2 != f2->n) {

		cmp1 = cmp2 = 0;

                if ((i1 != f1->n || !f1->h[j1]) && (i2 != f2->n || !f2->h[j2]) && (cmp1 = GET(f1->hmask, j1)) & (cmp2 = GET(f2->hmask, j2))) {

                	for (i = 0; i < f1->c; i++)
                		memcpy(d1 + i3 + i * n1, f1->data + i1 + i * f1->n, sizeof(chunk) * f1->h[j1]);

                	for (i = 0; i < f2->c; i++)
                		memcpy(d2 + i4 + i * n2, f2->data + i2 + i * f2->n, sizeof(chunk) * f2->h[j2]);

			memcpy(v1 + i3, f1->v + i1, sizeof(value) * f1->h[j1]);
			memcpy(v2 + i4, f2->v + i2, sizeof(value) * f2->h[j2]);
                	h1[j3++] = f1->h[j1];
                	h2[j4++] = f2->h[j2];
                        i1 += f1->h[j1];
                        i2 += f2->h[j2];
                        i3 += f1->h[j1++];
                        i4 += f2->h[j2++];

		} else {

			if (i1 != f1->n && !cmp1) {
                        	if (def) {
		                	for (i = 0; i < fn1->c; i++)
		        			memcpy(fn1->data + i5 + i * fn1->n, f1->data + i1 + i * f1->n, sizeof(chunk) * f1->h[j1]);
					memcpy(fn1->v + i5, f1->v + i1, sizeof(value) * f1->h[j1]);
		        		fn1->h[j5++] = f1->h[j1];
					i5 += f1->h[j1];
				}
                        	i1 += f1->h[j1++];
                        }

                        if (i2 != f2->n && !cmp2) {
				if (def) {
		                	for (i = 0; i < fn2->c; i++)
		        			memcpy(fn2->data + i6 + i * fn2->n, f2->data + i2 + i * f2->n, sizeof(chunk) * f2->h[j2]);
					memcpy(fn2->v + i6, f2->v + i2, sizeof(value) * f2->h[j2]);
		        		fn2->h[j6++] = f2->h[j2];
					i6 += f2->h[j2];
				}
                        	i2 += f2->h[j2++];
                        }
		}
	}

	//print(fn1);
	//print(fn2);
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
