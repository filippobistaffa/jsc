#include "chunk.h"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

template<typename T, dim S>
struct compare { __host__ __device__ bool operator()(const T &a, const T &b) const {

	register const chunk *const ac = (chunk *)(&a);
	register const chunk *const bc = (chunk *)(&b);
	register int cmp = 0;

	/*if (DIVBPC(S)) for (dim i = 0; i < DIVBPC(S); i++) cmp += (CEILBPC(S) - i) * CMP(ac[i], bc[i]);
	if (MODBPC(S))*/ cmp += CMP(ac[DIVBPC(S)] & ((1ULL << S) - 1), bc[DIVBPC(S)] & ((1ULL << S) - 1));

	return cmp < 0;
} };

template<typename T, dim S>
void insertionsort(T *data, value *val, dim n) {

	register value v;
	register T d;

	for (dim i = 1; i < n; i++) {

		register int j = i - 1;
		d = data[i];
		v = val[i];

		for (; j >= 0 && COMPARE((chunk *)(&d), (chunk *)(data + j), S, (1ULL << S) - 1) < 0; j--) {
			data[j + 1] = data[j];
			val[j + 1] = val[j];
		}

		data[j + 1] = d;
		val[j + 1] = v;
	}
}

template<typename T, dim S>
__attribute__((always_inline)) inline
void templatesort(chunk *data, value *v, dim n) {

	register T *const casted = (T *)data;
	//insertionsort<T,S>(casted, v, n);
	thrust::device_vector<T> thdata(casted, casted + n);
	thrust::device_vector<value> thv(v, v + n);
	thrust::sort_by_key(thdata.begin(), thdata.end(), thv.begin(), compare<T,S>());
	thrust::copy(thdata.begin(), thdata.end(), casted);
	thrust::copy(thv.begin(), thv.end(), v);
}

__attribute__((always_inline)) inline
void sort(const func *f) {

	if (f->n < 2) return;
	assert(f->c == 1);
	assert(f->s <= 50);
	#include "switch.i"
}
