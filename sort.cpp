#include "chunk.h"
#include <cub/util_allocator.cuh>
//#include <thrust/device_vector.h>
//#include <thrust/copy.h>
//#include <thrust/sort.h>

#define NATIVESORT(F, S, I) (cubsort<chunk,S,I>(F))
//#define NATIVESORT(F, S, I) (qsort<chunk,S,I>(F))
//#define NATIVESORT(DATA, VALUE, N, S) thrustsort<chunk,S>(DATA, VALUE, N);
//#define NATIVESORT(DATA, VALUE, N, S) qsort<chunk,S>(DATA, VALUE, N);

template<typename T, dim S>
struct compare { __host__ __device__ bool operator()(const T &a, const T &b) const {

	register const chunk *const ac = (chunk *)(&a);
	register const chunk *const bc = (chunk *)(&b);
	register int cmp = 0;

	/*if (DIVBPC(S)) for (dim i = 0; i < DIVBPC(S); i++) cmp += (CEILBPC(S) - i) * CMP(ac[i], bc[i]);
	if (MODBPC(S))*/ cmp += CMP(ac[DIVBPC(S)] & ((ONE << S) - 1), bc[DIVBPC(S)] & ((ONE << S) - 1));

	return cmp < 0;
} };

using namespace cub;
CachingDeviceAllocator g_allocator(true);

template<typename T, dim S, bool I>
__attribute__((always_inline)) inline
void cubsort(const func *f) {

	DoubleBuffer<chunk> d_keys;
	DoubleBuffer<value> d_values;
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(chunk) * f->n));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(chunk) * f->n));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[0], sizeof(value) * f->n));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[1], sizeof(value) * f->n));
	size_t tsn = 0;
	void *ts = NULL;
	CubDebugExit(DeviceRadixSort::SortPairs(ts, tsn, d_keys, d_values, f->n, I ? S : 0, I ? f->m : S));
	CubDebugExit(g_allocator.DeviceAllocate(&ts, tsn));
	CubDebugExit(cudaMemcpy(d_keys.d_buffers[d_keys.selector], f->data, sizeof(chunk) * f->n, cudaMemcpyHostToDevice));
	CubDebugExit(cudaMemcpy(d_values.d_buffers[d_values.selector], f->v, sizeof(value) * f->n, cudaMemcpyHostToDevice));
	CubDebugExit(DeviceRadixSort::SortPairs(ts, tsn, d_keys, d_values, f->n, I ? S : 0, I ? f->m : S));
	CubDebugExit(cudaMemcpy(f->data, d_keys.d_buffers[d_keys.selector], sizeof(chunk) * f->n, cudaMemcpyDeviceToHost));
	CubDebugExit(cudaMemcpy(f->v, d_values.d_buffers[d_values.selector], sizeof(value) * f->n, cudaMemcpyDeviceToHost));
	CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[0]));
	CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[1]));
	CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[0]));
	CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[1]));
	CubDebugExit(g_allocator.DeviceFree(ts));
}

template<typename T, dim S>
__attribute__((always_inline)) inline
void thrustsort(T *casted, value *v, dim n) {

	//thrust::device_vector<T> thdata(casted, casted + n);
	//thrust::device_vector<value> thv(v, v + n);
	//thrust::sort_by_key(thdata.begin(), thdata.end(), thv.begin(), compare<T,S>());
	//thrust::copy(thdata.begin(), thdata.end(), casted);
	//thrust::copy(thv.begin(), thv.end(), v);
}

template<typename T, dim S>
__attribute__((always_inline)) inline
void merge(T *casted, value *v, dim mid, dim n) {

	T *tmpdata = (T *)malloc(sizeof(T) * n);
	value *tmpv = (value *)malloc(sizeof(value) * n);

	register dim i = 0, j = 0, k = 0;
	register T *const leftc = casted;
	register T *const rightc = casted + mid;
	register value *const leftv = v;
	register value *const rightv = v + mid;
	register const dim a = mid;
	register const dim b = n - mid;

	while (i < a && j < b) {

		if (compare<T,S>()(leftc[i], rightc[j])) {
			tmpdata[k] = leftc[i];
			tmpv[k] = leftv[i];
			k++, i++;
		} else {
			tmpdata[k] = rightc[j];
			tmpv[k] = rightv[j];
			k++, j++;
		}
	}

	if (i == a) {
		memcpy(tmpdata + k, rightc + j, sizeof(T) * (b - j));
		memcpy(tmpv + k, rightv + j, sizeof(value) * (b - j));
	} else {
		memcpy(tmpdata + k, leftc + i, sizeof(T) * (a - i));
		memcpy(tmpv + k, leftv + i, sizeof(value) * (a - i));
	}

	memcpy(casted, tmpdata, sizeof(T) * n);
	memcpy(v, tmpv, sizeof(value) * n);
	free(tmpdata);
	free(tmpv);
}

template<typename T, dim S>
inline void mergesort(T *casted, value *v, dim n) {

	//printf(RED("Table size = %zu bytes\n"), (sizeof(T) + sizeof(value)) * n);

	if ((sizeof(T) + sizeof(value)) * n > (GLOBALSIZE - GLOBALMARGIN) / 2) {

		register const dim mid = n / 2;
		printf(MAGENTA("Sorting left\n"));
		mergesort<T,S>(casted, v, mid);
		printf(MAGENTA("Sorting right\n"));
		mergesort<T,S>(casted + mid, v + mid, n - mid);
		printf(MAGENTA("Merging...\n"));
		merge<T,S>(casted, v, mid, n);

	} else thrustsort<T,S>(casted, v, n);
}

#include "qsort.cpp"

template<typename T, dim S, bool I>
__attribute__((always_inline)) inline
void templatesort(const func *f) {

	//mergesort<T,S>(casted, v, n);
	//thrustsort<T,S>(casted, v, n);
	qsort<T,S,I>(f);
}

template<bool I = false>
__attribute__((always_inline)) inline
void sort(const func *f) {

	if (f->n < 2) return;
	assert(f->c <= 2);
	#include "switch.i"
}
