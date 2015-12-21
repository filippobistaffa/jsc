#include "chunk.h"
#include <cub/util_allocator.cuh>
//#include <thrust/device_vector.h>
//#include <thrust/copy.h>
//#include <thrust/sort.h>

#ifdef PRINTTIME
static struct timeval t1, t2;
#endif

//#define NATIVESORT(F, I) (cubsort<chunk,I>(F))
#define NATIVESORT(F, I) (templatesort<chunk,I>(F))

using namespace cub;
CachingDeviceAllocator g_allocator(true);

template<typename T, bool I>
__attribute__((always_inline)) inline
void cubsort(const func *f) {

	TIMER_START(GREEN("Sort..."));
	DoubleBuffer<chunk> d_keys;
	DoubleBuffer<value> d_values;
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(chunk) * f->n));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(chunk) * f->n));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[0], sizeof(value) * f->n));
	CubDebugExit(g_allocator.DeviceAllocate((void**)&d_values.d_buffers[1], sizeof(value) * f->n));
	size_t tsn = 0;
	void *ts = NULL;
	CubDebugExit(DeviceRadixSort::SortPairs(ts, tsn, d_keys, d_values, f->n, I ? f->s : 0, I ? f->m : f->s));
	CubDebugExit(g_allocator.DeviceAllocate(&ts, tsn));
	CubDebugExit(cudaMemcpy(d_keys.d_buffers[d_keys.selector], f->data, sizeof(chunk) * f->n, cudaMemcpyHostToDevice));
	CubDebugExit(cudaMemcpy(d_values.d_buffers[d_values.selector], f->v, sizeof(value) * f->n, cudaMemcpyHostToDevice));
	CubDebugExit(DeviceRadixSort::SortPairs(ts, tsn, d_keys, d_values, f->n, I ? f->s : 0, I ? f->m : f->s));
	CubDebugExit(cudaMemcpy(f->data, d_keys.d_buffers[d_keys.selector], sizeof(chunk) * f->n, cudaMemcpyDeviceToHost));
	CubDebugExit(cudaMemcpy(f->v, d_values.d_buffers[d_values.selector], sizeof(value) * f->n, cudaMemcpyDeviceToHost));
	CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[0]));
	CubDebugExit(g_allocator.DeviceFree(d_keys.d_buffers[1]));
	CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[0]));
	CubDebugExit(g_allocator.DeviceFree(d_values.d_buffers[1]));
	CubDebugExit(g_allocator.DeviceFree(ts));
	TIMER_STOP;
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

#include "qsort.cpp"

template<typename T, bool I>
__attribute__((always_inline)) inline
void templatesort(const func *f) {

	//thrustsort<T,S>(casted, v, n);
	TIMER_START(YELLOW("Sort..."));
	qsort<T,I>(f);
	TIMER_STOP;
}

template<bool I = false>
__attribute__((always_inline)) inline
void sort(const func *f) {

	if (f->n < 2) return;
	assert(f->c <= 10);
	#include "switch.i"
}
