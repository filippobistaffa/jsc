/* Copyright (C) 1991-2013 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Douglas C. Schmidt (schmidt@ics.uci.edu).

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */

/* If you consider tuning this algorithm, you should consult first:
   Engineering a sort function; Jon Bentley and M. Douglas McIlroy;
   Software - Practice and Experience; Vol. 23 (11), 1249-1265, 1993.  */

#define QSWAP(A, B) do { register T td = *(A); *(A) = *(B); *(B) = td; \
			 register value tv = *(v + ((A) - base_ptr)); \
			 *(v + ((A) - base_ptr)) = *(v + ((B) - base_ptr)); \
			 *(v + ((B) - base_ptr)) = tv; } while (0)

// Discontinue quicksort algorithm when partition gets below this size.
// This particular magic number was chosen to work best on a Sun 4/260.
#define MAX_THRESH 4

/*
The next 4 #defines implement a very fast in-line stack abstraction.
The stack needs log (total_elements) entries (we could even subtract
log(MAX_THRESH)).
*/
#define STACK_SIZE	(64 * sizeof(size_t))
#define PUSH(low, high)	((void) ((top->lo = (low)), (top->hi = (high)), ++top))
#define	POP(low, high)	((void) (--top, (low = top->lo), (high = top->hi)))
#define	STACK_NOT_EMPTY	(stack < top)

/* Order size using quicksort.  This implementation incorporates
   four optimizations discussed in Sedgewick:

   1. Non-recursive, using an explicit stack of pointer that store the
      next array partition to sort.  To save time, this maximum amount
      of space required to store an array of SIZE_MAX is allocated on the
      stack.  Assuming a 32-bit (64 bit) integer for size_t, this needs
      only 32 * sizeof(stack_node) == 256 bytes (for 64 bit: 1024 bytes).
      Pretty cheap, actually.

   2. Chose the pivot element using a median-of-three decision tree.
      This reduces the probability of selecting a bad pivot value and
      eliminates certain extraneous comparisons.

   3. Only quicksorts n / MAX_THRESH partitions, leaving
      insertion sort to order the MAX_THRESH items within each partition.
      This is a big win, since insertion sort is faster for small, mostly
      sorted array segments.

   4. The larger of the two sub-partitions is always pushed onto the
      stack first, with the algorithm then concentrating on the
      smaller partition.  This *guarantees* no more than log (n)
      stack size is needed (actually O(1) in this case)!  */

#define CHUNK(X) ((chunk *)(X))

template<typename T, dim S>
__attribute__((always_inline)) inline
void qsort(T *data, value *v, dim n) {

	// Stack node declarations used to store unfulfilled partition obligations.
	typedef struct {
		T *lo;
		T *hi;
	} stack_node;

	register T *const base_ptr = data;
	const size_t max_thresh = MAX_THRESH;

	if (n > MAX_THRESH) {

		T *lo = base_ptr;
		T *hi = lo + n - 1;
		stack_node stack[STACK_SIZE];
		stack_node *top = stack;
		PUSH(NULL, NULL);

		while (STACK_NOT_EMPTY) {

			T *left_ptr;
			T *right_ptr;

			/* Select median value from among LO, MID, and HI. Rearrange
			LO and HI so the three values are sorted. This lowers the
			probability of picking a pathological pivot value and
			skips a comparison for both the LEFT_PTR and RIGHT_PTR in
			the while loops. */

			T *mid = lo + ((hi - lo) >> 1);

			if (COMPARE(CHUNK(mid), CHUNK(lo), S, (ONE << S) - 1) < 0) QSWAP(mid, lo);
			if (COMPARE(CHUNK(hi), CHUNK(mid), S, (ONE << S) - 1) < 0) QSWAP(mid, hi);
			else goto jump_over;
			if (COMPARE(CHUNK(mid), CHUNK(lo), S, (ONE << S) - 1) < 0) QSWAP(mid, lo);
			jump_over:;

			left_ptr = lo + 1;
			right_ptr = hi - 1;

			/* Here's the famous "collapse the walls" section of quicksort.
			Gotta like those tight inner loops! They are the main reason
			that this algorithm runs much faster than others. */
			do {
				while (COMPARE(CHUNK(left_ptr), CHUNK(mid), S, (ONE << S) - 1) < 0) left_ptr++;
				while (COMPARE(CHUNK(mid), CHUNK(right_ptr), S, (ONE << S) - 1) < 0) right_ptr--;

				if (left_ptr < right_ptr) {
					QSWAP(left_ptr, right_ptr);
					if (mid == left_ptr) mid = right_ptr;
					else if (mid == right_ptr) mid = left_ptr;
					left_ptr++;
					right_ptr--;
				}
				else if (left_ptr == right_ptr) {
					left_ptr++;
					right_ptr--;
					break;
				}
			}
			while (left_ptr <= right_ptr);

			/* Set up pointers for next iteration. First determine whether
			left and right partitions are below the threshold size. If so,
			ignore one or both. Otherwise, push the lfer partition's
			bounds on the stack and continue sorting the smaller one. */

			if ((size_t) (right_ptr - lo) <= max_thresh) {
				if ((size_t) (hi - left_ptr) <= max_thresh) POP(lo, hi); // Ignore both small partitions
				else lo = left_ptr; // Ignore small left partition
			}
			else if ((size_t) (hi - left_ptr) <= max_thresh) hi = right_ptr; // Ignore small right partition
			else if ((right_ptr - lo) > (hi - left_ptr)) {
				/* Push lfer left partition indices. */
				PUSH(lo, right_ptr);
				lo = left_ptr;
			}
			else {
				/* Push lfer right partition indices. */
				PUSH(left_ptr, hi);
				hi = right_ptr;
			}
		}
	}

	/* Once the BASE_PTR array is partially sorted by quicksort the rest
	is completely sorted using insertion sort, since this is efficient
	for partitions below MAX_THRESH size. BASE_PTR points to the beginning
	of the array to sort, and END_PTR points at the very last element in
	the array (*not* one beyond it!). */

	{
		T *const end_ptr = base_ptr + n - 1;
		T *tmp_ptr = base_ptr;
		T *thresh = MIN(end_ptr, base_ptr + max_thresh);
		T *run_ptr;

		/* Find smallest element in first threshold and place it at the
		array's beginning.  This is the smallest array element,
		and the operation speeds up insertion sort's inner loop. */

		for (run_ptr = tmp_ptr + 1; run_ptr <= thresh; run_ptr++)
			if (COMPARE(CHUNK(run_ptr), CHUNK(tmp_ptr), S, (ONE << S) - 1) < 0) tmp_ptr = run_ptr;

		if (tmp_ptr != base_ptr) QSWAP(tmp_ptr, base_ptr);

		// Insertion sort, running from left-hand-side up to right-hand-side.
		run_ptr = base_ptr + 1; // current element (first of unordered)

		while ((run_ptr += 1) <= end_ptr) {

			tmp_ptr = run_ptr - 1;
			while (COMPARE(CHUNK(run_ptr), CHUNK(tmp_ptr), S, (ONE << S) - 1) < 0) tmp_ptr--;
			tmp_ptr++; // current element's final position

			if (tmp_ptr != run_ptr) {
				register T td = *(data + (run_ptr - base_ptr));
				memmove(data + (tmp_ptr - base_ptr) + 1, data + (tmp_ptr - base_ptr), sizeof(T) * (run_ptr - tmp_ptr));
				*(data + (tmp_ptr - base_ptr)) = td;
				register value tv = *(v + (run_ptr - base_ptr));
				memmove(v + (tmp_ptr - base_ptr) + 1, v + (tmp_ptr - base_ptr), sizeof(value) * (run_ptr - tmp_ptr));
				*(v + (tmp_ptr - base_ptr)) = tv;
			}
		}
	}
}
