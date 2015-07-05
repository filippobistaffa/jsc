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

// Compare row at address A in function F with row at address B in function G

#define DATACOMPARE(A, B, F, G) ({ register dim _i; register char cmp = 0; for (_i = 0; _i < DIVBPC((F)->s); _i++) \
				   if ((cmp = CMP((A)[_i * (F)->n], (B)[_i * (G)->n]))) break; \
                                   if (!cmp) cmp = ((F)->mask ? CMP((F)->mask & (A)[(DIVBPC((F)->s)) * (F)->n], (F)->mask & (B)[(DIVBPC((F)->s)) * (G)->n]) : 0); cmp; })

#define CARECOMPARE(A, B, S, MASK) ({ register const dim ds = DIVBPC(S); register char cmp = memcmp(A, B, sizeof(chunk) * ds); \
				      if (!cmp && MASK) cmp = CMP((A)[ds] & MASK, (B)[ds] & MASK); cmp; })

#define COMPARE(A, B, F, G) ({ register chunk *const ca = (F)->care[(A) - (F)->data]; register chunk *const cb = (G)->care[(B) - (G)->data]; \
			       register char cmp = DATACOMPARE(A, B, F, G); if (!cmp) { if (ca && cb) cmp = CARECOMPARE(ca, cb, (F)->s, (F)->mask); \
			       else cmp = CMP(ca, cb); } cmp; }) 

#define _SWAP(A, B, N, C) do { if ((A) != (B)) { \
	register chunk *__a = (A), *__b = (B), *__bp = base_ptr; \
	register chunk *__c = *(f->care + (__a - __bp)); \
	*(f->care + (__a - __bp)) = *(f->care + (__b - __bp)); \
	*(f->care + (__b - __bp)) = __c; \
	register value __t = *(f->v + (__a - __bp)); \
	*(f->v + (__a - __bp)) = *(f->v + (__b - __bp)); \
	*(f->v + (__b - __bp)) = __t; \
	register dim i; \
	for (i = 0; i < (C); i++) { \
		register chunk __tmp = *(__a + i * (N)); \
		*(__a + i * (N)) = *(__b + i * (N)); \
		*(__b + i * (N)) = __tmp; \
	} \
} } while (0)

/*template <char type = 0>
__attribute__((always_inline)) inline
void equalconsecutive(func f) {

	register chunk *base_ptr = f->data;
	register dim i = 0;

	while (i < f->n - 1) {

		register dim j = i + 1, k = 1;

		while (j < f->n) {
			if (!COMPARE(f->data + i, f->data + j, f, f)) { // equals according to COMPARE
				_SWAP(f->data + j, f->data + i + k, f->n, f->c);
				k++;
			}
			j++;
		}

		i += k;
	}
}*/

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

// Stack node declarations used to store unfulfilled partition obligations.
typedef struct {
	chunk *lo;
	chunk *hi;
} stack_node;

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

   3. Only quicksorts f->n / MAX_THRESH partitions, leaving
      insertion sort to order the MAX_THRESH items within each partition.
      This is a big win, since insertion sort is faster for small, mostly
      sorted array segments.

   4. The larger of the two sub-partitions is always pushed onto the
      stack first, with the algorithm then concentrating on the
      smaller partition.  This *guarantees* no more than log (f->n)
      stack size is needed (actually O(1) in this case)!  */

__attribute__((always_inline)) inline
void sort(func *f) {

	register chunk *base_ptr = f->data;
	const size_t max_thresh = MAX_THRESH;

	if (f->n > MAX_THRESH) {

		chunk *lo = base_ptr;
		chunk *hi = lo + f->n - 1;
		stack_node stack[STACK_SIZE];
		stack_node *top = stack;
		PUSH(NULL, NULL);

		while (STACK_NOT_EMPTY) {

			chunk *left_ptr;
			chunk *right_ptr;

			/* Select median value from among LO, MID, and HI. Rearrange
			LO and HI so the three values are sorted. This lowers the
			probability of picking a pathological pivot value and
			skips a comparison for both the LEFT_PTR and RIGHT_PTR in
			the while loops. */

			chunk *mid = lo + ((hi - lo) >> 1);

			if (COMPARE(mid, lo, f, f) < 0) _SWAP(mid, lo, f->n, f->c);
			if (COMPARE(hi, mid, f, f) < 0) _SWAP(mid, hi, f->n, f->c);
			else goto jump_over;
			if (COMPARE(mid, lo, f, f) < 0) _SWAP(mid, lo, f->n, f->c);
			jump_over:;

			left_ptr = lo + 1;
			right_ptr = hi - 1;

			/* Here's the famous "collapse the walls" section of quicksort.
			Gotta like those tight inner loops! They are the main reason
			that this algorithm runs much faster than others. */
			do {
				while (COMPARE(left_ptr, mid, f, f) < 0) left_ptr++;
				while (COMPARE(mid, right_ptr, f, f) < 0) right_ptr--;

				if (left_ptr < right_ptr) {
					_SWAP(left_ptr, right_ptr, f->n, f->c);
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
		chunk *const end_ptr = base_ptr + f->n - 1;
		chunk *tmp_ptr = base_ptr;
		chunk *thresh = MIN(end_ptr, base_ptr + max_thresh);
		chunk *run_ptr;

		/* Find smallest element in first threshold and place it at the
		array's beginning.  This is the smallest array element,
		and the operation speeds up insertion sort's inner loop. */

		for (run_ptr = tmp_ptr + 1; run_ptr <= thresh; run_ptr++)
			if (COMPARE(run_ptr, tmp_ptr, f, f) < 0) tmp_ptr = run_ptr;

		if (tmp_ptr != base_ptr) _SWAP(tmp_ptr, base_ptr, f->n, f->c);

		// Insertion sort, running from left-hand-side up to right-hand-side.
		run_ptr = base_ptr + 1; // current element (first of unordered)

		while ((run_ptr += 1) <= end_ptr) {

			tmp_ptr = run_ptr - 1;
			while (COMPARE(run_ptr, tmp_ptr, f, f) < 0) tmp_ptr--;
			tmp_ptr++; // current element's final position

			if (tmp_ptr != run_ptr) {
				register dim i;
				for (i = 0; i < f->c; i++) {
					register chunk trav = *(run_ptr + i * f->n);
					memmove(tmp_ptr + i * f->n + 1, tmp_ptr + i * f->n, sizeof(chunk) * (run_ptr - tmp_ptr));
					*(tmp_ptr + i * f->n) = trav;
				}
				register value tmp = *(f->v + (run_ptr - base_ptr));
				memmove(f->v + (tmp_ptr - base_ptr) + 1, f->v + (tmp_ptr - base_ptr), sizeof(value) * (run_ptr - tmp_ptr));
				*(f->v + (tmp_ptr - base_ptr)) = tmp;
				register chunk *tmpcare = *(f->care + (run_ptr - base_ptr));
				memmove(f->care + (tmp_ptr - base_ptr) + 1, f->care + (tmp_ptr - base_ptr), sizeof(value) * (run_ptr - tmp_ptr));
				*(f->care + (tmp_ptr - base_ptr)) = tmpcare;
			}
		}
	}
}
