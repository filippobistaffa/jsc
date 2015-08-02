#!/usr/bin/env bash

bpc=32

echo -e "switch (f->c) {"

for c in `seq 1 $1`; do

	echo -e "\tcase $c: {"
	echo -e "\t\tswitch (f->s) {"

	for s in `seq 1 $(( 32 * $1 - 1 ))`; do
		echo -en "\t\t\tcase $s: { "
		if [ $c = 1 ]
			then echo "cubsort<$s>(f->data, f->v, f->n); break; }"
			else echo "templatesort<chunk$c,$s>(f->data, f->v, f->n); break; }"
		fi
	done

	echo -e "\t\t\tdefault:;\n\t\t}\n\t\tbreak;\n\t}"
done

echo -e "\tdefault:;\n}"
