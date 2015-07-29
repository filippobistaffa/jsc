#!/usr/bin/env bash

echo -e "switch (f->c) {"

for c in `seq 1 $1`; do

	echo -e "\tcase $c: {"
	echo -e "\t\tswitch (f->s) {"

	for s in `seq 1 $2`; do
		echo -e "\t\t\tcase $s: { templatesort<chunk$c,$s>(f->data, f->v, f->n); break; }"
	done

	echo -e "\t\t\tdefault:;\n\t\t}\n\t\tbreak;\n\t}"
done

echo -e "\tdefault:;\n}"
