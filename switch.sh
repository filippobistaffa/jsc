#!/usr/bin/env bash

echo -e "switch (f->c) {"

for c in `seq 1 $1`; do

	echo -en "\tcase $c: { "
	if [ $c = 1 ]
		then echo "NATIVESORT(f, I); break; }"
		else echo "templatesort<chunk$c,I>(f); break; }"
	fi
done

echo -e "\tdefault:;\n}"
