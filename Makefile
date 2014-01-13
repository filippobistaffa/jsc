.PHONY:
	
column:
	gcc -Wall -Ofast -march=native -funroll-loops -fopenmp columnmajor.c qsort.c marsenne.c crc32.c common.c jsc.c -o jsc

row:
	gcc -Wall -Ofast -march=native -funroll-loops -fopenmp -DROWMAJOR rowmajor.c marsenne.c crc32.c common.c jsc.c -o jsc

cuda:
	nvcc --compiler-options -Wall,-Ofast,-march=native,-funroll-loops,-fopenmp --ptxas-options=-v -arch=sm_30 columnmajor.c qsort.c marsenne.c crc32.c common.c *.cu -o jsc

run:
	./jsc
