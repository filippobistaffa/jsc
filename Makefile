.PHONY:
	
column:
	g++ -g -DJSCMAIN -DCOLOURS -DPRINTSIZE -DPRINTTIME -std=gnu++11 -Wall -Ofast -march=native -funroll-loops -fopenmp columnmajor.c marsenne.c crc32.c common.c jsc.c -o jsc

cuda:
	nvcc -DJSCMAIN --compiler-options -Wall,-Ofast,-march=native,-funroll-loops,-fopenmp --ptxas-options=-v -arch=sm_30 columnmajor.c qsort.c marsenne.c crc32.c common.c *.cu -o jsc

run:
	./jsc
