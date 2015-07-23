.PHONY:
	
column:
	g++ -g -DJSCMAIN -DCOLOURS -DPRINTSIZE -DPRINTTIME -std=gnu++11 -Wall -O0 -march=native -fopenmp columnmajor.c marsenne.c crc32.c common.c jsc.c -o jsc

cuda:
	nvcc --x c++ --std c++11 -DJSCMAIN --compiler-options -Wall,-O0,-fopenmp -arch=sm_30 columnmajor.c marsenne.c crc32.c common.c *.cu -o jsc

run:
	./jsc
