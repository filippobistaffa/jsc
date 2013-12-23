.PHONY:

all:
	gcc -Wall -Ofast -march=native -funroll-loops *.c -o jsc

cuda:
	nvcc --compiler-options -Wall,-Ofast,-march=native,-funroll-loops --ptxas-options=-v -arch=sm_30 marsenne.c *.cu -o jsc

run:
	./jsc
