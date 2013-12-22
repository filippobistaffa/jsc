.PHONY:

all:
	gcc -Wall -Ofast -march=native -funroll-loops *.c -o jsc

cuda:
	nvcc --compiler-options -Wall,-Ofast,-march=native,-funroll-loops *.c *.cu -o jsc

run:
	./jsc
