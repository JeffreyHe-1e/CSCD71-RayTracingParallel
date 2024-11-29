#!/bin/sh
# single-threaded version
#g++ -O3 -g svdDynamic.c RayTracer.c utils.c -lm -o RayTracer

# multi-threaded version
# g++ -O3 -g -fopenmp svdDynamic.c RayTracer.c utils.c -lm -o RayTracer

# compile multi-threaded (on Mac with homebrew libomp)
g++ -g -O3 -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp svdDynamic.c RayTracer.c utils.c -o RayTracer