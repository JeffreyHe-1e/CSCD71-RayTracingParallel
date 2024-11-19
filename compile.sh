#!/bin/sh
# single-threaded version
g++ -O3 -g svdDynamic.c RayTracer.c utils.c -lm -o RayTracer

# multi-threaded version
# g++ -O3 -g -fopenmp svdDynamic.c RayTracer.c utils.c -lm -o RayTracer
