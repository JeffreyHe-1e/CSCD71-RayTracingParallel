#!/bin/sh

# compile multi-threaded with OpenMP
g++ -O3 -g -fopenmp svdDynamic.c RayTracer.c utils.c -lm -o RayTracer