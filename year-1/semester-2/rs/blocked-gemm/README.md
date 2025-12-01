# rs-blocked-gemm

A repository for the first assignment for the Computer Systems course, implementing the Blocked GEMM algorithm. The purpose of this assignment was to study the effects of caching and blocking the GEMM algorithm. It is optimized to work on the Slurm supercomputer.

# Project structure

- The `src` directory contains the actual implementation of the naive and blocked GEMM algorithm in C.
- The `analysis` directory contains the bandwidth data measured on the supercomputer, along with a python script for the analysis of the data. 