target:
	nvcc src/bitonic_sort.cu -o bitonic_sort

run:
	./bitonic_sort 1048576
