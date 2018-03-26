/*
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc -arch=sm_11 bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 * License: BSD 3
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/* Every thread gets exactly one value in the unsorted array. */
#define THREADS 512 // 2^9
#define NIM1 13515141
#define NIM2 13515147
#define SWAP(x,y) t = x; x = y; y = t;

char* input_path = "data/input";
char* output_path = "data/output";
FILE* input_file;
FILE* output_file;
const int up = 1;
const int down = 0;
int * array;
int array_size;
int NUM_VALS;
int BLOCKS;

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

int random_float()
{
  return (int)rand();
}

void array_print(int *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%d ",  arr[i]);
  }
  printf("\n");
}

void array_fill(int *arr, int length)
{
  srand(13515147);
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}

__global__ void bitonic_sort_step(int *dev_values, int j, int k)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[ixj]) {
        /* exchange(i,ixj); */
        int temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
        /* exchange(i,ixj); */
        int temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(int *values)
{
  int *dev_values;
  size_t size = NUM_VALS * sizeof(int);

  cudaMalloc((void**) &dev_values, size);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */

  int j, k;
  /* Major step */
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
    }
  }
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaFree(dev_values);
}

void compare(int i, int j, int dir){
  int t;
  if (dir == (array[i] > array[j])){
    SWAP(array[i], array[j]);
  }
}

/*
 * Returns the greatest power of two number that is less than n
 */
 int greatestPowerOfTwoLessThan(int n){
  int k=1;
  while(k>0 && k<n)
    k=k<<1;
  return k>>1;
}
/*
 * Sorts a bitonic sequence in ascending order if dir=1
 * otherwise in descending order
 */
void bitonicMerge(int low, int c, int dir){
  int k, i;
  if (c > 1){
    k = greatestPowerOfTwoLessThan(c);
    for (i = low;i < low+c-k ;i++)
      compare(i, i+k, dir);
    bitonicMerge(low, k, dir);
    bitonicMerge(low+k, c-k, dir);
  }
}

/*
 * Generates bitonic sequence by sorting recursively
 * two halves of the array in opposite sorting orders
 * bitonicMerge will merge the resultant array
 */
void recursiveBitonic(int low, int c, int dir){
  int k;
  if (c > 1) {
    k = c / 2;
    recursiveBitonic(low, k, !dir);
    recursiveBitonic(low + k, c-k, dir);
    bitonicMerge(low, c, dir);
  }
}



/*
 * Sort array with serial bitonic sorting
 */
void sort_serial(){
  recursiveBitonic(0, array_size, up);
}

int is_sorted() {
  int i;
  for (i=0; i<array_size-1; i++) {
    if (array[i] > array[i+1]) return 0;
  }
  return 1;
}

int main(int argc, char * argv[])
{
  clock_t start, stop;


  array_size = atoi(argv[1]);
  NUM_VALS=array_size;
  BLOCKS=NUM_VALS/512;
  array = (int*) malloc( NUM_VALS * sizeof(int));
  array_fill(array, NUM_VALS);
  start = clock();
  sort_serial();
  stop = clock();
  printf("[SERIAL]\n");
        if (is_sorted()) {
          printf("Sorting successful\n");
        } else {
          printf("Sorting failed\n");
        }
  print_elapsed(start, stop);
  free(array);
  array = (int*) malloc( NUM_VALS * sizeof(int));
  array_fill(array, NUM_VALS);
  start = clock();
  bitonic_sort(array); /* Inplace */
  printf("[PARALEL]\n");
        if (is_sorted()) {
          printf("Sorting successful\n");
        } else {
          printf("Sorting failed\n");
        }
  stop = clock();

  print_elapsed(start, stop);
}

