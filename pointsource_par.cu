#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>


#define THREADS_PER_BLOCK 32
#define TIME 3600000


__global__ void compute(float *a_d, float *b_d, float *c_d, float arraySize)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;	

	if( ix > 0 && ix < arraySize-1){
	   b_d[ix] = (a_d[ix+1]+a_d[ix-1])/2.0;
	}
	__syncthreads();

	
		 	
}		

__global__ void initialize(float *a_d, float *b_d, float *c_d, float arraySize)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;	
	if(ix==0)
	{
		a_d[ix]=200.0;
		b_d[ix]=200.0;

	}

	else if (ix<arraySize)
	{
		a_d[ix]=0.0;
		b_d[ix]=0.0;
	}
	

}
 


extern "C" void pointsource_pollution (float *a, float *b, int *c, int arraySize)
{

	float *a_d, *b_d, *c_d;

	cudaMalloc ((void**) &a_d, sizeof(float) * arraySize);
	cudaMalloc ((void**) &b_d, sizeof(float) * arraySize);
	cudaMalloc ((void**) &c_d, sizeof(float) * arraySize);
	

	initialize<<< ceil((float) arraySize/THREADS_PER_BLOCK), THREADS_PER_BLOCK >>> (a_d, b_d, c_d, arraySize);

	for(int i=0;i<TIME;i++){

	compute <<< ceil((float) arraySize/THREADS_PER_BLOCK), THREADS_PER_BLOCK >>> (a_d, b_d, c_d, arraySize);
	a_d=b_d;
	
	}

	cudaMemcpy (a, a_d, sizeof(float) * arraySize, cudaMemcpyDeviceToHost);
	
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf ("CUDA error: %s\n", cudaGetErrorString(err));
		
	
	cudaFree (a_d);
	cudaFree (b_d);
	cudaFree (c_d);
		
	
}