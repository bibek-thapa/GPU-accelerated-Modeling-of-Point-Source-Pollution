#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>


#define THREADS_PER_BLOCK 32
#define TIME 3600000


__global__ void compute(float *a_d, float *b_d, float *c_d, float arraySize)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int t = threadIdx.x;
	int blockdim=blockDim.x;
		 
	if(ix<arraySize){
	if(ix==0){	
		b_d[ix]=200.0;
		}
		else{
		b_d[ix]=0.0;
		}
	}
	
	
	for(int k=0;k<TIME;k++) // time-loop
    {
	if( ix > 0 && ix < arraySize-1){
	   b_d[ix] = (b_d[ix+1]+b_d[ix-1])/2.0;
	}
	a_d[ix]=b_d[ix];
    	

}		
} 


extern "C" void pointsource_pollution (float *a, float *b, int *c, int arraySize)
{
	int numDevices = 0;    
	cudaGetDeviceCount(&numDevices); 
	   if (numDevices > 1)
	    {       int maxMultiprocessors = 0, maxDevice = 0; 
	          for (int device=0; device<numDevices; device++) {          cudaDeviceProp props;          cudaGetDeviceProperties(&props, device); 
	                   if (maxMultiprocessors < props.multiProcessorCount) {           
	                     maxMultiprocessors = props.multiProcessorCount;  
	                                maxDevice = device;          }       }    
	       cudaSetDevice(maxDevice);   } 

	float *a_d, *b_d, *c_d;

	cudaMalloc ((void**) &a_d, sizeof(float) * arraySize);
	cudaMalloc ((void**) &b_d, sizeof(float) * arraySize);
	cudaMalloc ((void**) &c_d, sizeof(float) * arraySize);
	

	compute <<< ceil((float) arraySize/THREADS_PER_BLOCK), THREADS_PER_BLOCK >>> (a_d, b_d, c_d, arraySize);
	cudaMemcpy (a, a_d, sizeof(float) * arraySize, cudaMemcpyDeviceToHost);
	
	
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf ("CUDA error: %s\n", cudaGetErrorString(err));
		
	
	cudaFree (a_d);
	cudaFree (b_d);
	cudaFree (c_d);
		
	
}