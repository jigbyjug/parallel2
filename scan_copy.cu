#include <stdio.h>
#include <cstdlib.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

using namespace std;

__global__ void vectorAdd(float *g_odata, float *g_idata, int n)
{
	extern __shared__ float temp[]; //allocated on invocation
	int thid = threadIdx.x;
	int offset = 1;

	temp[2*thid] = g_idata[2*thid];		// load input into shared memory
	temp[2*thid+1] = g_idata[2*thid+1];
		
	for (int d = n>>1; d > 0; d >>= 1) // build sum in place up the tree 
	{
		__syncthreads();
		
		if (thid < d)
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;

			temp[bi] += temp[1i];
		}
		offset *= 2;
	}
		

	if (thid==0)  { temp[n-1] = 0;} // clear the last element

	for (int d = 1; d < n; d*= 2)   // traverse down tree & build scan
	{
		offset >>=1;
		__syncthreads();

		if (thid < d)
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;

			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads(); 

	g_odata[2*thid] = temp[2*thid]; //write results to device memory
	g_odata[2*thid+1] = temp[2*thid+1];
}


int main(void) 
{
	int numElements = 15; 
	size_t size = numElements * sizeof(float);
	
	//Allocate the host input vector A
	float *h_A = (float *)malloc(size);
	
	//Verify that allocations succeeded
	if (h_A == Null)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}
	
	//Initialize the host input vectors
	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = rand() % 101;
		cout<<h_A[i]<<'\n';
	}	

	// Allocate the deivce input vector A
	float *d_A = NULL;
	err = cudaMalloc((void **)&d_A, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A!\n");
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
    	float *d_B = NULL;
    	err = cudaMalloc((void **)&d_B, size);
	
	if (err != cudaSuccess)
        {
                fprintf(stderr, "Failed to allocate device vector B!\n");
                exit(EXIT_FAILURE);
        }


	//Copy the host input vectors A in host memory to the device input vectors
	// in device memory

	 printf("Copy input data from the host memory to the CUDA device\n");
	 err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

   	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to copy vector A from host to device!\n");
        	exit(EXIT_FAILURE);
    	}


	printf("Copy input data from the host memory to the CUDA device\n");
        err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        if (err != cudaSuccess)
        {
                fprintf(stderr, "Failed to copy vector A from host to device!\n");
                exit(EXIT_FAILURE);
        }





	// Launch the vector add cuda kernel
	int threadsPerBlock = 256;	
	int blocksPerGrid = (numElements + threadsPerBlock - 1)/threadsPerBlock;
	
	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, numElements);


	//Copy the device result vector in device memory to the host result vector
	// in host memory

	err = cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to copy vector B from device to host!\n");
        	exit(EXIT_FAILURE);
    	}

	for (int i = 0; i < numElements; i++)
	{
		cout<<h_B[i]<<'\n';
	}


	// Free device global memory
	err = cudaFree(d_A);

	if (err != cudaSuccess)
        {
                fprintf(stderr, "Failed to free device vector A!\n");
                exit(EXIT_FAILURE);
        }

	err = cudaFree(d_B);

	if (err != cudaSuccess)
        {
                fprintf(stderr, "Failed to free device vector B!\n");
                exit(EXIT_FAILURE);
        }

	//free host memory
	free(h_A);
	free(h_B);

	// reset device and exit
	err = cudaDeviceReset();
	
	if (err != cudaSuccess)
        {
                fprintf(stderr, "Failed to deinitialize the device!\n");
                exit(EXIT_FAILURE);
        }

	return 0;
}



