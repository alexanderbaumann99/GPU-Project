/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code
the name of the author above.
***************************************************************/
#include "rng.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>

// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {
	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}
// Has to be defined in the compilation in order to get the correct value 
// of the macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

#define nt 15
#define nk 6

__constant__ float Tg[nt];
__constant__ float rg[nt];
__constant__ float Kg[nk];
__constant__ float Cg[16 * (nt - 1) * (nk - 1)];

float* Cgc, * Kgc, * Tgc, * rgc;

// Allocate parameters
void VarMalloc()
{
	Kgc = (float*)calloc(nk, sizeof(float));
	Tgc = (float*)calloc(nt, sizeof(float));
	rgc = (float*)calloc(nt, sizeof(float));
	Cgc = (float*)calloc(16 * (nk - 1) * (nt - 1), sizeof(float));
}

// Free parameters
void FreeVar()
{
	free(Cgc);
	free(Kgc);
	free(Tgc);
	free(rgc);
}

// Time parameters
void parameters()
{
	Kgc[0] = 20.f;
	Kgc[1] = 70.f;
	Kgc[2] = 120.f;
	Kgc[3] = 160.f;
	Kgc[4] = 200.f;
	Kgc[5] = 250.0f;

	float d, w, m, y;
	d = 1.0f / 360.0f;
	w = 7.0f * d;
	m = 30.0f * d;
	y = 12.0f * m;

	Tgc[0] = d;
	Tgc[1] = 2.f * d;
	Tgc[2] = w;
	Tgc[3] = 2.f * w;
	Tgc[4] = m;
	Tgc[5] = 2.f * m;
	Tgc[6] = 3.f * m;
	Tgc[7] = 6.f * m;
	Tgc[8] = y;
	Tgc[9] = y + 3.f * m;
	Tgc[10] = y + 6.f * m;
	Tgc[11] = 2.f * y;
	Tgc[12] = 2.f * y + 6.f * m;
	Tgc[13] = 3.f * y;
	Tgc[14] = 3.f * y + 6.f * m;

	rgc[0] = 0.05f;
	rgc[1] = 0.07f;
	rgc[2] = 0.08f;
	rgc[3] = 0.06f;
	rgc[4] = 0.07f;
	rgc[5] = 0.1f;
	rgc[6] = 0.11f;
	rgc[7] = 0.13f;
	rgc[8] = 0.12f;
	rgc[9] = 0.14f;
	rgc[10] = 0.145f;
	rgc[11] = 0.14f;
	rgc[12] = 0.135f;
	rgc[13] = 0.13f;
	rgc[14] = 0.f * y;

	int k;
	FILE* ParFp;
	string TmpString;
	//Spline Volatility parameters------------------------------
	// - Read values from input file on CPU
	TmpString = "Cg.txt";
	ParFp = fopen(TmpString.c_str(), "r");
	if (ParFp == NULL) {
		fprintf(stderr, "File '%s' unreachable!\n", TmpString.c_str());
		exit(EXIT_FAILURE);
	}
	// - Store values in input data tables on CPU
	for (k = 0; k < 1120; k++) {
		if (fscanf(ParFp, "%f", &Cgc[k]) <= 0) {
			fprintf(stderr, "Error while reading file '%s'!\n", TmpString.c_str());
			exit(EXIT_FAILURE);
		}
	}
	fclose(ParFp);


	cudaMemcpyToSymbol(Kg, Kgc, nk * sizeof(float));
	cudaMemcpyToSymbol(Tg, Tgc, nt * sizeof(float));
	cudaMemcpyToSymbol(rg, rgc, nt * sizeof(float));
	cudaMemcpyToSymbol(Cg, Cgc, 16 * (nt - 1) * (nk - 1) * sizeof(float));
}

// Time index  
__device__ int timeIdx(float t) {
	int i, I;
	for (i = 14; i >= 0; i--) {
		if (t < Tg[i]) {
			I = i;
		}
	}
	return I;
}

// Interest rate time integral
__device__ float rt_int(float t, float T, int i, int j)
{
	float res;
	int k;
	if (i == j) {
		res = (T - t) * rg[i];
	}
	else {
		res = (T - Tg[j - 1]) * rg[j] + (Tg[i] - t) * rg[i];
		for (k = i + 1; k < j; k++) {
			res += (Tg[k] - Tg[k - 1]) * rg[k];
		}
	}

	return res;
}

// Monomials till third degree
__device__ float mon(float x, int i) { 
	return 1.0f * (i == 0) + x * (i == 1) + x * x * (i == 2) + x * x * x * (i == 3); 
}

// Local volatility from bicubic interpolation of implied volatility
__device__ void vol_d(float x, float x0, float t, float* V, int q) {

	float u1 = 0.0f;
	float u2 = 0.0f;
	float d1, d2, d_1;
	float y = 0.0f;
	float y_1 = 0.0f, y_2 = 0.0f, y_22 = 0.0f;
	int k = 0;


	if (x >= Kg[5]) {
		k = 4;
		d2 = 1.0f / (Kg[k + 1] - Kg[k]);
		u2 = 1.0f;
	}
	else {
		if (x <= Kg[0]) {
			k = 0;
			d2 = 1.0f / (Kg[k + 1] - Kg[k]);
			u2 = 1.0f;
		}
		else {
			while (Kg[k + 1] < x) {
				k++;
			}
			d2 = 1.0f / (Kg[k + 1] - Kg[k]);
			u2 = (x - Kg[k]) / (Kg[k + 1] - Kg[k]);
		}
	}

	d1 = 1.0f / (Tg[q + 1] - Tg[q]);
	u1 = (t - Tg[q]) / (Tg[q + 1] - Tg[q]);

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			y += Cg[k * 14 * 16 + q * 16 + j + i * 4] * mon(u1, i) * mon(u2, j);
			y_1 += i * Cg[k * 14 * 16 + q * 16 + i * 4 + j] * mon(u1, i - 1) * mon(u2, j) * d1;
			y_2 += j * Cg[k * 14 * 16 + q * 16 + i * 4 + j] * mon(u1, i) * mon(u2, j - 1) * d2;
			y_22 += j * (j - 1) * Cg[k * 14 * 16 + q * 16 + i * 4 + j] * mon(u1, i) * mon(u2, j - 2) * d2 * d2;
		}
	}
	d_1 = (logf(x0 / x) + rt_int(0.0f, t, 0, q)) / (y * sqrtf(t)) + 0.5f * y * sqrtf(t);
	u1 = x * x * (y_22 - d_1 * sqrtf(t) * y_2 * y_2 + (1.0f / y) * ((1.0f / (x * sqrtf(t)))
		+ d_1 * y_2) * ((1.0f / (x * sqrtf(t))) + d_1 * y_2));
	u2 = 2.0f * y_1 + y / t + 2.0f * x * rg[q] * y_2;

	*V = sqrtf(fminf(fmaxf(u2 / u1, 0.0001f), 0.5f));
}


// Set the new RNG seed
__device__ void CMRG_set_d(int* a0, int* a1, int* a2, int* a3, int* a4,
	int* a5, int* CMRG_Out) {
	CMRG_Out[0] = *a0;
	CMRG_Out[1] = *a1;
	CMRG_Out[2] = *a2;
	CMRG_Out[3] = *a3;
	CMRG_Out[4] = *a4;
	CMRG_Out[5] = *a5;
}

// Get the RNG Seed
__device__ void CMRG_get_d(int* a0, int* a1, int* a2, int* a3, int* a4,
	int* a5, int* CMRG_In) {
	*a0 = CMRG_In[0];
	*a1 = CMRG_In[1];
	*a2 = CMRG_In[2];
	*a3 = CMRG_In[3];
	*a4 = CMRG_In[4];
	*a5 = CMRG_In[5];
}

// Generate uniformly distributed random variables
__device__ void CMRG_d(int* a0, int* a1, int* a2, int* a3, int* a4,
	int* a5, float* g0, float* g1, int nb) {

	const int m1 = 2147483647;// Requested for the simulation
	const int m2 = 2145483479;// Requested for the simulation
	int h, p12, p13, p21, p23, k, loc;// Requested local parameters

	for (k = 0; k < nb; k++) {
		// First Component 
		h = *a0 / q13;
		p13 = a13 * (h * q13 - *a0) - h * r13;
		h = *a1 / q12;
		p12 = a12 * (*a1 - h * q12) - h * r12;

		if (p13 < 0) {
			p13 = p13 + m1;
		}
		if (p12 < 0) {
			p12 = p12 + m1;
		}
		*a0 = *a1;
		*a1 = *a2;
		if ((p12 - p13) < 0) {
			*a2 = p12 - p13 + m1;
		}
		else {
			*a2 = p12 - p13;
		}

		// Second Component 
		h = *a3 / q23;
		p23 = a23 * (h * q23 - *a3) - h * r23;
		h = *a5 / q21;
		p21 = a21 * (*a5 - h * q21) - h * r21;

		if (p23 < 0) {
			p23 = p23 + m2;
		}
		if (p12 < 0) {
			p21 = p21 + m2;
		}
		*a3 = *a4;
		*a4 = *a5;
		if ((p21 - p23) < 0) {
			*a5 = p21 - p23 + m2;
		}
		else {
			*a5 = p21 - p23;
		}

		// Combines the two MRGs
		if (*a2 < *a5) {
			loc = *a2 - *a5 + m1;
		}
		else { loc = *a2 - *a5; }

		if (k) {
			if (loc == 0) {
				*g1 = Invmp * m1;
			}
			else { *g1 = Invmp * loc; }
		}
		else {
			*g1 = 0.0f;
			if (loc == 0) {
				*g0 = Invmp * m1;
			}
			else { *g0 = Invmp * loc; }
		}
	}
}


// Genrates Gaussian distribution from a uniform one (Box-Muller)
__device__ void BoxMuller_d(float* g0, float* g1) {

	float loc;
	if (*g1 < 1.45e-6f) {
		loc = sqrtf(-2.0f * logf(0.00001f)) * cosf(*g0 * 2.0f * MoPI);
	}
	else {
		if (*g1 > 0.99999f) {
			loc = 0.0f;
		}
		else { loc = sqrtf(-2.0f * logf(*g1)) * cosf(*g0 * 2.0f * MoPI); }
	}
	*g0 = loc;
}


// Euler for local volatility
__device__ void Euler_d(float* S2, float S1, float r0,
	float sigma, float dt, float e) {

	*S2 = S1 * (1.0f + r0 * dt * dt + sigma * dt * e);
}

// Monte Carlo routine
__global__ void MC_k(int P1, int P2, float x_0, float dt,
	float B, float K, int L, int M,
	int Ntraj, TabSeedCMRG_t* pt_cmrg,
	float* time, float* price, float* i_t, float* sum, float* sum2) {

	// Define global index in x-coordinate
	int gb_index_x = threadIdx.x + blockIdx.x * blockDim.x;
	// Define global index in y-coordinate
	int gb_index_y = threadIdx.y + blockIdx.y* blockDim.y;
	int a0, a1, a2, a3, a4, a5, k, i, q, P;
	float g0, g1, Sk, Skp1, t, v;

	extern __shared__ float H[];
	 
	
	Sk = x_0;
	P = 0;

	CMRG_get_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][gb_index_x][gb_index_y]);

	// First Loop over the T_i -> predetermined schedule
	for (k = 0; k < M; k++) {
		if (gb_index_y == 0) {
			//Going to next time point with threadIdx.y==0
			for (i = 1; i <= L; i++) {
				t = dt * dt * (i + L * k);
				q = timeIdx(t);
				// Local Volatility
				vol_d(Sk, x_0, t, &v, q);
				// Get uniformly RN
				CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &g0, &g1, 2);
				// Get Gaussian
				BoxMuller_d(&g0, &g1);
				// Update price into Skpl
				Euler_d(&Skp1, Sk, rg[q], v, dt, g0);
				Sk = Skp1;
			}
			//Reached new time point
			// Update I
			P += (Sk < B);
			time[gb_index_x + k * blockDim.x * gridDim.x] = k;
			price[gb_index_x + k * blockDim.x * gridDim.x] = Sk;
			i_t[gb_index_x + k * blockDim.x * gridDim.x] = P;
		}
		__syncthreads();
		//From here, do the complete inner trajectories with threadIdx.y
		Sk = price[gb_index_x + k * blockDim.x * gridDim.x];
		P = i_t[gb_index_x + k * blockDim.x * gridDim.x];
		for (int j = k+1; j < M; j++) {
			for (i = 1; i <= L; i++) {
				t = dt * dt * (i + L * j); //Changed k to j
				q = timeIdx(t);
				// Local Volatility
				vol_d(Sk, x_0, t, &v, q);
				// Get uniformly RN
				CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &g0, &g1, 2);
				// Get Gaussian
				BoxMuller_d(&g0, &g1);
				// Update price into Skpl
				Euler_d(&Skp1, Sk, rg[q], v, dt, g0);
				Sk = Skp1;
			}
			// Update I
			P += (Sk < B);
		}
		// Changed discount factor
		H[threadIdx.y] = expf(-rt_int(dt * dt *L * k, t, 0, q)) * fmaxf(0.0f, Sk - K) * ((P <= P2) && (P >= P1)) / Ntraj;
		// Changed index to .y
		H[threadIdx.y + blockDim.y] = Ntraj * H[threadIdx.y] * H[threadIdx.y];
		__syncthreads();

		i = blockDim.y / 2;
		while (i != 0) {
			if (threadIdx.y < i) {
				H[threadIdx.y] += H[threadIdx.y + i];
				H[threadIdx.y + blockDim.y] += H[threadIdx.y + blockDim.y + i];
			}
			__syncthreads();
			i /= 2;
		}

		if (threadIdx.y == 0) {
			atomicAdd(&sum[gb_index_x + k * blockDim.x * gridDim.x], H[0]);
			atomicAdd(&sum2[gb_index_x + k * blockDim.x * gridDim.x], H[blockDim.y]);
		}


	}


	/*// Reduction phase
	// Calculate fair price at t=0 for this trajectory
	H[threadIdx.x] = expf(-rt_int(0.0f, t, 0, q)) * fmaxf(0.0f, Sk - K) * ((P <= P2) && (P >= P1)) / Ntraj;
	// ???
	H[threadIdx.x + blockDim.x] = Ntraj * H[threadIdx.x] * H[threadIdx.x];
	__syncthreads();

	i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i) {
			H[threadIdx.x] += H[threadIdx.x + i];
			H[threadIdx.x + blockDim.x] += H[threadIdx.x + blockDim.x + i];
		}
		__syncthreads();
		i /= 2;
	}

	if (threadIdx.x == 0) {
		atomicAdd(sum, H[0]);
		atomicAdd(sum2, H[blockDim.x]);
	}

	//CMRG_set_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][gb_index_x]);*/
}


int main()
{

	float T = 1.0f;
	float K = 100.0f;
	float x_0 = 100.0f;
	float B = 120.0f;
	// Number of the T_i's
	int M = 100;
	int P1 = 10;
	int P2 = 49;
	// Number of time steps for the price process
	int Nt = 200;
	float dt = sqrtf(T / Nt);
	int leng = Nt / M;
	float Tim;							// GPU timer instructions
	cudaEvent_t start, stop;			// GPU timer instructions
	int Ntraj = 512 * 128;
	float* time;
	float* price;
	float* i_t;
	float* sum;
	float* sum2;
	float* time_c = (float*)malloc(sizeof(float) * Ntraj * M);
	float* price_c = (float*)malloc(sizeof(float) * Ntraj * M);
	float* i_t_c = (float*)malloc(sizeof(float) * Ntraj * M);
	float* sum_c = (float*)malloc(sizeof(float) * Ntraj * M);
	float* sum2_c = (float*)malloc(sizeof(float) * Ntraj * M);

	cudaMalloc(&time, sizeof(float) * Ntraj * M);
	cudaMalloc(&price, sizeof(float) * Ntraj * M);
	cudaMalloc(&i_t, sizeof(float) * Ntraj * M);
	cudaMalloc(&sum, sizeof(float) * Ntraj * M);
	cudaMalloc(&sum2, sizeof(float) * Ntraj * M);

	cudaMemset(sum, 0.0f, sizeof(float) * Ntraj * M);
	cudaMemset(sum2, 0.0f, sizeof(float) * Ntraj * M);
	printf("Check1");
	cudaMemcpy(time_c, time, sizeof(float) * Ntraj * M, cudaMemcpyDeviceToHost);
	printf("Check4");
	
	VarMalloc();
	PostInitDataCMRG();

	parameters();

	cudaEventCreate(&start);			// GPU timer instructions
	cudaEventCreate(&stop);				// GPU timer instructions
	cudaEventRecord(start, 0);			// GPU timer instructions

	printf("Check2");

	dim3 num_threads(128, 128);
	dim3 num_blocks(512, 128);
	// Modify NTPB to 2 dimensions
	MC_k <<< num_blocks, num_threads, 2 * 128 * sizeof(float) >> > (P1, P2, x_0, dt, B, K,
		leng, M, Ntraj, CMRG,time,price,i_t,sum,sum2);
	printf("Check3");
	cudaEventRecord(stop, 0);					// GPU timer instructions
	cudaEventSynchronize(stop);					// GPU timer instructions
	cudaEventElapsedTime(&Tim, start, stop);	// GPU timer instructions
	cudaEventDestroy(start);					// GPU timer instructions
	cudaEventDestroy(stop);						// GPU timer instructions
	printf("Check3.5");
	cudaMemcpy(sum_c, sum, sizeof(float) * Ntraj * M, cudaMemcpyDeviceToHost);
	printf("Check4");
	cudaFree(sum);
	printf("Check5\n");
	printf("The price is equal to %f",sum_c[0]);

	printf("Execution time %f ms\n", Tim);


	FreeCMRG();
	FreeVar();

	return 0;
}


