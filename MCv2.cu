/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code 
the name of the author above.
***************************************************************/
#include "rng.h"

#define nt 15
#define nk 6

__constant__ float Tg[nt];
__constant__ float rg[nt];
__constant__ float Kg[nk];
__constant__ float Cg[16*(nt-1)*(nk-1)];

float *Cgc, *Kgc, *Tgc, *rgc;

// Allocate parameters
void VarMalloc()
{
	Kgc = (float *)calloc(nk, sizeof(float));
	Tgc = (float *)calloc(nt, sizeof(float));
	rgc = (float *)calloc(nt, sizeof(float));
	Cgc = (float *)calloc(16*(nk-1)*(nt-1), sizeof(float));
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
 	Tgc[1] = 2.f*d;
 	Tgc[2] = w;
 	Tgc[3] = 2.f*w;
	Tgc[4] = m;
 	Tgc[5] = 2.f*m;
 	Tgc[6] = 3.f*m;
 	Tgc[7] = 6.f*m;
 	Tgc[8] = y;
 	Tgc[9] = y + 3.f*m;
 	Tgc[10] =y + 6.f*m;
 	Tgc[11] = 2.f*y;
 	Tgc[12] = 2.f*y + 6.f*m;
 	Tgc[13] = 3.f*y;
 	Tgc[14] = 3.f*y + 6.f*m;

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
	rgc[14] = 0.f*y;

	int k;
	FILE *ParFp;
	string TmpString;
	//Spline Volatility parameters------------------------------
	// - Read values from input file on CPU
	TmpString = "Cg.txt";
	ParFp = fopen(TmpString.c_str(),"r");
	if (ParFp == NULL) {
	  fprintf(stderr,"File '%s' unreachable!\n",TmpString.c_str());
	  exit(EXIT_FAILURE);   
	}
	// - Store values in input data tables on CPU
	for (k = 0; k < 1120; k++) {
		if (fscanf(ParFp,"%f",&Cgc[k]) <= 0) {
		  fprintf(stderr,"Error while reading file '%s'!\n",TmpString.c_str());
		  exit(EXIT_FAILURE);          
		}
	}
	fclose(ParFp);

	
	cudaMemcpyToSymbol(Kg, Kgc, nk*sizeof(float));
	cudaMemcpyToSymbol(Tg, Tgc, nt*sizeof(float));
	cudaMemcpyToSymbol(rg, rgc, nt*sizeof(float));
	cudaMemcpyToSymbol(Cg, Cgc, 16*(nt-1)*(nk-1)*sizeof(float));
}

// Time index  
__device__ int timeIdx(float t) {
	int i, I;
	for (i=14; i>=0; i--) {
		if(t<Tg[i]){
			I = i;
		}
	}
	return I;
}

// Interest rate time integral
__device__ float rt_int(float t,  float T, int i, int j)
{
	float res;
	int k;
	if(i==j){
		res = (T-t)*rg[i];
	}else{
		res = (T-Tg[j-1])*rg[j] + (Tg[i]-t)*rg[i];
		for(k=i+1; k<j; k++){
			res += (Tg[k]-Tg[k-1])*rg[k];
		}
	}

	return res;
}

// Monomials till third degree
__device__ float mon(float x, int i){return 1.0f*(i==0) + x*(i==1) + x*x*(i==2) + x*x*x*(i==3);}

// Local volatility from bicubic interpolation of implied volatility
__device__ void vol_d(float x, float x0, float t, float *V, int q){

	float u1 = 0.0f;
	float u2 = 0.0f;
	float d1, d2, d_1;
	float y = 0.0f;
	float y_1 = 0.0f, y_2 = 0.0f, y_22 = 0.0f;
	int k = 0;
	
	
	if (x >= Kg[5]){
		k = 4;
		d2 = 1.0f /(Kg[k + 1] - Kg[k]);
		u2 = 1.0f;
	}else{
		if (x <= Kg[0]){
			k = 0;
			d2 = 1.0f/(Kg[k + 1] - Kg[k]);
			u2 = 1.0f;
		}else{
			while (Kg[k+1] < x){
				k++;
			}
			d2 = 1.0f/(Kg[k+1] - Kg[k]);
			u2 = (x - Kg[k])/(Kg[k+1] - Kg[k]);
		}
	}

	d1 = 1.0f/(Tg[q + 1] - Tg[q]);
	u1 = (t - Tg[q])/(Tg[q + 1] - Tg[q]);

	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			y += Cg[k * 14 * 16 + q * 16 + j + i * 4] * mon(u1, i)*mon(u2, j);
			y_1 += i *Cg[k * 14 * 16 + q * 16 + i * 4 + j] * mon(u1, i-1)*mon(u2, j)*d1;
			y_2 += j*Cg[k * 14 * 16 + q * 16 + i * 4 + j] * mon(u1, i)*mon(u2, j-1)*d2;
			y_22 += j *(j - 1)*Cg[k * 14 * 16 + q * 16 + i * 4 + j] * mon(u1, i)*mon(u2, j-2)*d2*d2;
		}
	}
	d_1 = (logf(x0/x) + rt_int(0.0f, t, 0, q))/(y*sqrtf(t)) + 0.5f*y*sqrtf(t);
	u1 = x*x*(y_22 - d_1*sqrtf(t)*y_2*y_2 + (1.0f/y)*((1.0f/(x*sqrtf(t))) 
		+ d_1*y_2)*((1.0f /(x*sqrtf(t))) + d_1*y_2));
	u2 = 2.0f*y_1 + y /t + 2.0f*x*rg[q]*y_2;
	
	*V = sqrtf(fminf(fmaxf(u2/u1,0.0001f),0.5f));
}


// Set the new RNG seed
__device__ void CMRG_set_d(int *a0, int *a1, int *a2, int *a3, int *a4, 
			         int *a5, int *CMRG_Out){
	CMRG_Out[0] = *a0;
	CMRG_Out[1] = *a1;
	CMRG_Out[2] = *a2;
	CMRG_Out[3] = *a3;
	CMRG_Out[4] = *a4;
	CMRG_Out[5] = *a5;
}

// Get the RNG Seed
__device__ void CMRG_get_d(int *a0, int *a1, int *a2, int *a3, int *a4, 
			         int *a5, int *CMRG_In){
	*a0 = CMRG_In[0];
	*a1 = CMRG_In[1];
	*a2 = CMRG_In[2];
	*a3 = CMRG_In[3];
	*a4 = CMRG_In[4];
	*a5 = CMRG_In[5];
}

// Generate uniformly distributed random variables
__device__ void CMRG_d(int *a0, int *a1, int *a2, int *a3, int *a4, 
			     int *a5, float *g0, float *g1, int nb){

 const int m1 = 2147483647;// Requested for the simulation
 const int m2 = 2145483479;// Requested for the simulation
 int h, p12, p13, p21, p23, k, loc;// Requested local parameters

 for(k=0; k<nb; k++){
	 // First Component 
	 h = *a0/q13; 
	 p13 = a13*(h*q13-*a0)-h*r13;
	 h = *a1/q12; 
	 p12 = a12*(*a1-h*q12)-h*r12;

	 if (p13 < 0) {
	   p13 = p13 + m1;
	 }
	 if (p12 < 0) {
	   p12 = p12 + m1;
	 }
	 *a0 = *a1;
	 *a1 = *a2;
	 if( (p12 - p13) < 0){
	   *a2 = p12 - p13 + m1;  
	 } else {
	   *a2 = p12 - p13;
	 }
  
	 // Second Component 
	 h = *a3/q23; 
	 p23 = a23*(h*q23-*a3)-h*r23;
	 h = *a5/q21; 
	 p21 = a21*(*a5-h*q21)-h*r21;

	 if (p23 < 0){
	   p23 = p23 + m2;
	 }
	 if (p12 < 0){
	   p21 = p21 + m2;
	 }
	 *a3 = *a4;
	 *a4 = *a5;
	 if ( (p21 - p23) < 0) {
	   *a5 = p21 - p23 + m2;  
	 } else {
	   *a5 = p21 - p23;
	 }

	 // Combines the two MRGs
	 if(*a2 < *a5){
		loc = *a2 - *a5 + m1;
	 }else{loc = *a2 - *a5;} 

	 if(k){
		if(loc == 0){
			*g1 = Invmp*m1;
		}else{*g1 = Invmp*loc;}
	 }else{
		*g1 = 0.0f; 
		if(loc == 0){
			*g0 = Invmp*m1;
		}else{*g0 = Invmp*loc;}
	 }
  }
}


// Genrates Gaussian distribution from a uniform one (Box-Muller)
__device__ void BoxMuller_d(float *g0, float *g1){

  float loc;
  if (*g1 < 1.45e-6f){
    loc = sqrtf(-2.0f*logf(0.00001f))*cosf(*g0*2.0f*MoPI);
  } else {
    if (*g1 > 0.99999f){
      loc = 0.0f;
    } else {loc = sqrtf(-2.0f*logf(*g1))*cosf(*g0*2.0f*MoPI);}
  }
  *g0 = loc;
}


// Euler for local volatility
__device__ void Euler_d(float *S2, float S1, float r0,
						float sigma, float dt, float e){

  *S2 = S1*(1.0f + r0*dt*dt + sigma*dt*e);
}

// Monte Carlo routines
__global__ void MCouter_k(int P1, int P2, float x_0, float dt, 
					 float B, float K, int L, int M,
					 int Nouter, TabSeedCMRG_t *pt_cmrg,
					 float* time, float* price, int* i_t){

  // threadIdx.x and blockIdx.x -> index outer trajectory
  int idx_outer = threadIdx.x + blockDim.x * blockIdx.x;

  int a0, a1, a2, a3, a4, a5, k, i, q, P;
  float g0, g1, Sk, Skp1, t, v;

  //extern __shared__ float H[];


  if(idx_outer < Nouter){

    Sk = x_0;
    P = 0;

    // can we make this access to pt_cmrg contiguous?
    CMRG_get_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][idx_outer][0]);

    for (k=0; k<M-1; k++){
      for (i=1; i<=L; i++){
        t = dt*dt*(i+L*k);
        q = timeIdx(t);
        vol_d(Sk, x_0, t, &v, q);
        CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &g0, &g1, 2);
        BoxMuller_d(&g0, &g1);
        Euler_d(&Skp1, Sk, rg[q], v, dt, g0);
        Sk = Skp1;  
      }
      P += (Sk<B);
      
      // save results
      // maybe its faster to put values in shared memory first and to have one
      // thread per block copy the results for the whole block over to the global
      // memory? does this even matter in outer MC?
      time[idx_outer+k*Nouter] = t;
      price[idx_outer+k*Nouter] = Sk;
      i_t[idx_outer+k*Nouter] = P;
    }

    CMRG_set_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][idx_outer][0]);
  }
  
}

__global__ void MCinner_k(int P1, int P2, float dt, 
					 float B, float K, int L, int M,
					 int Ninner, TabSeedCMRG_t *pt_cmrg, int k_start,
					 float* time, float* price, int* i_t, float* sum, float* sum2){

  // blockIdx.x -> index outer trajectory
  int idx_outer = blockIdx.x;
  // threadIdx.x and blockIdx.y -> index inner trajectory
  int idx_inner = threadIdx.x + blockDim.x * blockIdx.y;

  int a0, a1, a2, a3, a4, a5, k, i, q, P;
  float g0, g1, Sk, Skp1, t, v;

  extern __shared__ float H[];

  if(idx_inner < Ninner){

    // is it quicker to make this access to global memory in a batch once per block?
    Sk = price[idx_outer];
    P = i_t[idx_outer];

    CMRG_get_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][idx_outer][idx_inner]);

    for (k=k_start; k<M; k++){
      for (i=1; i<=L; i++){
        t = dt*dt*(i+L*k);
        q = timeIdx(t);
        vol_d(Sk, price[idx_outer], t, &v, q);
        CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &g0, &g1, 2);
        BoxMuller_d(&g0, &g1);
        Euler_d(&Skp1, Sk, rg[q], v, dt, g0);
        Sk = Skp1;  
      }
      P += (Sk<B);
    }

    CMRG_set_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][idx_outer][idx_inner]);

    // Reduction phase
    H[threadIdx.x] = expf(-rt_int(0.0f, t, 0, q))*fmaxf(0.0f, Sk-K)*((P<=P2)&&(P>=P1))/Ninner;
    H[threadIdx.x + blockDim.x] = Ninner*H[threadIdx.x]*H[threadIdx.x];
    __syncthreads();

    i = blockDim.x/2;
    while (i != 0) {
      if (threadIdx.x < i){
        H[threadIdx.x] += H[threadIdx.x + i];
        H[threadIdx.x + blockDim.x] += H[threadIdx.x + blockDim.x + i];
      }
      __syncthreads();
      i /= 2;
    }

    if (threadIdx.x == 0){
      atomicAdd(sum + idx_outer, H[0]);
      atomicAdd(sum2 + idx_outer, H[blockDim.x]);
    }
  }
  
}

__global__ void MCreg_k(int P1, int P2, float dt, 
					 float B, float K, int L, int M,
					 int Nouter, TabSeedCMRG_t *pt_cmrg, int k_start,
					 float* time, float* price, int* i_t, float* x1, float* x2){

  // blockIdx.x -> index outer trajectory
  int idx_outer = threadIdx.x + blockDim.x * blockIdx.x;
  // blockIdx.y -> index inner trajectory
  int idx_inner = blockIdx.y;

  int a0, a1, a2, a3, a4, a5, k, i, q, P;
  float g0, g1, Sk, Skp1, t, v;

  if(idx_outer < Nouter){

    // is it quicker to make this access to global memory in a batch once per block?
    Sk = price[idx_outer];
    P = i_t[idx_outer];

    CMRG_get_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][idx_outer][idx_inner]);

    for (k=k_start; k<M; k++){
      for (i=1; i<=L; i++){
        t = dt*dt*(i+L*k);
        q = timeIdx(t);
        vol_d(Sk, price[idx_outer], t, &v, q);
        CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &g0, &g1, 2);
        BoxMuller_d(&g0, &g1);
        Euler_d(&Skp1, Sk, rg[q], v, dt, g0);
        Sk = Skp1;  
      }
      P += (Sk<B);
    }

    CMRG_set_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][idx_outer][idx_inner]);

    if(blockIdx.y == 0){
      x1[idx_outer] = expf(-rt_int(0.0f, t, 0, q))*fmaxf(0.0f, Sk-K)*((P<=P2)&&(P>=P1));
    }
    else{
      x2[idx_outer] = expf(-rt_int(0.0f, t, 0, q))*fmaxf(0.0f, Sk-K)*((P<=P2)&&(P>=P1));
    }
  }
  
}


int main()
{

	float T = 1.0f;
	float K = 100.0f;
	float x_0 = 100.0f;
	float B = 120.0f;
	int M = 100;
	int P1 = 10;
	int P2 = 49;
	int Nt = 200;
	float dt = sqrtf(T/Nt);
	int leng = Nt/M;
	float Tim;							// GPU timer instructions
	cudaEvent_t start, stop;			// GPU timer instructions
  int Nouter = 8192; //2^15
  int Ninner = 4096; // 2^9
  int Ndiscret = Nouter * (M-1); // -1 since the last point of an outer trajectory is uninteresting
  int threads_per_block = 1024;

  printf("Simulating nested Monte Carlo with\n \tnumber of outer trajectories: %d\n\tnumber of inner trajectories: %d\n", 
    Nouter, Ninner);

	float* time;
	float* price;
	int* i_t;
	float* sum;
	float* sum2;
  float* x1;
  float* x2;
	float* time_c = (float*)malloc(sizeof(float) * Ndiscret);
	float* price_c = (float*)malloc(sizeof(float) * Ndiscret);
	int* i_t_c = (int*)malloc(sizeof(int) * Ndiscret);
	float* sum_c = (float*)malloc(sizeof(float) * Ndiscret);
	float* sum2_c = (float*)malloc(sizeof(float) * Ndiscret);
  float* x1_c = (float*)malloc(sizeof(float) * Ndiscret);
  float* x2_c = (float*)malloc(sizeof(float) * Ndiscret);

	cudaMalloc(&time, sizeof(float) * Ndiscret);
	cudaMalloc(&price, sizeof(float) * Ndiscret);
	cudaMalloc(&i_t, sizeof(int) * Ndiscret);
	cudaMalloc(&sum, sizeof(float) * Ndiscret);
	cudaMalloc(&sum2, sizeof(float) * Ndiscret);
  cudaMalloc(&x1, sizeof(float) * Ndiscret);
  cudaMalloc(&x2, sizeof(float) * Ndiscret);

	
	cudaMemset(sum, 0.0f, sizeof(float) * Ndiscret);
	cudaMemset(sum2, 0.0f, sizeof(float) * Ndiscret);

	VarMalloc();
	PostInitDataCMRG();
	
	parameters();

  // GPU timer instructions
	cudaEventCreate(&start);			
	cudaEventCreate(&stop);			
	cudaEventRecord(start,0);		


  // calculate outer trajectories
  int Nblocks = (Nouter+threads_per_block-1)/threads_per_block; // ceiling function
	MCouter_k<<<Nblocks,threads_per_block,2*threads_per_block*sizeof(float)>>>
    (P1, P2, x_0, dt, B, K, leng, M, Nouter, CMRG, time, price, i_t);

  // calculate inner trajectories
  Nblocks = (Ninner+threads_per_block-1)/threads_per_block; // ceiling function
  dim3 dim_blocks(Nouter, Nblocks);
  for(int i = 0; i < M-1; i++){
    MCinner_k<<<dim_blocks, threads_per_block, 2*threads_per_block*sizeof(float)>>>
      (P1, P2, dt, B, K, leng, M, Ninner, CMRG, i+1, time+i*Nouter, price+i*Nouter, i_t+i*Nouter, sum+i*Nouter, sum2+i*Nouter);
  }

  // GPU timer instructions
	cudaEventRecord(stop,0);			
	cudaEventSynchronize(stop);			
	cudaEventElapsedTime(&Tim,			
			 start, stop);				
	cudaEventDestroy(start);			
	cudaEventDestroy(stop);		

  // simulate trajectories for regression
  Nblocks = (Nouter+threads_per_block-1)/threads_per_block; // ceiling function
  dim3 dim_blocks_reg(Nblocks, 2);
  for(int i = 0; i < M-1; i++){
    MCreg_k<<<dim_blocks_reg, threads_per_block>>>
      (P1, P2, dt, B, K, leng, M, Nouter, CMRG, i+1, time+i*Nouter, price+i*Nouter, i_t+i*Nouter, x1+i*Nouter, x2+i*Nouter);
  }

	cudaMemcpy(price_c, price, sizeof(float) * Ndiscret,cudaMemcpyDeviceToHost);
	cudaMemcpy(i_t_c, i_t, sizeof(int) * Ndiscret, cudaMemcpyDeviceToHost);
	cudaMemcpy(time_c, time, sizeof(int) * Ndiscret, cudaMemcpyDeviceToHost);
  cudaMemcpy(sum_c, sum, sizeof(float) * Ndiscret, cudaMemcpyDeviceToHost);
  cudaMemcpy(sum2_c, sum2, sizeof(float) * Ndiscret, cudaMemcpyDeviceToHost);
  cudaMemcpy(x1_c, x1, sizeof(float) * Ndiscret, cudaMemcpyDeviceToHost);
  cudaMemcpy(x2_c, x2, sizeof(float) * Ndiscret, cudaMemcpyDeviceToHost);

  cudaFree(price);
	cudaFree(i_t);
	cudaFree(time);
	cudaFree(sum);
	cudaFree(x1);
  cudaFree(x2);

	FILE* fp;
	fp = fopen("price_c.txt", "w");
	for (unsigned i = 0; i < Ndiscret; i++) {
		fprintf(fp, "%d,%f\n", i, price_c[i]);

	}
	fclose(fp);
	fp = fopen("time_c.txt", "w");
	for (unsigned i = 0; i < Ndiscret; i++) {
		fprintf(fp, "%d,%f\n", i, time_c[i]);}
	fclose(fp);

	fp = fopen("i_t_c.txt", "w");
	for (unsigned i = 0; i < Ndiscret; i++) {
		fprintf(fp, "%d,%d\n", i, i_t_c[i]);}
	fclose(fp);

  fp = fopen("sum_c.txt", "w");
	for (unsigned i = 0; i < Ndiscret; i++) {
		fprintf(fp, "%d,%f\n", i, sum_c[i]);}
	fclose(fp);

  fp = fopen("sum2_c.txt", "w");
	for (unsigned i = 0; i < Ndiscret; i++) {
		fprintf(fp, "%d,%f\n", i, sum2_c[i]);}
	fclose(fp);

  fp = fopen("x1_c.txt", "w");
	for (unsigned i = 0; i < Ndiscret; i++) {
		fprintf(fp, "%d,%f\n", i, x1_c[i]);}
	fclose(fp);

  fp = fopen("x2_c.txt", "w");
	for (unsigned i = 0; i < Ndiscret; i++) {
		fprintf(fp, "%d,%f\n", i, x2_c[i]);}
	fclose(fp);

  printf("All files generated.\n");
	printf("Execution time of nested MC %f ms\n", Tim);

  free(price_c);
  free(time_c);
  free(i_t_c);
  free(sum_c);
  free(sum2_c);
  free(x1_c);
  free(x2_c);

	FreeCMRG();
	FreeVar();
	
	return 0;
}


