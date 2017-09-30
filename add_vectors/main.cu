#include<stdio.h>
#include<stdlib.h>
#define N 35000
#define BLOCK_DIM_SIZE 32
#define MAX_INT_GEN 50
#define EXIT_SUCESS 0
#define EXIT_ERROR -1

void hostProGerarRandomNumbers(int * v, unsigned int n) {
  if (v == NULL || n == 0)
    exit(EXIT_ERROR);

  for(int i = 0; i < n; ++i) 
    v[i] = rand() % (MAX_INT_GEN);
}

__global__ void devProAddVectors(int * va, int * vb, int * vc, int n) {
  //Id da thread no vetor global
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  //Verifica se o id é menor que o tamanho do vetor e soma
  if (id < n)
    vc[id] = va[id] + vb[id];
}


int main(int argc, char **argv) {
  //Pointers de vetor do host
  int *ha, *hb, *hc;

  //Pointers de vetor do device(placa)
  int *da, *db, *dc;

  //Quantidade de bytes alocados
  unsigned int bytesToAlloc = sizeof(int) * N;

  //Alocação dos vetores no host
  ha = (int *) malloc(bytesToAlloc);
  hb = (int *) malloc(bytesToAlloc);
  hc = (int *) malloc(bytesToAlloc);

  //Alocação dos vetores no device
  cudaMalloc((void**)&da, bytesToAlloc);
  cudaMalloc((void**)&db, bytesToAlloc);
  cudaMalloc((void**)&dc, bytesToAlloc);

  //man srand
  srand(time(NULL));
  hostProGerarRandomNumbers(ha, N);
  hostProGerarRandomNumbers(hb, N);
  hostProGerarRandomNumbers(hc, N);

  //Cópia da memória do host pro device (os dados serão operados na memória do device)
  cudaMemcpy(da, ha, bytesToAlloc, cudaMemcpyHostToDevice);
  cudaMemcpy(db, hb, bytesToAlloc, cudaMemcpyHostToDevice);

  //Alocação dos blocos de threads
  // 32 * 32 = 1024 (threads em paralelo)
  // N / blkSize.x = quantidade de blocos na dimensão X
  // N / blkSize.x = quantidade de blocos na dimensão Y
  dim3 blkSize(BLOCK_DIM_SIZE, BLOCK_DIM_SIZE);
  dim3 numBlks((N / blkSize.x) + 1, (N /blkSize.y) + 1);

  //Chamada do kernel <<<Número de blocos, tamanho de cada bloco>>>
  devProAddVectors<<<numBlks, blkSize>>>(da, db, dc, N);

  //Copia memória processada do device para o host
  cudaMemcpy(hc, dc, bytesToAlloc, cudaMemcpyDeviceToHost);

  int ok = 1;
  int vEsperado, vObtido, sI;
  for(int i = 0; i < N; i++) {
    sI = ha[i] + hb[i]; 

    if (sI != hc[i]) {
      ok = 0;
      vEsperado = sI;
      vObtido = hc[i];
      break;
    }
  }

  if (!ok) 
    printf("Solução incorreta. Valor esperado = %d, valor obtido = %d\n", vEsperado, vObtido);
  else
    printf("Solução correta.\n");

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  free(ha);
  free(hb);
  free(hc);

  return EXIT_SUCESS;
}
