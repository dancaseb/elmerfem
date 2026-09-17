/*
 * C wrapper around NVIDIA's cuDSS GPU sparse direct solver, callable
 * from Fortran (see CUDSS_SolveSystem in DirectSolve.F90). 
 * Three entry points (factorize, solve, free) with an opaque
 * handle round-tripped through Fortran as an INTEGER(KIND=AddrInt).
 *
 * Single-GPU only. The CSR matrix is copied to device memory once, at
 * factorize time; only the right-hand side and solution are transferred on
 * every solve.
 */

#include "../config.h"
#ifdef HAVE_CUDSS

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cudss.h>

typedef struct {
  cudssHandle_t handle;
  cudssConfig_t config;
  cudssData_t   data;
  cudssMatrix_t Amat, bmat, xmat;

  int n;

  int    *d_rows;
  int    *d_cols;
  double *d_vals;
  double *d_b;
  double *d_x;
} cudss_t;

static int cuda_ok(cudaError_t st, const char *what)
{
  if (st != cudaSuccess) {
    fprintf(stderr, "CUDSS_SolveSystem: CUDA call '%s' failed: %s\n",
            what, cudaGetErrorString(st));
    return 0;
  }
  return 1;
}

static int cudss_ok(cudssStatus_t st, const char *what)
{
  if (st != CUDSS_STATUS_SUCCESS) {
    fprintf(stderr, "CUDSS_SolveSystem: cuDSS call '%s' failed with status %d\n",
            what, (int)st);
    return 0;
  }
  return 1;
}

static void cudss_teardown(cudss_t *h)
{
  if (!h) return;

  if (h->Amat) cudssMatrixDestroy(h->Amat);
  if (h->bmat) cudssMatrixDestroy(h->bmat);
  if (h->xmat) cudssMatrixDestroy(h->xmat);
  if (h->data) cudssDataDestroy(h->handle, h->data);
  if (h->config) cudssConfigDestroy(h->config);
  if (h->handle) cudssDestroy(h->handle);

  if (h->d_rows) cudaFree(h->d_rows);
  if (h->d_cols) cudaFree(h->d_cols);
  if (h->d_vals) cudaFree(h->d_vals);
  if (h->d_b) cudaFree(h->d_b);
  if (h->d_x) cudaFree(h->d_x);

  free(h);
}

/* mtype: 0 = general (nonsymmetric), 1 = symmetric (indefinite or
 * structurally symmetric), 2 = symmetric positive definite. The caller
 * (CUDSS_SolveSystem) is responsible for passing only the upper triangle
 * (including the diagonal) of rows/cols/vals when mtype is 1 or 2 */
cudss_t *FC_FUNC_(cudss_ffactorize,CUDSS_FFACTORIZE)
    (int *n, int *nnz, int *rows, int *cols, double *vals, int *mtype)
{
  cudss_t *h;
  cudssMatrixType_t mt;
  cudssMatrixViewType_t mv;

  h = (cudss_t *)calloc(1, sizeof(cudss_t));
  if (!h) {
    fprintf(stderr, "CUDSS_SolveSystem: out of host memory\n");
    return NULL;
  }
  h->n = *n;

  switch (*mtype) {
    case 2:  mt = CUDSS_MTYPE_SPD;       mv = CUDSS_MVIEW_UPPER; break;
    case 1:  mt = CUDSS_MTYPE_SYMMETRIC; mv = CUDSS_MVIEW_UPPER; break;
    default: mt = CUDSS_MTYPE_GENERAL;   mv = CUDSS_MVIEW_FULL;  break;
  }

  if (!cuda_ok(cudaMalloc((void **)&h->d_rows, (size_t)(*n + 1) * sizeof(int)), "cudaMalloc(rows)") ||
      !cuda_ok(cudaMalloc((void **)&h->d_cols, (size_t)(*nnz) * sizeof(int)), "cudaMalloc(cols)") ||
      !cuda_ok(cudaMalloc((void **)&h->d_vals, (size_t)(*nnz) * sizeof(double)), "cudaMalloc(vals)") ||
      !cuda_ok(cudaMalloc((void **)&h->d_b, (size_t)(*n) * sizeof(double)), "cudaMalloc(b)") ||
      !cuda_ok(cudaMalloc((void **)&h->d_x, (size_t)(*n) * sizeof(double)), "cudaMalloc(x)")) {
    cudss_teardown(h);
    return NULL;
  }

  if (!cuda_ok(cudaMemcpy(h->d_rows, rows, (size_t)(*n + 1) * sizeof(int), cudaMemcpyHostToDevice), "memcpy(rows)") ||
      !cuda_ok(cudaMemcpy(h->d_cols, cols, (size_t)(*nnz) * sizeof(int), cudaMemcpyHostToDevice), "memcpy(cols)") ||
      !cuda_ok(cudaMemcpy(h->d_vals, vals, (size_t)(*nnz) * sizeof(double), cudaMemcpyHostToDevice), "memcpy(vals)")) {
    cudss_teardown(h);
    return NULL;
  }

  if (!cudss_ok(cudssCreate(&h->handle), "cudssCreate") ||
      !cudss_ok(cudssConfigCreate(&h->config), "cudssConfigCreate") ||
      !cudss_ok(cudssDataCreate(h->handle, &h->data), "cudssDataCreate")) {
    cudss_teardown(h);
    return NULL;
  }

  /* Standard (gap-free) CSR: row i's entries run from d_rows[i] to
   * d_rows[i+1]-1, so rowEnd is simply rowStart shifted by one int. Elmer's
   * CRS row pointers are already 1-based, which cuDSS supports natively via
   * CUDSS_BASE_ONE -- no reindexing needed. */
  if (!cudss_ok(cudssMatrixCreateCsr(&h->Amat, *n, *n, *nnz,
                    h->d_rows, h->d_rows + 1, h->d_cols, h->d_vals,
                    CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F, mt, mv, CUDSS_BASE_ONE),
                "cudssMatrixCreateCsr") ||
      !cudss_ok(cudssMatrixCreateDn(&h->bmat, *n, 1, *n, h->d_b, CUDSS_R_64F,
                    CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn(b)") ||
      !cudss_ok(cudssMatrixCreateDn(&h->xmat, *n, 1, *n, h->d_x, CUDSS_R_64F,
                    CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn(x)")) {
    cudss_teardown(h);
    return NULL;
  }

  if (!cudss_ok(cudssExecute(h->handle, CUDSS_PHASE_ANALYSIS, h->config, h->data,
                    h->Amat, h->xmat, h->bmat), "cudssExecute(ANALYSIS)") ||
      !cudss_ok(cudssExecute(h->handle, CUDSS_PHASE_FACTORIZATION, h->config, h->data,
                    h->Amat, h->xmat, h->bmat), "cudssExecute(FACTORIZATION)")) {
    cudss_teardown(h);
    return NULL;
  }

  return h;
}

void FC_FUNC_(cudss_fsolve,CUDSS_FSOLVE)(cudss_t **handle, int *n, double *x, double *b)
{
  cudss_t *h = *handle;

  if (!h) {
    fprintf(stderr, "CUDSS_SolveSystem: cudss_fsolve called without a factorization\n");
    return;
  }

  if (!cuda_ok(cudaMemcpy(h->d_b, b, (size_t)(*n) * sizeof(double), cudaMemcpyHostToDevice),
               "memcpy(b)"))
    return;

  if (!cudss_ok(cudssExecute(h->handle, CUDSS_PHASE_SOLVE, h->config, h->data,
                    h->Amat, h->xmat, h->bmat), "cudssExecute(SOLVE)"))
    return;

  cuda_ok(cudaMemcpy(x, h->d_x, (size_t)(*n) * sizeof(double), cudaMemcpyDeviceToHost),
          "memcpy(x)");
}

void FC_FUNC_(cudss_ffree,CUDSS_FFREE)(cudss_t **handle)
{
  cudss_teardown(*handle);
  *handle = NULL;
}

#endif /* HAVE_CUDSS */
