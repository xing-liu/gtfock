#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>
#include <macdecls.h>
#include <sys/time.h>
#include "config.h"
#include "taskq.h"
#include "fock_task.h"

#ifdef __INTEL_OFFLOAD
#include <offload.h>
#ifdef __OFFLOAD_DEBUG
#define __USE_GNU
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#include <ucontext.h>
#include <sys/time.h>
#include <sys/resource.h>
#endif /* __OFFLOAD_DEBUG */
#endif /* __INTEL_OFFLOAD */


#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(push, target(mic))
#endif

static inline void atomic_add_f64(volatile double* global_value, const double addend)
{
    uint64_t expected_value, new_value;
    do {
        const double old_value = *global_value;
        expected_value = _castf64_u64(old_value);
        new_value = _castf64_u64(old_value + addend);
    } while (!__sync_bool_compare_and_swap((volatile uint64_t*)global_value, expected_value, new_value));
}


void update_F (double *integrals, int dimM, int dimN,
               int dimP, int dimQ,
               int flag1, int flag2, int flag3,
               int iMN, int iPQ, int iMP, int iNP, int iMQ, int iNQ,
               int iMP0, int iMQ0, int iNP0,
               double *D1, double *D2, double *D3,
               double *F_MN, double *F_PQ, double *F_NQ,
               double *F_MP, double *F_MQ, double *F_NP,
               int ldMN, int ldPQ, int ldNQ,
               int ldMP, int ldMQ, int ldNP)
{
    int iM;
    int iN;
    int iP;
    int iQ;
    double I;
    int flag4;
    int flag5;
    int flag6;
    int flag7;
    double *D_MN;
    double *D_PQ;
    double *D_MP;
    double *D_NP;
    double *D_MQ;
    double *D_NQ;    
    double *J_MN;
    double *J_PQ;
    double *K_MP;
    double *K_NP;
    double *K_MQ;
    double *K_NQ;

    flag4 = (flag1 == 1 && flag2 == 1) ? 1 : 0;
    flag5 = (flag1 == 1 && flag3 == 1) ? 1 : 0;
    flag6 = (flag2 == 1 && flag3 == 1) ? 1 : 0;
    flag7 = (flag4 == 1 && flag3 == 1) ? 1 : 0;                    

    D_MN = D1 + iMN;
    D_PQ = D2 + iPQ;
    D_NQ = D3 + iNQ;
    D_MP = D3 + iMP0;
    D_MQ = D3 + iMQ0;
    D_NP = D3 + iNP0;    
    J_MN = F_MN + iMN;
    J_PQ = F_PQ + iPQ;
    K_NQ = F_NQ + iNQ;
    K_MP = F_MP + iMP;
    K_MQ = F_MQ + iMQ;
    K_NP = F_NP + iNP;
    
    for (iN = 0; iN < dimN; iN++)
    {
        for (iQ = 0; iQ < dimQ; iQ++)
        {
            int inq = iN * ldNQ + iQ;
            double k_NQ = 0;
            for (iM = 0; iM < dimM; iM++)
            {
                int imn = iM * ldMN + iN;
                int imq = iM * ldMQ + iQ;
                double j_MN = 0;
                double k_MQ = 0;

                for (iP = 0; iP < dimP; iP++)
                {
                    int ipq = iP * ldPQ + iQ;
                    int imp = iM * ldMP + iP;
                    int inp = iN * ldNP + iP;

                    I = integrals[iM + dimM * (iN + dimN * (iP + dimP * iQ))];
                    // F(m, n) += D(p, q) * 2 * I(m, n, p, q)
                    // F(n, m) += D(p, q) * 2 * I(n, m, p, q)
                    // F(m, n) += D(q, p) * 2 * I(m, n, q, p)
                    // F(n, m) += D(q, p) * 2 * I(n, m, q, p)
                    double vMN = 2.0 * (1 + flag1 + flag2 + flag4) *
                        D_PQ[iP * ldPQ + iQ] * I;
                    j_MN += vMN;

                    // F(p, q) += D(m, n) * 2 * I(p, q, m, n)
                    // F(p, q) += D(n, m) * 2 * I(p, q, n, m)
                    // F(q, p) += D(m, n) * 2 * I(q, p, m, n)
                    // F(q, p) += D(n, m) * 2 * I(q, p, n, m)
                    double vPQ = 2.0 * (flag3 + flag5 + flag6 + flag7) *
                        D_MN[iM * ldMN + iN] * I;
                    atomic_add_f64(&J_PQ[ipq], vPQ);
                    // F(m, p) -= D(n, q) * I(m, n, p, q)
                    // F(p, m) -= D(q, n) * I(p, q, m, n)
                    double vMP = (1 + flag3) *
                        1.0 * D_NQ[iN * ldNQ + iQ] * I;
                    atomic_add_f64(&K_MP[imp], -vMP);
                    // F(n, p) -= D(m, q) * I(n, m, p, q)
                    // F(p, n) -= D(q, m) * I(p, q, n, m)
                    double vNP = (flag1 + flag5) *
                        1.0 * D_MQ[iM * ldNQ + iQ] * I;
                    atomic_add_f64(&K_NP[inp], -vNP);
                    // F(m, q) -= D(n, p) * I(m, n, q, p)
                    // F(q, m) -= D(p, n) * I(q, p, m, n)
                    double vMQ = (flag2 + flag6) *
                        1.0 * D_NP[iN * ldNQ + iP] * I;
                    k_MQ -= vMQ;
                    // F(n, q) -= D(m, p) * I(n, m, q, p)
                    // F(q, n) -= D(p, m) * I(q, p, n, m)
                    double vNQ = (flag4 + flag7) *
                        1.0 * D_MP[iM * ldNQ + iP] * I;
                    k_NQ -= vNQ;
                }
                atomic_add_f64(&J_MN[imn], j_MN);
                atomic_add_f64(&K_MQ[imq], k_MQ);
            }
            atomic_add_f64(&K_NQ[inq], k_NQ);
        }
    }
}

#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif


// for SCF, J = K
void fock_task (BasisSet_t basis, ERD_t erd, int ncpu_f,
                int *shellptr, double *shellvalue,
                int *shellid, int *shellrid, int *f_startind,
                int *rowpos, int *colpos, int *rowptr, int *colptr,
                double tolscr2, int startrow, int startcol,
                int startM, int endM, int startP, int endP,
                double *D1, double *D2, double *D3,
                double *F1, double *F2, double *F3,
                double *F4, double *F5, double *F6, 
                int ldX1, int ldX2, int ldX3,
                int ldX4, int ldX5, int ldX6,
                int sizeX1, int sizeX2, int sizeX3,
                int sizeX4, int sizeX5, int sizeX6,
                double *nitl, double *nsq)              
{
    int startMN = shellptr[startM];
    int endMN = shellptr[endM + 1];
    int startPQ = shellptr[startP];
    int endPQ = shellptr[endP + 1];
    
    #pragma omp parallel
    {
        // init    
        int nt = omp_get_thread_num ();
        int nf = nt/ncpu_f;
        double *F_MN = &(F1[nf * sizeX1]);
        double *F_PQ = &(F2[nf * sizeX2]);
        double *F_NQ = F3;
        double *F_MP = &(F4[nf * sizeX4]);
        double *F_MQ = &(F5[nf * sizeX5]);
        double *F_NP = &(F6[nf * sizeX6]);
        double mynsq = 0.0;
        double mynitl = 0.0;
        
        #pragma omp for schedule(dynamic)
        for (int i = startMN; i < endMN; i++)
        {
            int M = shellrid[i];
            int N = shellid[i];
            double value1 = shellvalue[i];            
            int dimM = f_startind[M + 1] - f_startind[M];
            int dimN = f_startind[N + 1] - f_startind[N];
            int iX1M = f_startind[M] - f_startind[startrow];           
            int iX3M = rowpos[M]; 
            int iXN = rowptr[i];
            int iMN = iX1M * ldX1+ iXN;
            int flag1 = (value1 < 0.0) ? 1 : 0;
            if (nt == 0)
            {
                int flag;
                MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG,
                           MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
            }
            for (int j = startPQ; j < endPQ; j++)
            {
                int P = shellrid[j];
                int Q = shellid[j];
                if ((M > P && (M + P) % 2 == 1) || 
                    (M < P && (M + P) % 2 == 0))
                    continue;                
                if ((M == P) &&
                    ((N > Q && (N + Q) % 2 == 1) ||
                    (N < Q && (N + Q) % 2 == 0)))
                    continue;
                double value2 = shellvalue[j];
                int dimP = f_startind[P + 1] - f_startind[P];
                int dimQ =  f_startind[Q + 1] - f_startind[Q];
                int iX2P = f_startind[P] - f_startind[startcol];
                int iX3P = colpos[P];
                int iXQ = colptr[j];               
                int iPQ = iX2P * ldX2+ iXQ;                             
                int iNQ = iXN * ldX3 + iXQ;                
                int iMP = iX1M * ldX4 + iX2P;
                int iMQ = iX1M * ldX5 + iXQ;
                int iNP = iXN * ldX6 + iX2P;
                int iMP0 = iX3M * ldX3 + iX3P;
                int iMQ0 = iX3M * ldX3 + iXQ;
                int iNP0 = iXN * ldX3 + iX3P;               
                int flag3 = (M == P && Q == N) ? 0 : 1;                    
                int flag2 = (value2 < 0.0) ? 1 : 0;
                if (fabs(value1 * value2) >= tolscr2)
                {
                    int nints;
                    double *integrals;
                    mynsq += 1.0;
                    mynitl += dimM*dimN*dimP*dimQ;                       
                    CInt_computeShellQuartet (basis, erd, nt,
                                              M, N, P, Q, &integrals, &nints);
                    if (nints != 0)
                    {
                        update_F (integrals, dimM, dimN, dimP, dimQ,
                                  flag1, flag2, flag3,
                                  iMN, iPQ, iMP, iNP, iMQ, iNQ,
                                  iMP0, iMQ0, iNP0,
                                  D1, D2, D3,
                                  F_MN, F_PQ, F_NQ,
                                  F_MP, F_MQ, F_NP,
                                  ldX1, ldX2, ldX3,
                                  ldX4, ldX5, ldX6);
                    }
                }
            }
        }

        #pragma omp critical
        {
            *nitl += mynitl;
            *nsq += mynsq;
        }
    } /* #pragma omp parallel */
}


void reset_F (int numF, double *F1, double *F2, double *F3,
              double *F4, double *F5, double *F6,
              int sizeX1, int sizeX2, int sizeX3,
              int sizeX4, int sizeX5, int sizeX6)
{
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int k = 0; k < numF * sizeX1; k++)
        {
            F1[k] = 0.0;    
        }
        #pragma omp for nowait
        for (int k = 0; k < numF * sizeX2; k++)
        {
            F2[k] = 0.0;
        }
        #pragma omp for nowait
        for (int k = 0; k < sizeX3; k++)
        {
            F3[k] = 0.0;
        }
        #pragma omp for nowait
        for (int k = 0; k < numF * sizeX4; k++)
        {
            F4[k] = 0.0;
        }
        #pragma omp for nowait
        for (int k = 0; k < numF * sizeX5; k++)
        {
            F5[k] = 0.0;
        }
        #pragma omp for nowait
        for (int k = 0; k < numF * sizeX6; k++)
        {
            F6[k] = 0.0;
        }
    }
}


void reduce_F (int numF, double *F1, double *F2, double *F3,
               double *F4, double *F5, double *F6,
               int sizeX1, int sizeX2, int sizeX3,
               int sizeX4, int sizeX5, int sizeX6,
               int maxrowsize, int maxcolsize,
               int maxrowfuncs, int maxcolfuncs,
               int iX3M, int iX3P,
               int ldX3, int ldX4, int ldX5, int ldX6)
{
    #pragma omp parallel
    {
        #pragma omp for
        for (int k = 0; k < sizeX1; k++)
        {
            for (int p = 1; p < numF; p++)
            {
                F1[k] += F1[k + p * sizeX1];
            }
        }
        #pragma omp for
        for (int k = 0; k < sizeX2; k++)
        {
            for (int p = 1; p < numF; p++)
            {
                F2[k] += F2[k + p * sizeX2];
            }
        }
        #pragma omp for
        for (int k = 0; k < sizeX4; k++)
        {
            for (int p = 1; p < numF; p++)
            {
                F4[k] += F4[k + p * sizeX4];   
            }
        }
        #pragma omp for
        for (int k = 0; k < sizeX5; k++)
        {
            for (int p = 1; p < numF; p++)
            {
                F5[k] += F5[k + p * sizeX5];   
            }
        }
        #pragma omp for
        for (int k = 0; k < sizeX6; k++)
        {
            for (int p = 1; p < numF; p++)
            {
                F6[k] += F6[k + p * sizeX6];   
            }
        }

        int iMP = iX3M * ldX3 + iX3P;
        int iMQ = iX3M * ldX3;
        int iNP = iX3P;
        #pragma omp for
        for (int iM = 0; iM < maxrowfuncs; iM++)
        {
            for (int iP = 0; iP < maxcolfuncs; iP++)
            {
                F3[iMP + iM * ldX3 + iP] += F4[iM * ldX4 + iP];
            }
            for (int iQ = 0; iQ < maxcolsize; iQ++)
            {
                F3[iMQ + iM * ldX3 + iQ] += F5[iM * ldX5 + iQ];    
            }
        }
        #pragma omp for
        for (int iN = 0; iN < maxrowsize; iN++)
        {
            for (int iP = 0; iP < maxcolfuncs; iP++)
            {
                F3[iNP + iN * ldX3 + iP] += F6[iN * ldX6 + iP];
            }
        }
    }
}


#ifdef __INTEL_OFFLOAD

#define CHUNK_SIZE 8
#define MIC_MIN_WORK_SIZE 20
#define ALLOC alloc_if(1) free_if(0)
#define REUSE alloc_if(0) free_if(0)
#define ONCE  alloc_if(1) free_if(1)
#define FREE  alloc_if(0) free_if(1)

#pragma offload_attribute(push, target(mic))

pfock_mic_t pfock_mic;

#ifdef __OFFLOAD_DEBUG
void sigsegv_handler(int signum, siginfo_t *info, void *ctx)
{
    ucontext_t* uc = (ucontext_t*)ctx;
	printf("SEGMENTATION FAULT\n");
    printf("\tSignal number: %d\n", info->si_signo);
    printf("\tSignal code: %d\n", info->si_code);
    printf("\tStack address: %p\n", uc->uc_mcontext.gregs[REG_RSP]);
    fflush (0);
    struct rlimit stackLimit;
    if (getrlimit(RLIMIT_STACK, &stackLimit) == 0) {
        printf("\tCurrent stack size: %zu\n", (size_t)stackLimit.rlim_cur);
        printf("\tMax stack size: %zu\n", (size_t)stackLimit.rlim_max);
    }
    ssize_t stack_offset = (uint8_t*)info->si_addr - (uint8_t*)uc->uc_mcontext.gregs[REG_RSP];
    printf("\tFailed data address: %p\n", info->si_addr);
    if (stack_offset > 0) {
        printf("\t\t%zu bytes above the stack pointer\n", stack_offset);
    } else {
        printf("\t\t%zu bytes below the stack pointer\n", -stack_offset);
        if (stack_offset >= -4096) {
            printf("\t\t!!!!! LOOKS LIKE STACK OVERFLOW !!!!!\n");
        }
    }
    printf("\tFailed code address: %p\n", uc->uc_mcontext.gregs[REG_RIP]);
    fflush(0);
    uint8_t* code = (uint8_t*)uc->uc_mcontext.gregs[REG_RIP];
    printf("\tCode bytes: %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X\n", code[0], code[1], code[2], code[3], code[4], code[5], code[6], code[7], code[8], code[9], code[10], code[11], code[12], code[13], code[14]);
    fflush(0);
    abort();
}
#endif


static void compute_chunk (BasisSet_t basis, ERD_t erd,
                    int *shellptr, double *shellvalue,
                    int *shellid, int *shellrid, int *f_startind,
                    int *rowpos, int *colpos, int *rowptr, int *colptr,
                    double tolscr2, int startrow, int startcol,
                    int startChunkMN, int endChunkMN, int startChunkPQ, int endChunkPQ,
                    double *D1, double *D2, double *D3,
                    double *F1, double *F2, double *F3,
                    double *F4, double *F5, double *F6, 
                    int ldX1, int ldX2, int ldX3,
                    int ldX4, int ldX5, int ldX6, int nt)
{
    for (int i = startChunkMN; i < endChunkMN; i++)
    {
        int M = shellrid[i];
        int N = shellid[i];
        double value1 = shellvalue[i];
        int dimM = f_startind[M + 1] - f_startind[M];
        int dimN = f_startind[N + 1] - f_startind[N];
        int iX3M = rowpos[M];
        int iX1M = f_startind[M] - f_startind[startrow];
        int iXN = rowptr[i];
        int iMN = iX1M * ldX1 + iXN;
        int flag1 = (value1 < 0.0) ? 1 : 0;
    #ifndef __MIC__
        if (nt == 0)
        {
            int mpiflag;
            MPI_Iprobe (MPI_ANY_SOURCE, MPI_ANY_TAG,
                        MPI_COMM_WORLD, &mpiflag, MPI_STATUS_IGNORE);
        }
    #endif
        for (int j = startChunkPQ; j < endChunkPQ; j++)
        {
            int P = shellrid[j];
            int Q = shellid[j];
            if ((M > P && (M + P) % 2 == 1) ||
                    (M < P && (M + P) % 2 == 0))
                continue;
            if ((M == P) &&
                ((N > Q && (N + Q) % 2 == 1) ||
                (N < Q && (N + Q) % 2 == 0)))
                continue;
            double value2 = shellvalue[j];
            int dimP = f_startind[P + 1] - f_startind[P];
            int dimQ = f_startind[Q + 1] - f_startind[Q];
            int iX3P = colpos[P];
            int iX2P = f_startind[P] - f_startind[startcol];
            int iXQ = colptr[j];
            int iPQ = iX2P * ldX2 + iXQ;
            int iNQ = iXN * ldX3 + iXQ;           
            int iMP = iX1M * ldX4 + iX2P;
            int iMQ = iX1M * ldX5 + iXQ;
            int iNP = iXN * ldX6 + iX2P;
            int iMP0 = iX3M * ldX3 + iX3P;
            int iMQ0 = iX3M * ldX3 + iXQ;
            int iNP0 = iXN * ldX3 + iX3P; 
            int flag3 = (M == P && Q == N) ? 0 : 1;
            int flag2 = (value2 < 0.0) ? 1 : 0;
            if (fabs (value1 * value2) >= tolscr2)
            {
                int nints;
                double *integrals;
                CInt_computeShellQuartet (basis, erd, nt,
                        M, N, P, Q, &integrals, &nints);
                if (nints != 0)
                {              
                    update_F (integrals, dimM, dimN, dimP, dimQ,
                              flag1, flag2, flag3,
                              iMN, iPQ, iMP, iNP, iMQ, iNQ,
                              iMP0, iMQ0, iNP0,
                              D1, D2, D3, F1, F2, F3, F4, F5, F6,
                              ldX1, ldX2, ldX3, ldX4, ldX5, ldX6);
                }
            }
        }
    }
}



static void compute_block_of_chunks (BasisSet_t basis, ERD_t erd,
                   int *shellptr, double *shellvalue,
                   int *shellid, int *shellrid, int *f_startind,
                   int *rowpos, int *colpos, int *rowptr, int *colptr,
                   double tolscr2, int startrow, int startcol,
                   int startMN, int endMN, int startPQ, int endPQ,
                   int startChunk, int endChunk, int chunksPQ,
                   double *D1, double *D2, double *D3,
                   int ldX1, int ldX2, int ldX3,
                   int ldX4, int ldX5, int ldX6)
{
    int k;

    #pragma omp parallel
    {
        int tid;
        
        tid = omp_get_thread_num ();
        int my_f = tid/pfock_mic.nmic_f;
        double *F_MN = pfock_mic.F1[my_f];
        double *F_PQ = pfock_mic.F2[my_f];
        double *F_NQ = pfock_mic.F3;
        double *F_MP = pfock_mic.F4[my_f];
        double *F_MQ = pfock_mic.F5[my_f];
        double *F_NP = pfock_mic.F6[my_f];

        #pragma omp for schedule(dynamic)
        for(k = startChunk; k < endChunk; k++)
        {
            int chunkIdMN = k / chunksPQ;
            int chunkIdPQ = k % chunksPQ;

            int startChunkMN = startMN + chunkIdMN * CHUNK_SIZE;
            int endChunkMN   = startChunkMN + CHUNK_SIZE;
            if(endChunkMN > endMN) endChunkMN = endMN;
            int startChunkPQ = startPQ + chunkIdPQ * CHUNK_SIZE;
            int endChunkPQ   = startChunkPQ + CHUNK_SIZE;
            if(endChunkPQ > endPQ) endChunkPQ = endPQ;

            compute_chunk (basis, erd,
                    shellptr, shellvalue,
                    shellid, shellrid, f_startind,
                    rowpos, colpos, rowptr, colptr,
                    tolscr2, startrow, startcol,
                    startChunkMN, endChunkMN, startChunkPQ, endChunkPQ,
                    D1, D2, D3, F_MN, F_PQ, F_NQ, F_MP, F_MQ, F_NP,
                    ldX1, ldX2, ldX3, ldX4, ldX5, ldX6, tid);
        }
    }
}

#pragma offload_attribute(pop)


void offload_reset_F (int mic_numdevs)
{
    for(int mic_id = 0; mic_id < mic_numdevs; mic_id++)
    {
        #pragma offload target(mic:mic_id)\
                nocopy(pfock_mic) \
                signal(mic_id)
        {
            int num_F = pfock_mic.num_F;
            double **F1 = pfock_mic.F1;
            double **F2 = pfock_mic.F2;
            double *F3 = pfock_mic.F3;
            double **F4 = pfock_mic.F4;
            double **F5 = pfock_mic.F5;
            double **F6 = pfock_mic.F6;
            int sizeD1 = pfock_mic.sizeD1;
            int sizeD2 = pfock_mic.sizeD2;
            int sizeD3 = pfock_mic.sizeD3;
            int sizeD4 = pfock_mic.sizeD4;
            int sizeD5 = pfock_mic.sizeD5;
            int sizeD6 = pfock_mic.sizeD6;
            #pragma omp parallel
            {
                for (int i = 0; i < num_F; i++)
                {
                    #pragma omp for
                    for (int j = 0; j < sizeD1; j++)
                    {
                        F1[i][j] = 0.0;
                    }
                    #pragma omp for
                    for (int j = 0; j < sizeD2; j++)
                    {
                        F2[i][j] = 0.0;
                    }
                    #pragma omp for
                    for (int j = 0; j < sizeD4; j++)
                    {
                        F4[i][j] = 0.0;
                    }
                    #pragma omp for
                    for (int j = 0; j < sizeD5; j++)
                    {
                        F5[i][j] = 0.0;
                    }
                    #pragma omp for
                    for (int j = 0; j < sizeD6; j++)
                    {
                        F6[i][j] = 0.0;
                    }
                }             
                #pragma omp for
                for (int j = 0; j < sizeD3; j++)
                {
                    F3[j] = 0.0;
                }
            }
        }
    }   
}


void offload_fock_task (int mic_numdevs,
                        BasisSet_t basis, ERD_t erd, int ncpu_f,
                        int *shellptr, double *shellvalue,
                        int *shellid, int *shellrid, int *f_startind,
                        int *rowpos, int *colpos, int *rowptr, int *colptr,
                        double tolscr2, int startrow, int startcol,
                        int startM, int endM, int startP, int endP,
                        double *D1, double *D2, double *D3,
                        double *F1, double *F2, double *F3,
                        double *F4, double *F5, double *F6, 
                        int ldX1, int ldX2, int ldX3,
                        int ldX4, int ldX5, int ldX6,
                        int sizeX1, int sizeX2, int sizeX3,
                        int sizeX4, int sizeX5, int sizeX6,
                        double mic_fraction)
{
    int startMN;
    int endMN;
    int startPQ;
    int endPQ;
    startMN = shellptr[startM];
    endMN = shellptr[endM + 1];
    startPQ = shellptr[startP];
    endPQ = shellptr[endP + 1];

    int chunksMN = ((endMN - startMN) + CHUNK_SIZE -1) / CHUNK_SIZE;
    int chunksPQ = ((endPQ - startPQ) + CHUNK_SIZE -1) / CHUNK_SIZE;    
    int totalChunks = chunksMN * chunksPQ;
    int head = 0;
    int initialChunksMIC = totalChunks * mic_fraction;
    head = mic_numdevs * initialChunksMIC;    
    #pragma omp parallel
    {
        int my_chunk;
        int tid;
        int nthreads;
        tid = omp_get_thread_num ();
        int nf = tid/ncpu_f;
        nthreads = omp_get_num_threads ();
        double *F_MN = &(F1[nf * sizeX1]);
        double *F_PQ = &(F2[nf * sizeX2]);
        double *F_NQ = F3;
        double *F_MP = &(F4[nf * sizeX4]);
        double *F_MQ = &(F5[nf * sizeX5]);
        double *F_NP = &(F6[nf * sizeX6]);

        if(tid == nthreads - 1)
        {
            int signalled[mic_numdevs];
            int mic_id;
            int startChunk = 0;
            int endChunk;
            int dummy_tag = 0;
            int *finish_tag = &dummy_tag;
            for(mic_id = 0; mic_id < mic_numdevs; mic_id++)
            {
                int endChunk = startChunk + initialChunksMIC;
                #pragma offload target(mic:mic_id) \
                    nocopy(basis_mic, erd_mic) \
                    in(shellptr, shellvalue, shellid, shellrid: length(0) REUSE) \
                    in(f_startind, rowpos, colpos, rowptr, colptr: length(0) REUSE) \
                    in(tolscr2, startrow, startcol) \
                    in(startMN, endMN, startPQ, endPQ) \
                    in(startChunk, endChunk, chunksPQ) \
                    in(D1, D2, D3: length(0) REUSE) \
                    in(ldX1, ldX2, ldX3, sizeX1, sizeX2, sizeX3)
                    //signal(finish_tag)
                compute_block_of_chunks (basis_mic, erd_mic,
                        shellptr, shellvalue,
                        shellid, shellrid, f_startind,
                        rowpos, colpos, rowptr, colptr,
                        tolscr2, startrow, startcol,
                        startMN, endMN, startPQ, endPQ,
                        startChunk, endChunk, chunksPQ, 
                        D1, D2, D3,
                        ldX1, ldX2, ldX3, ldX4, ldX5, ldX6);
                signalled[mic_id] = 1;
                startChunk += initialChunksMIC;
            }

            int mic_numdevs_active = mic_numdevs;

            while(mic_numdevs_active > 0)
            {
                for(mic_id = 0; mic_id < mic_numdevs; mic_id++)
                {
                    if(signalled[mic_id] == 1)
                    {
                        //int sig = _Offload_signaled (mic_id, finish_tag);
                        int chunksMIC = 0;
                        int sig = 1;

                        if (sig != 0)
                        {
                            signalled[mic_id] = 0;
                            #pragma omp critical
                            {
                                int remTasks = totalChunks - head;
                                chunksMIC = remTasks * mic_fraction;
                                if(chunksMIC < MIC_MIN_WORK_SIZE)
                                    chunksMIC = 0;
                                my_chunk = head;
                                head += chunksMIC;
                            }

                            if(chunksMIC > 0)
                            {
                                startChunk = my_chunk;
                                endChunk = startChunk + chunksMIC;
                                #pragma offload target(mic:mic_id) \
                                    nocopy(basis_mic, erd_mic) \
                                    in(shellptr, shellvalue, shellid, shellrid: length(0) REUSE) \
                                    in(f_startind, rowpos, colpos, rowptr, colptr: length(0) REUSE) \
                                    in(tolscr2, startrow, startcol) \
                                    in(startMN, endMN, startPQ, endPQ) \
                                    in(startChunk, endChunk, chunksPQ) \
                                    in(D1, D2, D3: length(0) REUSE) \
                                    in(ldX1, ldX2, ldX3, sizeX1, sizeX2, sizeX3)
                                    //signal(finish_tag)
                                compute_block_of_chunks (basis_mic, erd_mic,
                                        shellptr, shellvalue,
                                        shellid, shellrid, f_startind,
                                        rowpos, colpos, rowptr, colptr,
                                        tolscr2, startrow, startcol,
                                        startMN, endMN, startPQ, endPQ,
                                        startChunk, endChunk, chunksPQ, 
                                        D1, D2, D3,
                                        ldX1, ldX2, ldX3, ldX4, ldX5, ldX6);
                                signalled[mic_id] = 1;
                            }
                            else
                            {
                                mic_numdevs_active--;
                            }

                        }
                    }
                }
            }
            while(1)
            {
                #pragma omp critical
                {
                    my_chunk = head;
                    head++;
                }
                if(my_chunk >= totalChunks) break;

                int chunkIdMN = my_chunk / chunksPQ;
                int chunkIdPQ = my_chunk % chunksPQ;
                int startChunkMN =  startMN + chunkIdMN * CHUNK_SIZE;
                int endChunkMN   = startChunkMN + CHUNK_SIZE;
                if(endChunkMN > endMN) endChunkMN = endMN;
                int startChunkPQ =  startPQ + chunkIdPQ * CHUNK_SIZE;
                int endChunkPQ   = startChunkPQ + CHUNK_SIZE;
                if(endChunkPQ > endPQ) endChunkPQ = endPQ;
                compute_chunk (basis, erd,
                        shellptr, shellvalue,
                        shellid, shellrid, f_startind,
                        rowpos, colpos, rowptr, colptr,
                        tolscr2, startrow, startcol,
                        startChunkMN, endChunkMN, startChunkPQ, endChunkPQ,
                        D1, D2, D3, F_MN, F_PQ, F_NQ, F_MP, F_MQ, F_NP,
                        ldX1, ldX2, ldX3, ldX4, ldX5, ldX6, tid);
            }
        }
        else
        {
            while(1)
            {
                #pragma omp critical
                {
                    my_chunk = head;
                    head++;
                }
                if(my_chunk >= totalChunks) break;

                int chunkIdMN = my_chunk / chunksPQ;
                int chunkIdPQ = my_chunk % chunksPQ;
                int startChunkMN =  startMN + chunkIdMN * CHUNK_SIZE;
                int endChunkMN   = startChunkMN + CHUNK_SIZE;
                if(endChunkMN > endMN) endChunkMN = endMN;
                int startChunkPQ =  startPQ + chunkIdPQ * CHUNK_SIZE;
                int endChunkPQ   = startChunkPQ + CHUNK_SIZE;
                if(endChunkPQ > endPQ) endChunkPQ = endPQ;
                compute_chunk (basis, erd,
                        shellptr, shellvalue,
                        shellid, shellrid, f_startind,
                        rowpos, colpos, rowptr, colptr,
                        tolscr2, startrow, startcol,
                        startChunkMN, endChunkMN, startChunkPQ, endChunkPQ,
                        D1, D2, D3, F_MN, F_PQ, F_NQ, F_MP, F_MQ, F_NP,
                        ldX1, ldX2, ldX3, ldX4, ldX5, ldX6, tid);
            }
        }
    } /* #pragma omp parallel */

}


void offload_reduce_mic (int mic_numdevs,
                         int nfuncs_row, int nfuncs_col,
                         int iX3M, int iX3P,
                         double *F1_offload,
                         double *F2_offload,
                         double *F3_offload,
                         int sizeD1, int sizeD2, int sizeD3)
{
    for(int mic_id = 0; mic_id < mic_numdevs; mic_id++)
    {
        #pragma offload target(mic:mic_id)\
            nocopy(pfock_mic)\
            out(F1_offload[0:sizeD1]: into(F1_offload[mic_id*sizeD1:sizeD1]))\
            out(F2_offload[0:sizeD2]: into(F2_offload[mic_id*sizeD2:sizeD2]))\
            out(F3_offload[0:sizeD3]: into(F3_offload[mic_id*sizeD3:sizeD3]))\
            in(iX3M, iX3P, nfuncs_row, nfuncs_col) \
            signal(mic_id)
        {
            int num_F = pfock_mic.num_F;
            double **F1 = pfock_mic.F1;
            double **F2 = pfock_mic.F2;
            double **F4 = pfock_mic.F4;
            double **F5 = pfock_mic.F5;
            double **F6 = pfock_mic.F6;
            double *F3 = pfock_mic.F3;
            int sizeD1 = pfock_mic.sizeD1;
            int sizeD2 = pfock_mic.sizeD2;
            int sizeD4 = pfock_mic.sizeD4;
            int sizeD5 = pfock_mic.sizeD5;
            int sizeD6 = pfock_mic.sizeD6;
            int ldX3 = pfock_mic.ldX3;
            int ldX4 = pfock_mic.ldX4;
            int ldX5 = pfock_mic.ldX5;
            int ldX6 = pfock_mic.ldX6;
            double *_F4 = F4[0];
            double *_F5 = F5[0];
            double *_F6 = F6[0];
            int iMP = iX3M * ldX3 + iX3P;
            int iMQ = iX3M * ldX3;
            int iNP = iX3P;
            int maxrowsize = pfock_mic.ldX1;
            int maxcolsize = pfock_mic.ldX2;            
            #pragma omp parallel
            {
                // reduction on MIC
                for (int i = 1; i < num_F; i++)
                {
                    #pragma omp for
                    for (int j = 0; j < sizeD1; j++)
                    {
                        F1[0][j] += F1[i][j];
                    }
                    #pragma omp for
                    for (int j = 0; j < sizeD2; j++)
                    {
                        F2[0][j] += F2[i][j];
                    }
                    #pragma omp for
                    for (int j = 0; j < sizeD4; j++)
                    {
                        F4[0][j] += F4[i][j];
                    } 
                    #pragma omp for
                    for (int j = 0; j < sizeD5; j++)
                    {
                        F5[0][j] += F5[i][j];
                    } 
                    #pragma omp for
                    for (int j = 0; j < sizeD6; j++)
                    {
                        F6[0][j] += F6[i][j];
                    } 
                }
                #pragma omp for
                for (int iM = 0; iM < nfuncs_row; iM++)
                {
                    for (int iP = 0; iP < nfuncs_col; iP++)
                    {
                        F3[iMP + iM * ldX3 + iP] += _F4[iM * ldX4 + iP];
                    }
                    for (int iQ = 0; iQ < maxcolsize; iQ++)
                    {
                        F3[iMQ + iM * ldX3 + iQ] += _F5[iM * ldX5 + iQ];    
                    }
                }
                #pragma omp for
                for (int iN = 0; iN < maxrowsize; iN++)
                {
                    for (int iP = 0; iP < nfuncs_col; iP++)
                    {
                        F3[iNP + iN * ldX3 + iP] += _F6[iN * ldX6 + iP];
                    }
                }
            }           
        }
    }
}


void offload_reduce (int mic_numdevs,
                     double *F1, double *F2, double *F3,
                     double *F1_offload, double *F2_offload, double *F3_offload,
                     int sizeD1, int sizeD2, int sizeD3)
{
    int i, j;

    for(i = 0; i < mic_numdevs; i++)
    {
        #pragma omp parallel
        {
            #pragma omp for
            for (j = 0; j < sizeD1; j++)
            {
                F1[j + 0 * sizeD1] += F1_offload[j + i * sizeD1];
            }
            #pragma omp for
            for (j = 0; j < sizeD2; j++)
            {
                F2[j + 0 * sizeD2] += F2_offload[j + i * sizeD2];
            }
            #pragma omp for 
            for (j = 0; j < sizeD3; j++)
            {
                F3[j + 0 * sizeD3] += F3_offload[j + i * sizeD3];
            }
        }
    }
}


void offload_init (int nshells, int nnz,
                   int *shellptr, int *shellid, 
                   int *shellrid, double *shellvalue,
                   int *f_startind,
                   int *rowpos, int *colpos,
                   int *rowptr, int *colptr,
                   double *D1, double *D2, double *D3,
                   double *VD1, double *VD2, double *VD3,
                   double **_F1_offload,
                   double **_F2_offload,
                   double **_F3_offload,
                   int ldX1, int ldX2, int ldX3,
                   int ldX4, int ldX5, int ldX6,                   
                   int sizeD1, int sizeD2, int sizeD3,
                   int sizeD4, int sizeD5, int sizeD6,
                   int *_mic_numdevs,
                   int *_nthreads_mic)
{
    int mic_numdevs;
    int nthreads_mic;
    double *F1_offload;
    double *F2_offload;
    double *F3_offload;
    int nt_mic;

    mic_numdevs = _Offload_number_of_devices();
    if (mic_numdevs <= 0)
    {
        *_mic_numdevs = 0;
        return;
    }

    nthreads_mic = 0;
    for(int mic_id = 0; mic_id < mic_numdevs; mic_id++)
    {      
        #pragma offload target(mic:mic_id) out(nt_mic)
        {
            nt_mic = omp_get_max_threads ();
        }
        nthreads_mic = nthreads_mic > nt_mic ? nthreads_mic : nt_mic;
    }
    
    const char *nmic_str = getenv ("nMIC_F");
    int nmic_f;
    if (nmic_str == NULL)
    {
        nmic_f = MIN(16, nthreads_mic);
    }
    else
    {
        nmic_f = atoi (nmic_str);
        if (nmic_f <= 0)
        {
            nmic_f = MIN(16, nthreads_mic);
        }
        else if (nmic_f > nthreads_mic)
        {
            nmic_f = nthreads_mic;
        }
    }
    
    for(int mic_id = 0; mic_id < mic_numdevs; mic_id++)
    {
        #pragma offload target(mic:mic_id)\
            in(nthreads_mic, nmic_f)\
            nocopy(pfock_mic)
        {
            pfock_mic.mem_mic = 0.0;
            pfock_mic.nmic_f = nmic_f;
        }
    }
        
    F1_offload = 
        (double *) ALIGN_MALLOC (sizeof (double) * sizeD1 * mic_numdevs, 64);
    F2_offload =
        (double *) ALIGN_MALLOC (sizeof (double) * sizeD2 * mic_numdevs, 64);
    F3_offload =
        (double *) ALIGN_MALLOC (sizeof (double) * sizeD3 * mic_numdevs, 64);
    if (F1_offload == NULL ||
        F2_offload == NULL ||
        F3_offload == NULL)
    {
        *_mic_numdevs = 0;
        return;        
    }

    *_mic_numdevs = mic_numdevs;
    *_nthreads_mic = nthreads_mic;
    *_F1_offload = F1_offload;
    *_F2_offload = F2_offload;
    *_F3_offload = F3_offload;

    for(int mic_id = 0; mic_id < mic_numdevs; mic_id++)
    {
        #pragma offload target(mic: mic_id) \
            nocopy(pfock_mic)\
            in(sizeD1, sizeD2, sizeD3)\
            in(sizeD4, sizeD5, sizeD6, nthreads_mic) \
            in(ldX1, ldX2, ldX3, ldX4, ldX5, ldX6)\
            nocopy(D1: length(sizeD1) ALLOC) \
            nocopy(D2: length(sizeD2) ALLOC) \
            nocopy(D3: length(sizeD3) ALLOC) \
            nocopy(F1_offload: length(sizeD1) ALLOC) \
            nocopy(F2_offload: length(sizeD2) ALLOC) \
            nocopy(F3_offload: length(sizeD3) ALLOC) \
            in(shellptr: length(nshells + 1) ALLOC) \
            in(shellid: length(nnz) ALLOC)  \
            in(shellrid: length(nnz) ALLOC) \
            in(shellvalue: length(nnz) ALLOC) \
            in(f_startind: length(nshells + 1) ALLOC) \
            in(rowpos: length(nshells) ALLOC) \
            in(colpos: length(nshells) ALLOC) \
            in(rowptr: length(nnz) ALLOC) \
            in(colptr: length(nnz) ALLOC)
        {
            int num_F;
            pfock_mic.sizeD1 = sizeD1;
            pfock_mic.sizeD2 = sizeD2;
            pfock_mic.sizeD3 = sizeD3;
            pfock_mic.sizeD4 = sizeD4;
            pfock_mic.sizeD5 = sizeD5;
            pfock_mic.sizeD6 = sizeD6;            
            pfock_mic.ldX1 = ldX1;
            pfock_mic.ldX2 = ldX2;
            pfock_mic.ldX3 = ldX3;
            pfock_mic.ldX4 = ldX4;
            pfock_mic.ldX5 = ldX5;
            pfock_mic.ldX6 = ldX6;
            pfock_mic.mem_mic += 1.0 * sizeof(double) *
                (sizeD1 + sizeD2 + sizeD3);
            pfock_mic.mem_mic += 1.0 * sizeof(int) *
                (nshells + 1 + 4 * nnz + nshells + 1 + 2 * nshells);
            pfock_mic.mem_mic += 1.0 * sizeof(double) * nnz;
            pfock_mic.nthreads = nthreads_mic;
            num_F = pfock_mic.num_F = 
                (nthreads_mic + pfock_mic.nmic_f - 1)/pfock_mic.nmic_f;            
            pfock_mic.F1 = (double **)malloc (sizeof(double *) * num_F);
            pfock_mic.F2 = (double **)malloc (sizeof(double *) * num_F);
            pfock_mic.F4 = (double **)malloc (sizeof(double *) * num_F);
            pfock_mic.F5 = (double **)malloc (sizeof(double *) * num_F);
            pfock_mic.F6 = (double **)malloc (sizeof(double *) * num_F);
            assert (pfock_mic.F1 != NULL &&
                    pfock_mic.F2 != NULL &&
                    pfock_mic.F4 != NULL &&
                    pfock_mic.F5 != NULL &&
                    pfock_mic.F6 != NULL);
            pfock_mic.mem_mic += 1.0 * num_F *
                sizeof(double) * (sizeD1 + sizeD2 + sizeD4 + sizeD5 + sizeD6) + 
                1.0 * sizeof(double) * sizeD3; 
            pfock_mic.F1[0] = F1_offload;
            pfock_mic.F2[0] = F2_offload;
            pfock_mic.F3 = F3_offload;

        #ifdef __OFFLOAD_DEBUG
            #pragma omp parallel
            {
                stack_t sigstack;
                sigstack.ss_sp = malloc(SIGSTKSZ);
                sigstack.ss_size = SIGSTKSZ;
                sigstack.ss_flags = 0;
                assert(sigaltstack(&sigstack, NULL) == 0);
        
                struct sigaction action;
                memset(&action, 0, sizeof(action));
                sigemptyset(&action.sa_mask);
                action.sa_flags = SA_SIGINFO | SA_ONSTACK;
                action.sa_sigaction = sigsegv_handler;
                assert(sigaction(SIGSEGV, &action, NULL) == 0);
            }
        #endif
            #pragma omp parallel for
            for (int i = 0; i < num_F; i++) 
            {
                if (i > 0)
                {
                    pfock_mic.F1[i] = (double *)ALIGN_MALLOC (sizeof(double) * sizeD1, 64);
                    pfock_mic.F2[i] = (double *)ALIGN_MALLOC (sizeof(double) * sizeD2, 64);
                }
                pfock_mic.F4[i] = (double *)ALIGN_MALLOC (sizeof(double) * sizeD4, 64);
                pfock_mic.F5[i] = (double *)ALIGN_MALLOC (sizeof(double) * sizeD5, 64);
                pfock_mic.F6[i] = (double *)ALIGN_MALLOC (sizeof(double) * sizeD6, 64);
                assert (pfock_mic.F1[i] != NULL &&
                        pfock_mic.F2[i] != NULL &&
                        pfock_mic.F4[i] != NULL &&
                        pfock_mic.F5[i] != NULL &&
                        pfock_mic.F6[i] != NULL);
            }           
        }
    }
}


void offload_deinit (int mic_numdevs,
                     int *shellptr, int *shellid, 
                     int *shellrid, double *shellvalue,
                     int *f_startind,
                     int *rowpos, int *colpos,
                     int *rowptr, int *colptr,
                     double *D1, double *D2, double *D3,
                     double *VD1, double *VD2, double *VD3,
                     double *F1_offload,
                     double *F2_offload,
                     double *F3_offload)                     
{
    for(int mic_id = 0; mic_id < mic_numdevs; mic_id++)
    {
        #pragma offload target(mic: mic_id) \
            nocopy(pfock_mic)\
            nocopy(D1: length(0) FREE) \
            nocopy(D2: length(0) FREE) \
            nocopy(D3: length(0) FREE) \
            nocopy(F1_offload: length(0) FREE) \
            nocopy(F2_offload: length(0) FREE) \
            nocopy(F3_offload: length(0) FREE) \
            nocopy(shellptr: length(0) FREE) \
            nocopy(shellid: length(0) FREE)  \
            nocopy(shellrid: length(0) FREE) \
            nocopy(shellvalue: length(0) FREE) \
            nocopy(f_startind: length(0) FREE) \
            nocopy(rowpos: length(0) FREE) \
            nocopy(colpos: length(0) FREE) \
            nocopy(rowptr: length(0) FREE) \
            nocopy(colptr: length(0) FREE)
        {
            int num_F;
            num_F = pfock_mic.num_F;
            for (int i = 0; i < num_F; i++) 
            {
                if (i >= 1)
                {
                    ALIGN_FREE (pfock_mic.F1[i]);
                    ALIGN_FREE (pfock_mic.F2[i]);
                }
                ALIGN_FREE (pfock_mic.F4[i]);
                ALIGN_FREE (pfock_mic.F5[i]);
                ALIGN_FREE (pfock_mic.F6[i]);
            }
            free (pfock_mic.F1);
            free (pfock_mic.F2);
            free (pfock_mic.F4);           
            free (pfock_mic.F5);
            free (pfock_mic.F6);            
        }
    }
    
    ALIGN_FREE (F1_offload);
    ALIGN_FREE (F2_offload);
    ALIGN_FREE (F3_offload);
}


void offload_copy_D (int mic_numdevs, double *D, int sizeD)
{
    for (int mic_id = 0; mic_id < mic_numdevs; mic_id++)
    {
        #pragma offload_transfer target(mic:mic_id) \
                in(D: length(sizeD) REUSE)
    }
}


void offload_copy_D2 (int mic_numdevs, double *D1, double *D2, double *D3,
                      int sizeD1, int sizeD2, int sizeD3)
{
    for (int mic_id = 0; mic_id < mic_numdevs; mic_id++)
    {
        #pragma offload_transfer target(mic:mic_id) \
                in(D1: length(sizeD1) REUSE)\
                in(D2: length(sizeD2) REUSE)\
                in(D3: length(sizeD3) REUSE)\
                signal(mic_id)
    }
    for(int mic_id = 0; mic_id < mic_numdevs; mic_id++)
    {
        #pragma offload_wait target(mic:mic_id) wait(mic_id)
    }    
}


void offload_wait_mic (int mic_numdevs)
{
    for(int mic_id = 0; mic_id < mic_numdevs; mic_id++)
    {
        #pragma offload_wait target(mic:mic_id) wait(mic_id)
    }
}


void offfload_get_memsize (double *mem_mic)
{
    double _mem_mic;
    
    #pragma offload target(mic:0) \
        out(_mem_mic) nocopy(pfock_mic)
    {
        _mem_mic = pfock_mic.mem_mic;
    }

    *mem_mic = _mem_mic;
}


#endif /* __INTEL_OFFLOAD */
