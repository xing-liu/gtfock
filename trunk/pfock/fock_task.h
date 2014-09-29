#ifndef __FOCK_TASK_H__
#define __FOCK_TASK_H__


#include "CInt.h"


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
                double *nitl, double *nsq);

void reset_F (int nthreads, double *F1, double *F2, double *F3,
              double *F4, double *F5, double *F6,
              int sizeX1, int sizeX2, int sizeX3,
              int sizeX4, int sizeX5, int sizeX6);

void reduce_F (int nthreads, double *F1, double *F2, double *F3,
               double *F4, double *F5, double *F6,
               int sizeX1, int sizeX2, int sizeX3,
               int sizeX4, int sizeX5, int sizeX6,
               int maxrowsize, int maxcolsize,
               int maxrowfuncs, int maxcolfuncs,
               int iX3M, int iX3P,
               int ldX3, int ldX4, int ldX5, int ldX6);


#ifdef __INTEL_OFFLOAD

typedef struct _pfock_mic_t
{
    double **F1;
    double **F2;
    double *F3;
    double **F4;
    double **F5;
    double **F6;
    int sizeD1;
    int sizeD2;
    int sizeD3;
    int sizeD4;
    int sizeD5;
    int sizeD6;
    int num_F;
    int nthreads;
    int ldX1;
    int ldX2;
    int ldX3;
    int ldX4;
    int ldX5;
    int ldX6;
    double mem_mic;
    int nmic_f;
} pfock_mic_t;


#pragma offload_attribute(push, target(mic))

extern pfock_mic_t pfock_mic;

void update_F (double *integrals, int dimM, int dimN,
               int dimP, int dimQ,
               int flag1, int flag2, int flag3,
               int iMN, int iPQ, int iMP, int iNP, int iMQ, int iNQ,
               int iMP0, int iMQ0, int iNP0,
               double *D1, double *D2, double *D3,
               double *F_MN, double *F_PQ, double *F_NQ,
               double *F_MP, double *F_MQ, double *F_NP,
               int ldMN, int ldPQ, int ldNQ,
               int ldMP, int ldMQ, int ldNP);

#pragma offload_attribute(pop)


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
                        double mic_fraction);

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
                   int *_nthreads_mic);
                   
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
                     double *F3_offload);

void offload_reset_F (int mic_numdevs);

void offload_reduce_mic (int mic_numdevs,
                         int nfuncs_row, int nfuncs_col,
                         int iX3M, int iX3P,
                         double *F1_offload,
                         double *F2_offload,
                         double *F3_offload,
                         int sizeD1, int sizeD2, int sizeD3);


void offload_reduce (int mic_numdevs,
                     double *F1, double *F2, double *F3,
                     double *F1_offload, double *F2_offload, double *F3_offload,
                     int sizeD1, int sizeD2, int sizeD3);

void offload_copy_D (int mic_numdevs, double *D, int sizeD);

void offload_copy_D2 (int mic_numdevs, double *D1, double *D2, double *D3,
                      int sizeD1, int sizeD2, int sizeD3);

void offload_wait_mic (int mic_numdevs);

void offfload_get_memsize (double *mem_mic);

#endif /* __FOCK_OFFLOAD_H__ */

#endif /* #define __FOCK_TASK_H__ */
