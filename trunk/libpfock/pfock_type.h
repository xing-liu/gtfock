#ifndef __PFOCK_TYPE_H__
#define __PFOCK_TYPE_H__


#include <omp.h>
#include <ga.h>

#include "CInt.h"


struct PFock
{
    int nthreads;
    int maxnumdmat;
    int numdmat;
    
    // screening
    int nnz;
    double *shellvalue;
    int *shellptr;
    int *shellid;
    int *shellrid;
    double maxvalue;
    double tolscr;
    double tolscr2;

    // problem parameters
    int nbf;
    int nshells;
    int natoms;
    int maxnfuncs;

    int nprocs;
    int nprow; // np per row    
    int npcol; // np per col
    // task size
    int nbp_p;
    int nbp_row;
    int nbp_col;
    // pointers to shells
    int *blkrowptr_sh;
    int *blkcolptr_sh;
    // local number of taskes
    int ntasks;
    // pointers to blks, shells and functions
    int *rowptr_blk;
    int *colptr_blk;
    int *rowptr_sh;
    int *colptr_sh;
    int *rowptr_f;
    int *colptr_f;
    // blks
    int sblk_row;
    int eblk_row;
    int sblk_col;
    int eblk_col;
    int nblks_row;
    int nblks_col;
    // shells
    int sshell_row;
    int sshell_col;
    int eshell_row;
    int eshell_col;
    int nshells_row;
    int nshells_col;
    // functions
    int sfunc_row;
    int sfunc_col;
    int efunc_row;
    int efunc_col;
    int nfuncs_row;
    int nfuncs_col;
    int sizemyrow;
    int sizemycol;
    
    //task queue
    int ga_taskid;
    int icount;

    // integrals
    ERD_t erd;
    omp_lock_t lock;
    int *f_startind;
    int *s_startind;

    // arrays and buffers
    int ga_screening;
    int committed;
    int *rowptr;
    int *colptr;
    int *rowpos;
    int *colpos;
    int *rowsize;
    int *colsize;
    int *loadrow;
    int sizeloadrow;
    int *loadcol;
    int sizeloadcol;

    // global arrays
    int ga_F;
    int ga_D;
    int ga_K;
    int gatable[4];

    double *FT_block;

    // buf D and F
    int maxrowfuncs;
    int maxcolfuncs;
    int maxrowsize;
    int maxcolsize;
    int sizeX1;
    int sizeX2;
    int sizeX3;
    int sizeX4;
    int sizeX5;
    int sizeX6;
    int ldX1;
    int ldX2;
    int ldX3;
    int ldX4;
    int ldX5;
    int ldX6;
    double *D1;
    double *D2;
    double *D3;
    double *F1;
    double *F2;
    double *F3;
    double *F4;
    double *F5;
    double *F6;
    int numF;
    int ncpu_f;
    int ga_D1;
    int ga_D2;
    int ga_D3;
    int ga_F1;
    int ga_F2;
    int ga_F3;
    int tosteal;
    
    // statistics
    double mem_cpu;
    double mem_mic;
    double *mpi_timepass;
    double timepass;
    double *mpi_timereduce;
    double timereduce;
    double *mpi_timeinit;
    double timeinit;
    double *mpi_timegather;
    double timegather;
    double *mpi_timescatter;
    double timescatter;
    double *mpi_timecomp;
    double timecomp;
    double *mpi_usq;
    double usq;
    double *mpi_uitl;
    double uitl;
    double *mpi_steals;
    double steals;
    double *mpi_stealfrom;
    double stealfrom;
    double *mpi_ngacalls;
    double ngacalls;
    double *mpi_volumega;
    double volumega;

#ifdef __INTEL_OFFLOAD
    // offload
    int offload;
    double mic_fraction;
    int mic_numdevs;
    int nthreads_mic;
    double *F1_offload;
    double *F2_offload;
    double *F3_offload;
#endif
};


struct Ovl
{
    int ga;
};


struct CoreH
{
    int ga;
};


#endif /* #ifndef __PFOCK_TYPE_H__ */
