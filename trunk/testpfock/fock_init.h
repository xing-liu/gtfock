#ifndef __FOCK_INIT_H__
#define __FOCK_INIT_H__


#include "CInt.h"


#define TOLSRC 1e-10


void init_buffers (int myrank, int nshells, int nnz, int nprow, int npcol,
                   int *shellptr, int *shellid, int *f_startind,
                   int *rowptr_f, int *colptr_f,
                   int *rowptr_sh, int *colptr_sh,
                   int *rowptr_blk, int *colptr_blk,
                   int **_rowpos, int **_colpos,
                   int **_rowptr, int **_colptr,
                   int *_maxrowsize, int *_maxcolsize,
                   int *_maxrowfuncs, int *_maxcolfuncs);

void init_grid (int nshells, int nnz, int *shellptr, int *f_startind,
                int nprow, int npcol, int nbp_p,
                int **_rowptr_f, int **_colptr_f,
                int **_rowptr_sh, int **_colptr_sh,
                int **_rowptr_blk, int **_colptr_blk,
                int **_blkrowptr_sh, int **_blkcolptr_sh);

void screening (BasisSet_t basis, int **shellptrOut,
                int **shellidOut, int **shellridOut,
                double **shellvalueOut, int *nnzOut);


#endif /* __FOCK_INIT_H__ */
