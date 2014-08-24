#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <assert.h>

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "CInt.h"
#include "fock_init.h"


void screening (BasisSet_t basis, int **shellptrOut,
                int **shellidOut, int **shellridOut,
                double **shellvalueOut, int *nnzOut)
{
    int nthreads;

    nthreads = omp_get_max_threads ();
    ERD_t erd;
    CInt_createERD (basis, &erd, nthreads);
    const int nshells = CInt_getNumShells (basis);

    double *vpairs = (double *) malloc (sizeof (double) * nshells * nshells);
    assert (vpairs != NULL);

    double allmax = 0.0;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num ();
        #pragma omp for reduction(max:allmax)
        for (int M = 0; M < nshells; M++)
        {
            const int dimM = CInt_getShellDim (basis, M);
            for (int N = 0; N < nshells; N++)
            {
                const int dimN = CInt_getShellDim (basis, N);
                int nints;
                double *integrals;

                CInt_computeShellQuartet (basis, erd, tid, M, N, M, N, &integrals,
                                          &nints);
                double mvalue = 0.0;
                if (nints != 0)
                {
                    for (int iM = 0; iM < dimM; iM++)
                    {
                        for (int iN = 0; iN < dimN; iN++)
                        {
                            const int index =
                                iM * (dimN * dimM * dimN + dimN) +
                                iN * (dimM * dimN + 1);
                            if (mvalue < fabs (integrals[index]))
                            {
                                mvalue = fabs (integrals[index]);
                            }
                        }
                    }
                }
                vpairs[M * nshells + N] = mvalue;
                if (mvalue > allmax)
                {
                    allmax = mvalue;
                }
            }
        }
    }

    // init shellptr
    int nnz = 0;
    const double eta = TOLSRC * TOLSRC / allmax;
    int *shellptr = (int *) _mm_malloc (sizeof (int) * (nshells + 1), 64);
    assert (shellptr != NULL);
    memset (shellptr, 0, sizeof (int) * (nshells + 1));
    for (int M = 0; M < nshells; M++)
    {
        for (int N = 0; N < nshells; N++)
        {
            double mvalue = vpairs[M * nshells + N];
            if (mvalue > eta)
            {
                if (M > N && (M + N) % 2 == 1 || M < N && (M + N) % 2 == 0)
                {
                    continue;
                }
                else
                {
                    nnz++;
                }
            }
        }
        shellptr[M + 1] = nnz;
    }

    double *shellvalue = (double *) _mm_malloc (sizeof (double) * nnz, 64);
    int *shellid = (int *) _mm_malloc (sizeof (int) * nnz, 64);
    int *shellrid = (int *) _mm_malloc (sizeof (int) * nnz, 64);
    assert ((shellvalue != NULL) && (shellid != NULL) && (shellrid != NULL));
    nnz = 0;
    for (int M = 0; M < nshells; M++)
    {
        for (int N = 0; N < nshells; N++)
        {
            const double mvalue = vpairs[M * nshells + N];
            if (mvalue > eta)
            {
                if (M > N && (M + N) % 2 == 1 || M < N && (M + N) % 2 == 0)
                {
                    continue;
                }
                if (M == N)
                {
                    shellvalue[nnz] = mvalue;                       
                }
                else
                {
                    shellvalue[nnz] = -mvalue;
                }
                shellid[nnz] = N;
                shellrid[nnz] = M;
                nnz++;
            }
        }
    }
    *nnzOut = nnz;
    free (vpairs);
    CInt_destroyERD (erd);

    *shellidOut = shellid;
    *shellridOut = shellrid;
    *shellptrOut = shellptr;
    *shellvalueOut = shellvalue;
}


static void init_FD_ptr (int nshells, int *shellptr, int *shellid,
                         int *f_startind, int startM, int endM,
                         int *ptrrow, int *rowsize)
{
    int A;
    int B;
    int i;
    int start;
    int end;

    for (A = 0; A < nshells; A++)
    {
        ptrrow[A] = -1;
    }    
    // init row pointers
    for (A = startM; A <= endM; A++)
    {
        start = shellptr[A];
        end = shellptr[A + 1]; 
        for (i = start; i < end; i++)
        {
            B = shellid[i];
            ptrrow[B] = 1;
        }
    }
    
    *rowsize = 0;
    for (A = 0; A < nshells; A++)
    {
        if (ptrrow[A] == 1)
        {
            ptrrow[A] = *rowsize;           
            *rowsize += f_startind[A + 1] - f_startind[A];
        }
    }
}


void init_buffers (int myrank, int nshells, int nnz, int nprow, int npcol,
                   int *shellptr, int *shellid, int *f_startind,
                   int *rowptr_f, int *colptr_f,
                   int *rowptr_sh, int *colptr_sh,
                   int *rowptr_blk, int *colptr_blk,
                   int **_rowpos, int **_colpos,
                   int **_rowptr, int **_colptr,
                   int *_maxrowsize, int *_maxcolsize,
                   int *_maxrowfuncs, int *_maxcolfuncs)
{
    int i;
    int j;
    int k;
    int maxcolsize;
    int maxrowsize;
    int maxrowfuncs;
    int maxcolfuncs;
    int *ptrrow;
    int *ptrcol;
    int sh;
    int count;
    int nfuncs;
    int rowsize;
    int colsize;
               
    ptrrow = (int *)malloc (sizeof(int) * nshells);
    ptrcol = (int *)malloc (sizeof(int) * nshells);
    assert (NULL != ptrrow && NULL != ptrcol);
    
    // compute rowptr/pos and colptr/pos
    int *rowpos = (int *)_mm_malloc (sizeof(int) * nshells, 64);
    int *colpos = (int *)_mm_malloc (sizeof(int) * nshells, 64);
    int *rowptr = (int *)_mm_malloc (sizeof(int) * nnz, 64);
    int *colptr = (int *)_mm_malloc (sizeof(int) * nnz, 64);    
    assert (NULL != rowpos &&
            NULL != colpos  &&
            NULL != rowptr  && 
            NULL != colptr);
    
    // for rows
    count = 0;
    maxrowsize = 0;
    maxrowfuncs = 0;
    for (i = 0; i < nprow; i++)
    {
        init_FD_ptr (nshells, shellptr, shellid, f_startind,
                     rowptr_sh[i], rowptr_sh[i+1] - 1,
                     ptrrow, &rowsize);
        maxrowsize = rowsize > maxrowsize ? rowsize : maxrowsize;
        nfuncs = rowptr_f[i + 1] - rowptr_f[i];
        maxrowfuncs = nfuncs > maxrowfuncs ? nfuncs : maxrowfuncs;
        for (j = rowptr_sh[i]; j < rowptr_sh[i+1]; j++)
        {
            rowpos[j] = ptrrow[j];
            for (k = shellptr[j]; k < shellptr[j+1]; k++)
            {
                sh = shellid[k];
                rowptr[count] = ptrrow[sh];
                count++;
            }
        }
    }

    // for cols
    count = 0;
    maxcolsize = 0;
    maxcolfuncs = 0;
    for (i = 0; i < npcol; i++)
    {
        init_FD_ptr (nshells, shellptr, shellid, f_startind,
                        colptr_sh[i], colptr_sh[i+1] - 1,
                        ptrcol, &colsize);
        maxcolsize = colsize > maxcolsize ? colsize : maxcolsize;
        nfuncs = colptr_f[i + 1] - colptr_f[i];
        maxcolfuncs = nfuncs > maxcolfuncs ? nfuncs : maxcolfuncs;
        for (j = colptr_sh[i]; j < colptr_sh[i+1]; j++)
        {
            colpos[j] = ptrcol[j];
            for (k = shellptr[j]; k < shellptr[j+1]; k++)
            {
                sh = shellid[k];
                colptr[count] = ptrcol[sh];
                count++;
            }
        }
    }
    free (ptrrow);
    free (ptrcol);
    
    *_maxrowsize = maxrowsize;
    *_maxcolsize = maxcolsize;
    *_maxrowfuncs = maxrowfuncs;
    *_maxcolfuncs = maxcolfuncs;
    printf ("FD: (%d %d %d %d)\n",
        maxrowsize, maxcolsize, maxrowfuncs, maxcolfuncs);
    
    *_rowpos = rowpos;
    *_colpos = colpos;
    *_rowptr = rowptr;
    *_colptr = colptr;    
}


static void _recursive_bisection (int *rowptr, int first, int last,
                                  int npartitions, int *partition_ptr)
{
	int offset = rowptr[first];
	int nnz = rowptr[last] - rowptr[first];

	if(npartitions == 1)
	{
		partition_ptr[0] = first;
		return;
	}

	int left = npartitions/2;
	double ideal = ((double)nnz * (double)left)/npartitions;
	int i;
	for(i = first; i < last; i++)
	{
		double count = rowptr[i] - offset;
		double next_count = rowptr[i + 1] - offset;
		if(next_count > ideal)
		{
			if(next_count - ideal > ideal - count)
			{
				_recursive_bisection(rowptr, first, i, left, partition_ptr);
				_recursive_bisection(rowptr, i, last,
                                     npartitions - left, partition_ptr + left);
				return;
			}
			else
			{
				_recursive_bisection(rowptr, first, i + 1, left, partition_ptr);
				_recursive_bisection(rowptr, i + 1, last,
                                     npartitions - left, partition_ptr + left);
				return;
			}
		}
	}
}


static int _nnz_partition (int m, int nnz, int min_nrows,
                          int *rowptr, int npartitions, int *partition_ptr)
{
	_recursive_bisection(rowptr, 0, m, npartitions, partition_ptr);
	partition_ptr[npartitions] = m;

	for (int i = 0; i < npartitions; i++)
	{
		int nrows = partition_ptr[i + 1] - partition_ptr[i];
		if (nrows < min_nrows)
		{
			return -1;
		}
	}
    
	return 0;
}


void init_grid (int nshells, int nnz, int *shellptr, int *f_startind,
                int nprow, int npcol, int nbp_p,
                int **_rowptr_f, int **_colptr_f,
                int **_rowptr_sh, int **_colptr_sh,
                int **_rowptr_blk, int **_colptr_blk,
                int **_blkrowptr_sh, int **_blkcolptr_sh)
{
    int i;
    int j;
    int n0;
    int n1;
    int t;
    int n2;
    int nbp_row;
    int nbp_col;
    int nshells_p;
           
    nbp_row = nprow * nbp_p;
    nbp_col = npcol *nbp_p;
    
    // partition task blocks
    int *rowptr_f = (int *)malloc (sizeof(int) * (nprow + 1));
    int *colptr_f = (int *)malloc (sizeof(int) * (npcol + 1));
    int *rowptr_sh = (int *)malloc (sizeof(int) * (nprow + 1));
    int *colptr_sh = (int *)malloc (sizeof(int) * (npcol + 1));
    int *rowptr_blk = (int *)malloc (sizeof(int) * (nprow + 1));
    int *colptr_blk = (int *)malloc (sizeof(int) * (npcol + 1));
    int *blkrowptr_sh = (int *)malloc (sizeof(int) * (nbp_row + 1));
    int *blkcolptr_sh = (int *)malloc (sizeof(int) * (nbp_col + 1));    
    assert (NULL != rowptr_f &&
            NULL != colptr_f &&
            NULL != rowptr_sh &&
            NULL != colptr_sh &&
            NULL != rowptr_blk &&
            NULL != colptr_blk &&
            NULL != blkrowptr_sh &&
            NULL != blkcolptr_sh);
    int ret;
    ret = _nnz_partition (nshells, nnz, nbp_p, shellptr, nprow, rowptr_sh);    
    assert (ret == 0);
    ret = _nnz_partition (nshells, nnz, nbp_p, shellptr, npcol, colptr_sh);
    assert (ret == 0);
    
    // for row partition
    for (i = 0; i < nprow; i++)
    {
        rowptr_blk[i] = nbp_p * i;
        rowptr_f[i] = f_startind[rowptr_sh[i]];
    }
    rowptr_blk[i] = nbp_row;
    rowptr_f[i] = f_startind[nshells];
    
    // for col partition
    for (i = 0; i < npcol; i++)
    {
        colptr_blk[i] = nbp_p * i;
        colptr_f[i] = f_startind[colptr_sh[i]];
    }
    colptr_blk[i] = nbp_col;
    colptr_f[i] = f_startind[nshells];

    // row
    for (i = 0; i < nprow; i++)
    {
        nshells_p = rowptr_sh[i + 1] - rowptr_sh[i];
        n0 = nshells_p/nbp_p;
        t = nshells_p%nbp_p;
        n1 = (nshells_p + nbp_p - 1)/nbp_p;    
        n2 = n1 * t;
        for (j = 0; j < nbp_p; j++)
        {
            blkrowptr_sh[i *nbp_p + j] = rowptr_sh[i] +
                (j < t ? n1 * j : n2 + (j - t) * n0);
        }
    }
    blkrowptr_sh[i * nbp_p] = nshells;
    // col
    for (i = 0; i < npcol; i++)
    {
        nshells_p = colptr_sh[i + 1] - colptr_sh[i];
        n0 = nshells_p/nbp_p;
        t = nshells_p%nbp_p;
        n1 = (nshells_p + nbp_p - 1)/nbp_p;    
        n2 = n1 * t;
        for (j = 0; j < nbp_p; j++)
        {
            blkcolptr_sh[i *nbp_p + j] = colptr_sh[i] +
                (j < t ? n1 * j : n2 + (j - t) * n0);
        }
    }
    blkcolptr_sh[i * nbp_p] = nshells;

    *_rowptr_f = rowptr_f;
    *_colptr_f = colptr_f;
    *_rowptr_sh = rowptr_sh;
    *_colptr_sh = colptr_sh;
    *_rowptr_blk = rowptr_blk;
    *_colptr_blk = colptr_blk;
    *_blkrowptr_sh = blkrowptr_sh;
    *_blkcolptr_sh = blkcolptr_sh;
}