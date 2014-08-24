#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <ga.h>
#include <macdecls.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include <mkl.h>
#include <assert.h>

#include "pfock_def.h"
#include "config.h"
#include "fock_task.h"
#include "fock_buf.h"
#include "taskq.h"
#include "screening.h"
#include "one_electron.h"


static PFockStatus_t init_fock (PFock_t pfock)
{
    int nshells;
    int nprow;
    int npcol;
    int i;
    int j;
    int n0;
    int n1;
    int t;
    int n2;
    int myrank;
    int nbp_row;
    int nbp_col;
    int nbp_p;
    int nshells_p;
        
    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);

    nbp_p = pfock->nbp_p;
    nbp_row = pfock->nprow * nbp_p;
    nbp_col = pfock->npcol *nbp_p;
    nshells = pfock->nshells;
    // partition task blocks
    nprow = pfock->nprow;
    npcol = pfock->npcol;
    pfock->rowptr_f = (int *)ALIGN_MALLOC (sizeof(int) * (nprow + 1), 64);
    pfock->colptr_f = (int *)ALIGN_MALLOC (sizeof(int) * (npcol + 1), 64);
    pfock->rowptr_sh = (int *)ALIGN_MALLOC (sizeof(int) * (nprow + 1), 64);
    pfock->colptr_sh = (int *)ALIGN_MALLOC (sizeof(int) * (npcol + 1), 64);
    pfock->rowptr_blk = (int *)ALIGN_MALLOC (sizeof(int) * (nprow + 1), 64);
    pfock->colptr_blk = (int *)ALIGN_MALLOC (sizeof(int) * (npcol + 1), 64);
    if (NULL == pfock->rowptr_f || NULL == pfock->colptr_f ||
        NULL == pfock->rowptr_sh || NULL == pfock->colptr_sh ||
        NULL == pfock->rowptr_blk || NULL == pfock->colptr_blk)
    {
        PFOCK_PRINTF (1, "memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }
    pfock->mem_cpu += 3.0 * sizeof(int) * ((nprow + 1) + (npcol + 1));
    // for row partition
    n0 = nshells/nprow;
    t = nshells%nprow;
    n1 = (nshells + nprow - 1)/nprow;    
    n2 = n1 * t;
    for (i = 0; i < nprow; i++)
    {
        pfock->rowptr_blk[i] = nbp_p * i;
        pfock->rowptr_sh[i] = i < t ? n1 * i : n2 + (i - t) * n0;
        pfock->rowptr_f[i] = pfock->f_startind[pfock->rowptr_sh[i]];
    }
    pfock->rowptr_blk[i] = nbp_row;
    pfock->rowptr_sh[i] = nshells;
    pfock->rowptr_f[i] = pfock->nbf;
    // set own
    pfock->sblk_row = pfock->rowptr_blk[myrank/npcol];
    pfock->eblk_row = pfock->rowptr_blk[myrank/npcol + 1] - 1;
    pfock->nblks_row = pfock->eblk_row - pfock->sblk_row + 1;    
    pfock->sshell_row = pfock->rowptr_sh[myrank/npcol];
    pfock->eshell_row = pfock->rowptr_sh[myrank/npcol + 1] - 1;
    pfock->nshells_row = pfock->eshell_row - pfock->sshell_row + 1;    
    pfock->sfunc_row = pfock->rowptr_f[myrank/npcol];
    pfock->efunc_row = pfock->rowptr_f[myrank/npcol + 1] - 1;
    pfock->nfuncs_row = pfock->efunc_row - pfock->sfunc_row + 1;   
    // for col partition
    n0 = nshells/npcol;
    t = nshells%npcol;
    n1 = (nshells + npcol - 1)/npcol;    
    n2 = n1 * t;
    for (i = 0; i < npcol; i++)
    {
        pfock->colptr_blk[i] = nbp_p * i;
        pfock->colptr_sh[i] = i < t ? n1 * i : n2 + (i - t) * n0;
        pfock->colptr_f[i] = pfock->f_startind[pfock->colptr_sh[i]];
    }
    pfock->colptr_blk[i] = nbp_col;
    pfock->colptr_sh[i] = nshells;
    pfock->colptr_f[i] = pfock->nbf;    
    // set own
    pfock->sblk_col = pfock->colptr_blk[myrank%npcol];
    pfock->eblk_col = pfock->colptr_blk[myrank%npcol + 1] - 1;
    pfock->nblks_col = pfock->eblk_col - pfock->sblk_col + 1;    
    pfock->sshell_col = pfock->colptr_sh[myrank%npcol];
    pfock->eshell_col = pfock->colptr_sh[myrank%npcol + 1] - 1;
    pfock->nshells_col = pfock->eshell_col - pfock->sshell_col + 1;    
    pfock->sfunc_col = pfock->colptr_f[myrank%npcol];
    pfock->efunc_col = pfock->colptr_f[myrank%npcol + 1] - 1;
    pfock->nfuncs_col = pfock->efunc_col - pfock->sfunc_col + 1;
     
    pfock->ntasks = nbp_p * nbp_p;
    pfock->blkrowptr_sh = (int *)ALIGN_MALLOC (sizeof(int) * (nbp_row + 1), 64);
    pfock->blkcolptr_sh = (int *)ALIGN_MALLOC (sizeof(int) * (nbp_col + 1), 64);
    pfock->mem_cpu += sizeof(int) * ((nbp_row + 1) + (nbp_col + 1));
    if (NULL == pfock->blkrowptr_sh || NULL == pfock->blkcolptr_sh)
    {
        PFOCK_PRINTF (1, "memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }

    // tasks 2D partitioning
    // row
    for (i = 0; i < nprow; i++)
    {
        nshells_p = pfock->rowptr_sh[i + 1] - pfock->rowptr_sh[i];
        n0 = nshells_p/nbp_p;
        t = nshells_p%nbp_p;
        n1 = (nshells_p + nbp_p - 1)/nbp_p;    
        n2 = n1 * t;
        for (j = 0; j < nbp_p; j++)
        {
            pfock->blkrowptr_sh[i *nbp_p + j] = pfock->rowptr_sh[i] +
                (j < t ? n1 * j : n2 + (j - t) * n0);
        }
    }
    pfock->blkrowptr_sh[i * nbp_p] = nshells;
    // col
    for (i = 0; i < npcol; i++)
    {
        nshells_p = pfock->colptr_sh[i + 1] - pfock->colptr_sh[i];
        n0 = nshells_p/nbp_p;
        t = nshells_p%nbp_p;
        n1 = (nshells_p + nbp_p - 1)/nbp_p;    
        n2 = n1 * t;
        for (j = 0; j < nbp_p; j++)
        {
            pfock->blkcolptr_sh[i *nbp_p + j] = pfock->colptr_sh[i] +
                (j < t ? n1 * j : n2 + (j - t) * n0);
        }
    }
    pfock->blkcolptr_sh[i * nbp_p] = nshells;
 
    return PFOCK_STATUS_SUCCESS;
}


static void recursive_bisection (int *rowptr, int first, int last,
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
				recursive_bisection(rowptr, first, i, left, partition_ptr);
				recursive_bisection(rowptr, i, last,
                                    npartitions - left, partition_ptr + left);
				return;
			}
			else
			{
				recursive_bisection(rowptr, first, i + 1, left, partition_ptr);
				recursive_bisection(rowptr, i + 1, last,
                                    npartitions - left, partition_ptr + left);
				return;
			}
		}
	}
}


static int nnz_partition (int m, int nnz, int min_nrows,
                          int *rowptr, int npartitions, int *partition_ptr)
{
	recursive_bisection(rowptr, 0, m, npartitions, partition_ptr);
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


static PFockStatus_t repartition_fock (PFock_t pfock)
{
    const int nshells = pfock->nshells;
    const int nnz = pfock->nnz;
    const int nbp_p = pfock->nbp_p;
    const int nprow = pfock->nprow;
    const int npcol = pfock->npcol;
    int *shellptr = pfock->shellptr;
    int myrank;
    int ret;
    
    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);

    // for row partition
    int *newrowptr = (int *)malloc (sizeof(int) * (nprow + 1));
    int *newcolptr = (int *)malloc (sizeof(int) * (npcol + 1));
    ret = nnz_partition (nshells, nnz, nbp_p, shellptr, nprow, newrowptr);    
    if (ret != 0)
    {
        PFOCK_PRINTF (1, "nbp_p is too large\n");
        return PFOCK_STATUS_EXECUTION_FAILED;
    }
    ret = nnz_partition (nshells, nnz, nbp_p, shellptr, npcol, newcolptr);
    if (ret != 0)
    {
        PFOCK_PRINTF (1, "nbp_p is too large\n");
        return PFOCK_STATUS_EXECUTION_FAILED;
    }
    memcpy (pfock->rowptr_sh, newrowptr, sizeof(int) * (nprow + 1));    
    memcpy (pfock->colptr_sh, newcolptr, sizeof(int) * (npcol + 1));
    free (newrowptr);
    free (newcolptr);
    
    for (int i = 0; i < nprow; i++)
    {
        pfock->rowptr_f[i] = pfock->f_startind[pfock->rowptr_sh[i]];
    }
    pfock->rowptr_f[nprow] = pfock->nbf;
    // set own  
    pfock->sshell_row = pfock->rowptr_sh[myrank/npcol];
    pfock->eshell_row = pfock->rowptr_sh[myrank/npcol + 1] - 1;
    pfock->nshells_row = pfock->eshell_row - pfock->sshell_row + 1;    
    pfock->sfunc_row = pfock->rowptr_f[myrank/npcol];
    pfock->efunc_row = pfock->rowptr_f[myrank/npcol + 1] - 1;
    pfock->nfuncs_row = pfock->efunc_row - pfock->sfunc_row + 1;  
    
    // for col partition
    for (int i = 0; i < npcol; i++)
    {
        pfock->colptr_f[i] = pfock->f_startind[pfock->colptr_sh[i]];
    }
    pfock->colptr_f[npcol] = pfock->nbf;    
    // set own   
    pfock->sshell_col = pfock->colptr_sh[myrank%npcol];
    pfock->eshell_col = pfock->colptr_sh[myrank%npcol + 1] - 1;
    pfock->nshells_col = pfock->eshell_col - pfock->sshell_col + 1;    
    pfock->sfunc_col = pfock->colptr_f[myrank%npcol];
    pfock->efunc_col = pfock->colptr_f[myrank%npcol + 1] - 1;
    pfock->nfuncs_col = pfock->efunc_col - pfock->sfunc_col + 1;
     
    // tasks 2D partitioning
    // row
    int nshells_p;
    int n0;
    int t;
    int n1;
    int n2;
    for (int i = 0; i < nprow; i++)
    {
        nshells_p = pfock->rowptr_sh[i + 1] - pfock->rowptr_sh[i];
        n0 = nshells_p/nbp_p;
        t = nshells_p%nbp_p;
        n1 = (nshells_p + nbp_p - 1)/nbp_p;    
        n2 = n1 * t;
        for (int j = 0; j < nbp_p; j++)
        {
            pfock->blkrowptr_sh[i *nbp_p + j] = pfock->rowptr_sh[i] +
                (j < t ? n1 * j : n2 + (j - t) * n0);
        }
    }
    pfock->blkrowptr_sh[nprow * nbp_p] = nshells;
    // col
    for (int i = 0; i < npcol; i++)
    {
        nshells_p = pfock->colptr_sh[i + 1] - pfock->colptr_sh[i];
        n0 = nshells_p/nbp_p;
        t = nshells_p%nbp_p;
        n1 = (nshells_p + nbp_p - 1)/nbp_p;    
        n2 = n1 * t;
        for (int j = 0; j < nbp_p; j++)
        {
            pfock->blkcolptr_sh[i *nbp_p + j] = pfock->colptr_sh[i] +
                (j < t ? n1 * j : n2 + (j - t) * n0);
        }
    }
    pfock->blkcolptr_sh[npcol * nbp_p] = nshells;

    // for correct_F
    pfock->FT_block = (double *)ALIGN_MALLOC (sizeof(double) *
        pfock->nfuncs_row * pfock->nfuncs_col, 64);
    pfock->mem_cpu += 1.0 * pfock->nfuncs_row * pfock->nfuncs_col * sizeof(double);
    if (NULL == pfock->FT_block)
    {
        PFOCK_PRINTF (1, "memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }
    
    return PFOCK_STATUS_SUCCESS;
}


static PFockStatus_t init_GA (PFock_t pfock)
{
    int nbf;
    int nprow;
    int npcol;
    int *map;
    int i;
    int dims[2];
    int block[2];
    char str[8];

    // create global arrays
    nbf = pfock->nbf;
    nprow = pfock->nprow;
    npcol = pfock->npcol;
    map = (int *)ALIGN_MALLOC (sizeof(int) * (nprow + npcol), 64);
    if (NULL == map)
    {
        PFOCK_PRINTF (1, "memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }
    
    for (i = 0; i < nprow; i++)
    {       
        map[i] = pfock->rowptr_f[i];
    }   
    for (i = 0; i < npcol; i++)
    {
        map[i + nprow] = pfock->colptr_f[i];
    } 
    dims[0] = nbf;
    dims[1] = nbf;
    block[0] = nprow;
    block[1] = npcol;
    
    sprintf (str, "D");
    pfock->ga_D = NGA_Create_irreg (C_DBL, 2, dims, str, block, map);
    if (0 == pfock->ga_D)
    {
        PFOCK_PRINTF (1, "GA allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }
    
    sprintf (str, "F");
    pfock->ga_F = GA_Duplicate (pfock->ga_D, str);    
    if (0 == pfock->ga_F)
    {
        PFOCK_PRINTF (1, "GA allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }

    sprintf (str, "K");
    pfock->ga_K = GA_Duplicate (pfock->ga_D, str);    
    if (0 == pfock->ga_K)
    {
        PFOCK_PRINTF (1, "GA allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }
    
    pfock->gatable[PFOCK_MAT_TYPE_D] = pfock->ga_D;
    pfock->gatable[PFOCK_MAT_TYPE_F] = pfock->ga_F;
    pfock->gatable[PFOCK_MAT_TYPE_J] = pfock->ga_F;
    pfock->gatable[PFOCK_MAT_TYPE_K] = pfock->ga_K;

    ALIGN_FREE (map);
    
    return PFOCK_STATUS_SUCCESS;
}


static void clean_GA (PFock_t pfock)
{
    GA_Destroy (pfock->ga_D);        
    GA_Destroy (pfock->ga_F);
    GA_Destroy (pfock->ga_K);
}


static PFockStatus_t create_FD_GArrays (PFock_t pfock)
{
    int dims[2];
    int block[2];
    char str[8];
    
    int sizeD1 = pfock->sizeX1;
    int sizeD2 = pfock->sizeX2;
    int sizeD3 = pfock->sizeX3;
  
    int *map = (int *)ALIGN_MALLOC (sizeof(int) * (1 + pfock->nprocs), 64);
    if (NULL == map)
    {
        PFOCK_PRINTF (1, "memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }
    block[0] = pfock->nprocs;
    block[1] = 1;

    // for D1
    for (int i = 0; i < pfock->nprocs; i++)
    {
        map[i] = i;
    }
    map[pfock->nprocs] = 0;
    dims[0] = pfock->nprocs;
    dims[1] = sizeD1;
    sprintf (str, "D1");
    pfock->ga_D1 = NGA_Create_irreg (C_DBL, 2, dims, str, block, map);
        
    // for D2
    for (int i = 0; i < pfock->nprocs; i++)
    {
        map[i] = i;
    }
    map[pfock->nprocs] = 0;
    dims[0] = pfock->nprocs;
    dims[1] = sizeD2;
    sprintf (str, "D2");
    pfock->ga_D2 = NGA_Create_irreg (C_DBL, 2, dims, str, block, map);
    
    // for D3
    for (int i = 0; i < pfock->nprocs; i++)
    {
        map[i] = i;
    }
    map[pfock->nprocs] = 0;
    dims[0] = pfock->nprocs;
    dims[1] = sizeD3;
    sprintf (str, "D3");
    pfock->ga_D3 = NGA_Create_irreg (C_DBL, 2, dims, str, block, map);
    
    if (0 == pfock->ga_D1 ||
        0 == pfock->ga_D2 ||
        0 == pfock->ga_D3)
    {
        PFOCK_PRINTF (1, "GA allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }

    sprintf (str, "F1");
    pfock->ga_F1 = GA_Duplicate (pfock->ga_D1, str);
    sprintf (str, "F2");
    pfock->ga_F2 = GA_Duplicate (pfock->ga_D2, str);
    sprintf (str, "F3");
    pfock->ga_F3 = GA_Duplicate (pfock->ga_D3, str);
    if (0 == pfock->ga_F1 ||
        0 == pfock->ga_F2 ||
        0 == pfock->ga_F3)
    {
        PFOCK_PRINTF (1, "GA allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }    

    ALIGN_FREE (map);

    return PFOCK_STATUS_SUCCESS; 
}


static PFockStatus_t create_buffers (PFock_t pfock)
{
    int i;
    int j;
    int k;  
    int myrank;
    int maxcolsize;
    int maxrowsize;
    int maxrowfuncs;
    int maxcolfuncs;
    int *ptrrow;
    int *ptrcol;
    int sh;
    int count;
    int myrow;
    int mycol;
    int nfuncs;
    int nthreads;
            
    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
    myrow = myrank/pfock->npcol;
    mycol = myrank%pfock->npcol;   
    ptrrow = (int *)ALIGN_MALLOC (sizeof(int) * pfock->nshells, 64);
    ptrcol = (int *)ALIGN_MALLOC (sizeof(int) * pfock->nshells, 64);
    if (NULL == ptrrow ||
        NULL == ptrcol)
    {
        PFOCK_PRINTF (1, "memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;    
    }    

    // compute rowptr/pos and colptr/pos
    pfock->rowpos = (int *)ALIGN_MALLOC (sizeof(int) * pfock->nshells, 64);
    pfock->colpos = (int *)ALIGN_MALLOC (sizeof(int) * pfock->nshells, 64);
    pfock->rowptr = (int *)ALIGN_MALLOC (sizeof(int) * pfock->nnz, 64);
    pfock->colptr = (int *)ALIGN_MALLOC (sizeof(int) * pfock->nnz, 64);
    pfock->rowsize = (int *)ALIGN_MALLOC (sizeof(int) * pfock->nprow, 64);
    pfock->colsize = (int *)ALIGN_MALLOC (sizeof(int) * pfock->npcol, 64);
    pfock->mem_cpu += 1.0 * sizeof(int) *
        (2.0 * pfock->nshells + 2.0 * pfock->nnz +
         pfock->nprow + pfock->npcol);
    if (NULL == pfock->rowpos  ||
        NULL == pfock->colpos  ||
        NULL == pfock->rowptr  || 
        NULL == pfock->colptr  ||
        NULL == pfock->rowsize ||
        NULL == pfock->colsize)
    {
        PFOCK_PRINTF (1, "memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;    
    }   
    count = 0;
    maxrowsize = 0;
    maxrowfuncs = 0;
    for (i = 0; i < pfock->nprow; i++)
    {
        compute_FD_ptr (pfock,
                        pfock->rowptr_sh[i], pfock->rowptr_sh[i+1] - 1,
                        ptrrow, &(pfock->rowsize[i]));
        maxrowsize =
            pfock->rowsize[i] > maxrowsize ? pfock->rowsize[i] : maxrowsize;
        nfuncs = pfock->rowptr_f[i + 1] - pfock->rowptr_f[i];
        maxrowfuncs = nfuncs > maxrowfuncs ? nfuncs : maxrowfuncs;
        if (i == myrow)
        {
            pfock->sizemyrow = pfock->rowsize[i];
            init_FD_load (pfock, ptrrow, &(pfock->loadrow), &(pfock->sizeloadrow));  
        }
        for (j = pfock->rowptr_sh[i]; j < pfock->rowptr_sh[i+1]; j++)
        {
            pfock->rowpos[j] = ptrrow[j];
            for (k = pfock->shellptr[j]; k < pfock->shellptr[j+1]; k++)
            {
                sh = pfock->shellid[k];
                pfock->rowptr[count] = ptrrow[sh];
                count++;
            }
        }
    }
    count = 0;
    maxcolsize = 0;
    maxcolfuncs = 0;
    for (i = 0; i < pfock->npcol; i++)
    {
        compute_FD_ptr (pfock,
                        pfock->colptr_sh[i], pfock->colptr_sh[i+1] - 1,
                        ptrcol, &(pfock->colsize[i]));
        maxcolsize =
            pfock->colsize[i] > maxcolsize ? pfock->colsize[i] : maxcolsize;
        nfuncs = pfock->colptr_f[i + 1] - pfock->colptr_f[i];
        maxcolfuncs = nfuncs > maxcolfuncs ? nfuncs : maxcolfuncs;
        if (i == mycol)
        {
            pfock->sizemycol = pfock->colsize[i];
            init_FD_load (pfock, ptrcol, &(pfock->loadcol), &(pfock->sizeloadcol));  
        }
        for (j = pfock->colptr_sh[i]; j < pfock->colptr_sh[i+1]; j++)
        {
            pfock->colpos[j] = ptrcol[j];
            for (k = pfock->shellptr[j]; k < pfock->shellptr[j+1]; k++)
            {
                sh = pfock->shellid[k];
                pfock->colptr[count] = ptrcol[sh];
                count++;
            }
        }
    }
    ALIGN_FREE (ptrrow);
    ALIGN_FREE (ptrcol);
    pfock->maxrowsize = maxrowsize;
    pfock->maxcolsize = maxcolsize;
    pfock->maxrowfuncs = maxrowfuncs;
    pfock->maxcolfuncs = maxcolfuncs;
    int sizeX1 = maxrowfuncs * maxrowsize;
    int sizeX2 = maxcolfuncs * maxcolsize;
    int sizeX3 = maxrowsize * maxcolsize;
    pfock->sizeX1 = sizeX1;
    pfock->sizeX2 = sizeX2;
    pfock->sizeX3 = sizeX3;
    if (myrank == 0)
    {
        printf ("  FD size (%d %d %d %d)\n",
            maxrowfuncs, maxcolfuncs, maxrowsize, maxcolsize);
    }
    
    // D buf
    pfock->D1 = (double *)ALIGN_MALLOC (sizeof(double) * sizeX1, 64);
    pfock->D2 = (double *)ALIGN_MALLOC (sizeof(double) * sizeX2, 64); 
    pfock->D3 = (double *)ALIGN_MALLOC (sizeof(double) * sizeX3, 64);
    pfock->mem_cpu += 1.0 * sizeof(double) * (sizeX1 + sizeX2 + sizeX3);
    if (NULL == pfock->D1 ||
        NULL == pfock->D2 ||
        NULL == pfock->D3)
    {
        PFOCK_PRINTF (1, "memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }
    // F buf
    nthreads = pfock->nthreads;
    const char *ncpu_str = getenv ("nCPU_F");
    int ncpu_f;
    if (ncpu_str == NULL)
    {
        ncpu_f = 1;
    }
    else
    {
        ncpu_f = atoi (ncpu_str);
        if (ncpu_f <= 0 || ncpu_f > nthreads)
        {
            ncpu_f = 1;
        }
    }
    
    int sizeX4 = maxrowfuncs * maxcolfuncs;
    int sizeX6 = maxrowsize * maxcolfuncs;
    int sizeX5 = maxrowfuncs * maxcolsize;
    pfock->sizeX4 = sizeX4;
    pfock->sizeX5 = sizeX5;
    pfock->sizeX6 = sizeX6;
    pfock->ncpu_f = ncpu_f;
    int numF = pfock->numF = (nthreads + ncpu_f - 1)/ncpu_f;
    pfock->F1 = (double *)ALIGN_MALLOC (sizeof(double) * sizeX1 * numF, 64);
    pfock->F2 = (double *)ALIGN_MALLOC (sizeof(double) * sizeX2 * numF, 64); 
    pfock->F3 = (double *)ALIGN_MALLOC (sizeof(double) * sizeX3, 64);
    pfock->F4 = (double *)ALIGN_MALLOC (sizeof(double) * sizeX4 * numF, 64);
    pfock->F5 = (double *)ALIGN_MALLOC (sizeof(double) * sizeX5 * numF, 64); 
    pfock->F6 = (double *)ALIGN_MALLOC (sizeof(double) * sizeX6 * numF, 64);
    pfock->mem_cpu += 1.0 * sizeof(double) *
        (((double)sizeX1 + sizeX2 + sizeX4 + sizeX5 + sizeX6) * numF + sizeX3);
    if (NULL == pfock->F1 ||
        NULL == pfock->F2 ||
        NULL == pfock->F3 ||
        NULL == pfock->F4 ||
        NULL == pfock->F5 ||
        NULL == pfock->F6)
    {
        PFOCK_PRINTF (1, "memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }

    pfock->ldX1 = maxrowsize;
    pfock->ldX2 = maxcolsize;
    pfock->ldX3 = maxcolsize;
    pfock->ldX4 = maxcolfuncs;
    pfock->ldX5 = maxcolsize;
    pfock->ldX6 = maxcolfuncs;        
    return PFOCK_STATUS_SUCCESS;
}


static void destroy_buffers (PFock_t pfock)
{
    GA_Destroy (pfock->ga_D1);
    GA_Destroy (pfock->ga_D2);
    GA_Destroy (pfock->ga_D3);
    GA_Destroy (pfock->ga_F1);
    GA_Destroy (pfock->ga_F2);
    GA_Destroy (pfock->ga_F3);
        
    ALIGN_FREE (pfock->rowpos);
    ALIGN_FREE (pfock->colpos);
    ALIGN_FREE (pfock->rowptr);
    ALIGN_FREE (pfock->colptr);
    ALIGN_FREE (pfock->loadrow);
    ALIGN_FREE (pfock->loadcol);
    ALIGN_FREE (pfock->rowsize);
    ALIGN_FREE (pfock->colsize);
    
    ALIGN_FREE (pfock->D1);
    ALIGN_FREE (pfock->D2);
    ALIGN_FREE (pfock->D3);
    ALIGN_FREE (pfock->F1);
    ALIGN_FREE (pfock->F2);
    ALIGN_FREE (pfock->F3);
    ALIGN_FREE (pfock->F4);
    ALIGN_FREE (pfock->F5);
    ALIGN_FREE (pfock->F6);
}


#ifdef __INTEL_OFFLOAD
PFockStatus_t PFock_create (BasisSet_t basis,
                            int nprow,
                            int npcol,
                            int ntasks,
                            int maxnumdmat,
                            int symm,
                            double tolscr,
                            double offload_fraction,
                            PFock_t *_pfock)
#else
PFockStatus_t PFock_create (BasisSet_t basis,
                            int nprow,
                            int npcol,
                            int ntasks,
                            int maxnumdmat,
                            int symm,
                            double tolscr,
                            PFock_t *_pfock)
#endif
{
    PFock_t pfock;
    int i;
    int nprocs;
    int myrank;
    PFockStatus_t ret;
    int minnshells;
    
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);         
    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);

#ifdef __INTEL_OFFLOAD
    if (maxnumdmat > 1)
    {
        PFOCK_PRINTF (1, "offload mode only supports one density matrix\n");
        return PFOCK_STATUS_INVALID_VALUE;
    }
#endif

    // ALIGN_MALLOC pfock
    pfock = (PFock_t)ALIGN_MALLOC (sizeof(struct PFock), 64);    
    if (NULL == pfock)
    {
        PFOCK_PRINTF (1, "memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }
    memset (pfock, 0, sizeof(PFock_t));

    // initialization   
    pfock->maxnfuncs = CInt_getMaxShellDim (basis);
    pfock->nbf = CInt_getNumFuncs (basis);
    pfock->nshells = CInt_getNumShells (basis);
    pfock->natoms = CInt_getNumAtoms (basis);    
    pfock->maxnumdmat = maxnumdmat;
    pfock->nthreads = omp_get_max_threads ();
    pfock->mem_cpu = 0.0;
    pfock->mem_mic = 0.0;
    omp_set_num_threads (pfock->nthreads);

#ifdef __INTEL_OFFLOAD
    // offload
    if (offload_fraction <= 0.0)
    {
        pfock->offload = 0;
    }
    else
    {
        pfock->mic_fraction = offload_fraction;
        pfock->offload = 1;
    }
#endif

    // set nprow and nocol
    if (nprow <= 0 || nprow > pfock->nshells ||
        npcol <= 0 || npcol > pfock->nshells ||
        (nprow * npcol) > nprocs)
    {
        PFOCK_PRINTF (1, "invalid nprow or npcol\n");
        return PFOCK_STATUS_INVALID_VALUE;
    }
    else
    {
        pfock->nprow= nprow;
        pfock->npcol = npcol;
        pfock->nprocs = nprow * npcol;
    }
    
    // set tasks
    minnshells = (nprow > npcol ? nprow : npcol);
    minnshells = pfock->nshells/minnshells;
    if (ntasks >= minnshells)
    {
        pfock->nbp_p = minnshells;
    }
    else if (ntasks <= 0)
    {
        pfock->nbp_p = 4;
        pfock->nbp_p = MIN (pfock->nbp_p, minnshells);
    }
    else
    {
        pfock->nbp_p = ntasks;
    }
    pfock->nbp_row = pfock->nbp_col = pfock->nbp_p;
    
    // set screening threshold
    if (tolscr < 0.0)
    {
        PFOCK_PRINTF (1, "invalid screening threshold\n");
        return PFOCK_STATUS_INVALID_VALUE;
    }
    else
    {
        pfock->tolscr = tolscr;
        pfock->tolscr2 = tolscr * tolscr;
    }    

    // functions starting positions of shells
    pfock->f_startind = (int *)ALIGN_MALLOC (sizeof(int) * (pfock->nshells + 1), 64);
    pfock->mem_cpu += sizeof(int) * (pfock->nshells + 1);   
    if (NULL == pfock->f_startind)
    {
        PFOCK_PRINTF (1, "memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }  
    for (i = 0; i < pfock->nshells; i++)
    {
        pfock->f_startind[i] = CInt_getFuncStartInd (basis, i);
    }
    pfock->f_startind[pfock->nshells] = pfock->nbf;

    // shells starting positions of atoms
    pfock->s_startind = (int *)ALIGN_MALLOC (sizeof(int) *
        (pfock->natoms + 1), 64);
    pfock->mem_cpu += sizeof(int) * (pfock->natoms + 1); 
    if (NULL == pfock->s_startind)
    {
        PFOCK_PRINTF (1, "memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }
    for (i = 0; i < pfock->natoms; i++)
    {
        pfock->s_startind[i] = CInt_getAtomStartInd (basis, i);
    }
    pfock->s_startind[pfock->natoms] = pfock->nshells;

    // init comm
    if ((ret = init_fock (pfock)) != PFOCK_STATUS_SUCCESS)
    {
        return ret;
    }
    
    // init scheduler
    if (init_taskq (pfock) != 0)
    {
        PFOCK_PRINTF (1, "task queue initialization failed\n");
        return PFOCK_STATUS_INIT_FAILED;
    }

    if (myrank == 0)
    {
        PFOCK_INFO ("screening ...\n");
    }
    double t1 = MPI_Wtime ();
    // schwartz screening
    if (schwartz_screening (pfock, basis) != 0)
    {
        PFOCK_PRINTF (1, "schwartz screening failed\n");
        return PFOCK_STATUS_INIT_FAILED;
    }
    double t2 = MPI_Wtime ();
    if (myrank == 0)
    {
        PFOCK_INFO ("takes %.3lf secs\n", t2 - t1);
    }
#if 1
    // repartition
    if ((ret = repartition_fock (pfock)) != PFOCK_STATUS_SUCCESS)
    {
        return ret;
    }
#endif    
    // init global arrays
    if ((ret = init_GA (pfock)) != PFOCK_STATUS_SUCCESS)
    {
        return ret;
    }

    // create local buffers
    if ((ret = create_buffers (pfock)) != PFOCK_STATUS_SUCCESS)
    {
        return ret;
    }

    if ((ret = create_FD_GArrays (pfock)) != PFOCK_STATUS_SUCCESS)
    {
        return ret;
    }

#ifdef __INTEL_OFFLOAD
    if (pfock->offload == 1)
    {
        double *D1;
        double *D2;
        double *D3;
        int ldD;
        int lo[2];
        int hi[2];
        lo[0] = myrank;
        hi[0] = myrank;
        lo[1] = 0;
        hi[1] = pfock->sizeX1 - 1;
        NGA_Access (pfock->ga_D1, lo, hi, &D1, &ldD);
        hi[1] = pfock->sizeX2 - 1;
        NGA_Access (pfock->ga_D2, lo, hi, &D2, &ldD);
        hi[1] = pfock->sizeX3 - 1;
        NGA_Access (pfock->ga_D3, lo, hi, &D3, &ldD);          
        offload_init (pfock->nshells, pfock->nnz,
                      pfock->shellptr, pfock->shellid,
                      pfock->shellrid, pfock->shellvalue,
                      pfock->f_startind,
                      pfock->rowpos, pfock->colpos,
                      pfock->rowptr, pfock->colptr,
                      D1, D2, D3, pfock->D1, pfock->D2, pfock->D3,
                      &(pfock->F1_offload),
                      &(pfock->F2_offload),
                      &(pfock->F3_offload),
                      pfock->ldX1, pfock->ldX2, pfock->ldX3,
                      pfock->ldX4, pfock->ldX5, pfock->ldX6,
                      pfock->sizeX1, pfock->sizeX2, pfock->sizeX3,
                      pfock->sizeX4, pfock->sizeX5, pfock->sizeX6,
                      &(pfock->mic_numdevs),
                      &(pfock->nthreads_mic));
        pfock->mem_cpu += 1.0 * (pfock->sizeX1 + pfock->sizeX2 + pfock->sizeX3)
            * sizeof(double) * pfock->mic_numdevs;
        offfload_get_memsize (&(pfock->mem_mic));
        
        if (pfock->mic_numdevs == 0)
        {
            PFOCK_PRINTF (1, "fail to initialize offload devices\n");
            return PFOCK_STATUS_ALLOC_FAILED;        
        }
        // create ERD
        CInt_offload_createERD (basis, &(pfock->erd),
                                pfock->nthreads,
                                pfock->nthreads_mic);
        // dummy init F
        offload_reset_F (pfock->mic_numdevs);
        offload_wait_mic (pfock->mic_numdevs);
    }
    else
#endif
    {
        CInt_createERD (basis, &(pfock->erd), pfock->nthreads);
    }

    pfock->mpi_timepass
        = (double *)ALIGN_MALLOC (sizeof(double) * pfock->nprocs, 64);
    pfock->mpi_timereduce
        = (double *)ALIGN_MALLOC (sizeof(double) * pfock->nprocs, 64);
    pfock->mpi_timeinit
        = (double *)ALIGN_MALLOC (sizeof(double) * pfock->nprocs, 64);
    pfock->mpi_timecomp
        = (double *)ALIGN_MALLOC (sizeof(double) * pfock->nprocs, 64);
    pfock->mpi_timegather
        = (double *)ALIGN_MALLOC (sizeof(double) * pfock->nprocs, 64);
    pfock->mpi_timescatter
        = (double *)ALIGN_MALLOC (sizeof(double) * pfock->nprocs, 64);
    pfock->mpi_usq
        = (double *)ALIGN_MALLOC (sizeof(double) * pfock->nprocs, 64);
    pfock->mpi_uitl
        = (double *)ALIGN_MALLOC (sizeof(double) * pfock->nprocs, 64);
    pfock->mpi_steals
        = (double *)ALIGN_MALLOC (sizeof(double) * pfock->nprocs, 64);
    pfock->mpi_stealfrom
        = (double *)ALIGN_MALLOC (sizeof(double) * pfock->nprocs, 64);
    pfock->mpi_ngacalls
        = (double *)ALIGN_MALLOC (sizeof(double) * pfock->nprocs, 64);
    pfock->mpi_volumega
        = (double *)ALIGN_MALLOC (sizeof(double) * pfock->nprocs, 64);
    if (pfock->mpi_timepass == NULL ||
        pfock->mpi_timereduce == NULL ||
        pfock->mpi_timeinit == NULL ||
        pfock->mpi_timecomp == NULL ||        
        pfock->mpi_usq == NULL ||
        pfock->mpi_uitl == NULL ||
        pfock->mpi_steals == NULL ||
        pfock->mpi_stealfrom == NULL ||
        pfock->mpi_ngacalls == NULL ||
        pfock->mpi_volumega == NULL ||
        pfock->mpi_timegather == NULL ||
        pfock->mpi_timescatter == NULL)
    {
        PFOCK_PRINTF (1, "memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }
    
    pfock->committed = 0;
    pfock->tosteal = 1;
    *_pfock = pfock;
    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_destroy (PFock_t pfock)
{
#ifdef __INTEL_OFFLOAD
    if (pfock->offload == 1)
    {
        double *D1;
        double *D2;
        double *D3;
        int ldD;
        int lo[2];
        int hi[2];
        int myrank;
        MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
        lo[0] = myrank;
        hi[0] = myrank;
        lo[1] = 0;
        hi[1] = pfock->sizeX1 - 1;
        NGA_Access (pfock->ga_D1, lo, hi, &D1, &ldD);
        hi[1] = pfock->sizeX2 - 1;
        NGA_Access (pfock->ga_D2, lo, hi, &D2, &ldD);
        hi[1] = pfock->sizeX3 - 1;
        NGA_Access (pfock->ga_D3, lo, hi, &D3, &ldD);
        offload_deinit (pfock->mic_numdevs,
                        pfock->shellptr, pfock->shellid,
                        pfock->shellrid, pfock->shellvalue,
                        pfock->f_startind,
                        pfock->rowpos, pfock->colpos,
                        pfock->rowptr, pfock->colptr,
                        D1, D2, D3,
                        pfock->D1, pfock->D2, pfock->D3,
                        pfock->F1_offload,
                        pfock->F2_offload,
                        pfock->F3_offload);
    }
#endif    
    ALIGN_FREE (pfock->blkrowptr_sh);
    ALIGN_FREE (pfock->blkcolptr_sh);
    ALIGN_FREE (pfock->rowptr_sh);
    ALIGN_FREE (pfock->colptr_sh);
    ALIGN_FREE (pfock->rowptr_f);
    ALIGN_FREE (pfock->colptr_f);
    ALIGN_FREE (pfock->rowptr_blk);
    ALIGN_FREE (pfock->colptr_blk);
    ALIGN_FREE (pfock->FT_block);
    ALIGN_FREE (pfock->f_startind);
    ALIGN_FREE (pfock->s_startind);

    CInt_destroyERD (pfock->erd);
       
    clean_taskq (pfock);
    clean_screening (pfock);
    clean_GA (pfock);
    destroy_buffers (pfock);

    ALIGN_FREE (pfock->mpi_timepass);
    ALIGN_FREE (pfock->mpi_timereduce);
    ALIGN_FREE (pfock->mpi_timeinit);
    ALIGN_FREE (pfock->mpi_timecomp);
    ALIGN_FREE (pfock->mpi_timegather);
    ALIGN_FREE (pfock->mpi_timescatter);
    ALIGN_FREE (pfock->mpi_usq);
    ALIGN_FREE (pfock->mpi_uitl);
    ALIGN_FREE (pfock->mpi_steals);
    ALIGN_FREE (pfock->mpi_stealfrom);
    ALIGN_FREE (pfock->mpi_ngacalls);
    ALIGN_FREE (pfock->mpi_volumega);
    
    ALIGN_FREE (pfock);
    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_putDenMat( PFock_t pfock,
                               int index,
                               int rowstart,
                               int rowend,
                               int colstart,
                               int colend,
                               double *dmat,
                               int stride )
{
    int lo[2];
    int hi[2];

    if (pfock->committed == 1)
    {
        PFOCK_PRINTF (1, "can't change density matrix after PFock_commitDenMats().\n");
        return PFOCK_STATUS_EXECUTION_FAILED;
    }
    
    lo[0] = rowstart;
    hi[0] = rowend;    
    lo[1] = colstart;
    hi[1] = colend;
    
    NGA_Put (pfock->ga_D, lo, hi, dmat, &stride);
    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_fillDenMat( PFock_t pfock,
                                int index,
                                double value )
{
    if (pfock->committed == 1)
    {
        PFOCK_PRINTF (1, "can't change density matrix"
                         "after PFock_commitDenMats().\n");
        return PFOCK_STATUS_EXECUTION_FAILED;
    }
    
    if (index < 0 ||
        index >= pfock->numdmat)
    {
        PFOCK_PRINTF (1, "invalid index\n");
        return PFOCK_STATUS_INVALID_VALUE;
    }
    
    GA_Fill (pfock->ga_D, &value);
    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_commitDenMats (PFock_t pfock)
{
    GA_Sync ();
    pfock->committed = 1;
    
    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_getMat( PFock_t pfock,
                            int index,
                            PFockMatType_t type,
                            int rowstart,
                            int rowend,
                            int colstart,
                            int colend,
                            double *mat,
                            int stride )
{
    int lo[2];
    int hi[2];
    int ga;

    lo[0] = rowstart;
    hi[0] = rowend;    
    lo[1] = colstart;
    hi[1] = colend;

    ga = pfock->gatable[type];
    NGA_Get (ga, lo, hi, mat, &stride);
    
    return PFOCK_STATUS_SUCCESS;    
}


PFockStatus_t PFock_getLocalMatInds( PFock_t pfock,
                                     int *rowstart,
                                     int *rowend,
                                     int *colstart,
                                     int *colend )
{
    int lo[2];
    int hi[2];
    int myrank;

    
    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
    NGA_Distribution (pfock->ga_D[0], myrank, lo, hi);

    *rowstart = lo[0];
    *rowend = hi[0];
    *colstart = lo[1];
    *colend = hi[1];

    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_getLocalMatPtr ( PFock_t pfock,
                                     int index,
                                     PFockMatType_t type,
                                     int *rowstart,
                                     int *rowend,
                                     int *colstart,
                                     int *colend,
                                     double **mat,
                                     int *stride )
{
    int lo[2];
    int hi[2];
    int myrank;
    int ga;

    if (index < 0 ||
        index >= pfock->maxnumdmat)
    {
        PFOCK_PRINTF (1, "invalid index\n");
        return PFOCK_STATUS_INVALID_VALUE;
    }
    ga = pfock->gatable[type];
    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
    NGA_Distribution (ga[index], myrank, lo, hi);
    NGA_Access (ga, lo, hi, mat, stride);
    *rowstart = lo[0];
    *rowend = hi[0];
    *colstart = lo[1];
    *colend = hi[1];
    
    return PFOCK_STATUS_SUCCESS;  
}


PFockStatus_t PFock_getMatGAHandle( PFock_t pfock,
                                    int index,
                                    PFockMatType_t type,
                                    int *ga )
{
    *ga = pfock->gatable[type];
    return PFOCK_STATUS_SUCCESS;
}



static inline int select_victim (int myrank, int nprocs, int *counter_vals)
{
    int victim;
    int idx = (double)rand()/RAND_MAX * nprocs;

    while (1)
    {
        victim = (myrank + idx + 1)%nprocs;
        if (counter_vals[victim] == 0)
        {
            break;
        }
        idx++;
    }

    return victim;
}


PFockStatus_t PFock_computeFock (PFock_t pfock, BasisSet_t basis)
{
    double dzero = 0.0;
    int myrank;
    int task;
    int my_sshellrow;
    int my_sshellcol;
    int startM;
    int endM;
    int startP;
    int endP;
    int rowid;
    int colid;
    int myrow;
    int mycol;
    double *D1;
    double *D2;
    double *D3;
    double *F1 = pfock->F1;
    double *F2 = pfock->F2;
    double *F3 = pfock->F3;
    double *F4 = pfock->F4;
    double *F5 = pfock->F5;
    double *F6 = pfock->F6;
    int maxrowsize = pfock->maxrowsize;
    int maxcolfuncs = pfock->maxcolfuncs;
    int maxcolsize = pfock->maxcolsize;
    int ldX1 = maxrowsize;
    int ldX2 = maxcolsize;
    int ldX3 = maxcolsize;
    int ldX4 = maxcolfuncs;
    int ldX5 = maxcolsize;
    int ldX6 = maxcolfuncs;
#ifdef GA_NB
    ga_nbhdl_t nbhdlF1;
    ga_nbhdl_t nbhdlF2;
    ga_nbhdl_t nbhdlF3;
#endif
    struct timeval tv1;
    struct timeval tv2;
    struct timeval tv3;
    struct timeval tv4;
    int sizeX1 = pfock->sizeX1;
    int sizeX2 = pfock->sizeX2;
    int sizeX3 = pfock->sizeX3;
    int sizeX4 = pfock->sizeX4;    
    int sizeX5 = pfock->sizeX5;
    int sizeX6 = pfock->sizeX6;
    double done = 1.0;
    int lo[2];
    int hi[2];
    int ldD;
    
    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
    pfock->committed = 0;      
    pfock->timepass = 0.0;
    pfock->timereduce = 0.0;
    pfock->timeinit = 0.0;
    pfock->timecomp = 0.0;
    pfock->timegather = 0.0;
    pfock->timescatter = 0.0;
    pfock->usq = 0.0;
    pfock->uitl = 0.0;
    pfock->steals = 0.0;
    pfock->stealfrom = 0.0;
    pfock->ngacalls = 0.0;
    pfock->volumega = 0.0;

    gettimeofday (&tv1, NULL);    
    gettimeofday (&tv3, NULL);
    GA_Fill (pfock->ga_F, &dzero);
    GA_Fill (pfock->ga_F1, &dzero);
    GA_Fill (pfock->ga_F2, &dzero);
    GA_Fill (pfock->ga_F3, &dzero);
    my_sshellrow = pfock->sshell_row;
    my_sshellcol = pfock->sshell_col;
    myrow = myrank/pfock->npcol;
    mycol = myrank%pfock->npcol;    
    // local my D
    load_local_bufD (pfock);
    lo[0] = myrank;
    hi[0] = myrank;
    lo[1] = 0;
    hi[1] = sizeX1 - 1;
    NGA_Access (pfock->ga_D1, lo, hi, &D1, &ldD);
    hi[1] = sizeX2 - 1;
    NGA_Access (pfock->ga_D2, lo, hi, &D2, &ldD);
    hi[1] = sizeX3 - 1;
    NGA_Access (pfock->ga_D3, lo, hi, &D3, &ldD);
    gettimeofday (&tv4, NULL);
    pfock->timegather += (tv4.tv_sec - tv3.tv_sec) +
        (tv4.tv_usec - tv3.tv_usec) / 1000.0 / 1000.0;
    pfock->ngacalls += 3;
    pfock->volumega += (sizeX1 + sizeX2 + sizeX3) * sizeof(double);
    
    gettimeofday (&tv3, NULL);
#ifdef __INTEL_OFFLOAD
    if (pfock->offload == 1)
    {
        offload_copy_D2 (pfock->mic_numdevs, D1, D2, D3,
                         pfock->sizeX1, pfock->sizeX2, pfock->sizeX3);
        offload_reset_F (pfock->mic_numdevs);
    }
#endif    
    reset_F (pfock->numF, F1, F2, F3, F4, F5, F6,
             sizeX1, sizeX2, sizeX3, sizeX4, sizeX5, sizeX6);
#ifdef __INTEL_OFFLOAD
    if (pfock->offload == 1)
    {
        offload_wait_mic (pfock->mic_numdevs);
    }
#endif
    gettimeofday (&tv4, NULL);
    pfock->timeinit += (tv4.tv_sec - tv3.tv_sec) +
        (tv4.tv_usec - tv3.tv_usec) / 1000.0 / 1000.0;
    
    /* own part */
    reset_taskq (pfock);
    while ((task = taskq_next (pfock, myrow, mycol, 1)) < pfock->ntasks)
    {       
        rowid = task/pfock->nblks_col;
        colid = task%pfock->nblks_col;
        startM = pfock->blkrowptr_sh[pfock->sblk_row + rowid];
        endM = pfock->blkrowptr_sh[pfock->sblk_row + rowid + 1] - 1;
        startP = pfock->blkcolptr_sh[pfock->sblk_col + colid];
        endP = pfock->blkcolptr_sh[pfock->sblk_col + colid + 1] - 1;

        gettimeofday (&tv3, NULL);
    #ifdef __INTEL_OFFLOAD    
        if (pfock->offload == 1)
        {
            offload_fock_task (pfock->mic_numdevs,
                               basis, pfock->erd, pfock->ncpu_f,
                               pfock->shellptr, pfock->shellvalue,
                               pfock->shellid, pfock->shellrid,
                               pfock->f_startind,
                               pfock->rowpos, pfock->colpos,
                               pfock->rowptr, pfock->colptr,
                               pfock->tolscr2,
                               my_sshellrow, my_sshellcol,
                               startM, endM, startP, endP,
                               D1, D2, D3,
                               F1, F2, F3, F4, F5, F6,
                               ldX1, ldX2, ldX3, ldX4, ldX5, ldX6,
                               sizeX1, sizeX2, sizeX3,
                               sizeX4, sizeX5, sizeX6,
                               pfock->mic_fraction);                               
        }
        else
    #endif        
        {
            fock_task (basis, pfock->erd, pfock->ncpu_f,
                       pfock->shellptr, pfock->shellvalue,
                       pfock->shellid, pfock->shellrid,
                       pfock->f_startind,
                       pfock->rowpos, pfock->colpos,
                       pfock->rowptr, pfock->colptr,
                       pfock->tolscr2,
                       my_sshellrow, my_sshellcol,
                       startM, endM, startP, endP,
                       D1, D2, D3,
                       F1, F2, F3, F4, F5, F6,
                       ldX1, ldX2, ldX3, ldX4, ldX5, ldX6,
                       sizeX1, sizeX2, sizeX3,
                       sizeX4, sizeX5, sizeX6,
                       &(pfock->uitl), &(pfock->usq));
        }
        gettimeofday (&tv4, NULL);
        pfock->timecomp += (tv4.tv_sec - tv3.tv_sec) +
                    (tv4.tv_usec - tv3.tv_usec) / 1000.0 / 1000.0;
    } /* own part */

    gettimeofday (&tv3, NULL);

    
    // reduction on MIC
#ifdef __INTEL_OFFLOAD    
    if (pfock->offload == 1)
    {
        offload_reduce_mic (pfock->mic_numdevs,
                            pfock->nfuncs_row,
                            pfock->nfuncs_col,
                            pfock->rowpos[my_sshellrow],
                            pfock->colpos[my_sshellcol],
                            pfock->F1_offload,
                            pfock->F2_offload,
                            pfock->F3_offload,
                            pfock->sizeX1,
                            pfock->sizeX2,
                            pfock->sizeX3);
    }
#endif    
    // reduction on CPU
    reduce_F (pfock->numF, F1, F2, F3, F4, F5, F6,
              sizeX1, sizeX2, sizeX3,
              sizeX4, sizeX5, sizeX6,
              maxrowsize, maxcolsize,
              pfock->nfuncs_row, pfock->nfuncs_col,
              pfock->rowpos[my_sshellrow],
              pfock->colpos[my_sshellcol],
              ldX3, ldX4, ldX5, ldX6);   
#ifdef __INTEL_OFFLOAD
    if (pfock->offload == 1)
    {
        offload_wait_mic (pfock->mic_numdevs);
        offload_reduce (pfock->mic_numdevs,
                        F1, F2, F3,
                        pfock->F1_offload,
                        pfock->F2_offload,
                        pfock->F3_offload,
                        pfock->sizeX1,
                        pfock->sizeX2,
                        pfock->sizeX3);
    }
#endif    
    lo[0] = myrank;
    hi[0] = myrank;
    lo[1] = 0;
    hi[1] = sizeX1 - 1;
#ifdef GA_NB   
    // save results for local intergrals
    NGA_NbAcc (pfock->ga_F1, lo, hi, F1, &sizeX1, &done, &nbhdlF1);
    hi[1] = sizeX2 - 1;
    NGA_NbAcc (pfock->ga_F2, lo, hi, F2, &sizeX2, &done, &nbhdlF2); 
    hi[1] = sizeX3 - 1;
    NGA_NbAcc (pfock->ga_F3, lo, hi, F3, &sizeX3, &done, &nbhdlF3); 
#else
    // save results for local intergrals
    NGA_Acc (pfock->ga_F1, lo, hi, F1, &sizeX1, &done);
    hi[1] = sizeX2 - 1;
    NGA_Acc (pfock->ga_F2, lo, hi, F2, &sizeX2, &done); 
    hi[1] = sizeX3 - 1;
    NGA_Acc (pfock->ga_F3, lo, hi, F3, &sizeX3, &done);
#endif
    pfock->ngacalls += 3;
    pfock->volumega += (sizeX1 + sizeX2 + sizeX3) * sizeof(double);
    gettimeofday (&tv4, NULL);
    pfock->timereduce += (tv4.tv_sec - tv3.tv_sec) +
        (tv4.tv_usec - tv3.tv_usec) / 1000.0 / 1000.0;

    pfock->steals = 0;
    pfock->stealfrom = 0;
    if (pfock->tosteal == 0)
        goto end;
    pfock->tosteal = 0;
#ifdef __DYNAMIC__    
    //PFOCK_INFO ("rank %d starts to steal\n", myrank);   
    // victim info
    int vpid;
    int vrow;
    int vcol;
    //int vntasks;    
    int vsblk_row;
    int vsblk_col;
    int vnblks_col;
    int vnfuncs_row;
    int vnfuncs_col;
    int vsshellrow;
    int vsshellcol;
    int prevrow;
    int prevcol;
    int stealed;
    double *D1_task;
    double *D2_task;
    double *VD1 = pfock->D1;
    double *VD2 = pfock->D2;
    double *VD3 = pfock->D3;
#ifdef GA_NB
    ga_nbhdl_t nbhdlD1;
    ga_nbhdl_t nbhdlD2;
    ga_nbhdl_t nbhdlD3;
#endif
    int finished = 0;
    int counter_vals[16000] = {0};
    counter_vals[myrank] = 1;
    prevrow = myrow;
    prevcol = mycol;
    
    /* steal tasks */
    while (!finished && pfock->nprocs != 1)
    //for (idx = 0; idx < pfock->nprocs - 1; idx++)
    {
        vpid = select_victim (myrank, pfock->nprocs, counter_vals);
        //vpid = (myrank + idx + 1)%pfock->nprocs;
        vrow = vpid/pfock->npcol;
        vcol = vpid%pfock->npcol;
        vsblk_row = pfock->rowptr_blk[vrow];
        vsblk_col = pfock->colptr_blk[vcol];
        vnblks_col = pfock->colptr_blk[vcol + 1] - vsblk_col;       
        //vntasks = vnblks_col * (pfock->rowptr_blk[vrow + 1] - vsblk_row);
        vnfuncs_row = pfock->rowptr_f[vrow + 1] - pfock->rowptr_f[vrow];
        vnfuncs_col = pfock->colptr_f[vcol + 1] - pfock->colptr_f[vcol];
        vsshellrow = pfock->rowptr_sh[vrow];   
        vsshellcol = pfock->colptr_sh[vcol];
        stealed = 0;
        while ((task = taskq_next (pfock, vrow, vcol, 1)) < pfock->ntasks)
        {
            gettimeofday (&tv3, NULL);
            if (0 == stealed)
            {
                pfock->tosteal = 1;
                if (vrow != prevrow && vrow != myrow)
                {
                    D1_task = VD1;
                    lo[0] = vpid;
                    hi[0] = vpid;
                    lo[1] = 0;
                    hi[1] = sizeX1 - 1;
                #ifdef GA_NB    
                    NGA_NbGet (pfock->ga_D1, lo, hi,  VD1, &sizeX1, &nbhdlD1);
                #else
                    NGA_Get (pfock->ga_D1, lo, hi,  VD1, &sizeX1);                
                #endif
                    pfock->ngacalls += 1;
                    pfock->volumega += sizeX1 * sizeof(double);
                }
                else if (vrow == myrow)
                {
                    D1_task = D1;
                }  
                if (vcol != prevcol && vcol != mycol)
                {
                    D2_task =  VD2;
                    lo[0] = vpid;
                    hi[0] = vpid;
                    lo[1] = 0;
                    hi[1] = sizeX2 - 1;
                #ifdef GA_NB
                    NGA_NbGet (pfock->ga_D2, lo, hi, VD2, &sizeX2, &nbhdlD2);
                #else
                    NGA_Get (pfock->ga_D2, lo, hi, VD2, &sizeX2);               
                #endif
                    pfock->ngacalls += 1;
                    pfock->volumega += sizeX2 * sizeof(double);
                }
                else if (vcol == mycol)
                {
                    D2_task = D2;
                }
                lo[0] = vpid;
                hi[0] = vpid;
                lo[1] = 0;
                hi[1] = sizeX3 - 1;              
            #ifdef GA_NB
                NGA_NbGet (pfock->ga_D3, lo, hi,  VD3, &sizeX3, &nbhdlD3);
            #else
                NGA_Get (pfock->ga_D3, lo, hi,  VD3, &sizeX3);
            #endif
                pfock->ngacalls += 1;
                pfock->volumega += sizeX3 * sizeof(double);
            #ifdef GA_NB    
                // wait for last NbAcc F
                NGA_NbWait (&nbhdlF1);
                NGA_NbWait (&nbhdlF2);
                NGA_NbWait (&nbhdlF3);
            #endif    
                // init F bufs
                reset_F (pfock->numF, F1, F2, F3, F4, F5, F6,
                         sizeX1, sizeX2, sizeX3, sizeX4, sizeX5, sizeX6);
            #ifdef GA_NB    
                // wait for NbGet
                if (vrow != prevrow && vrow != myrow)
                {
                    NGA_NbWait (&nbhdlD1);
                }
                if (vcol != prevcol && vcol != mycol)
                {
                    NGA_NbWait (&nbhdlD2);
                }
                NGA_NbWait (&nbhdlD3);
            #endif    
                pfock->stealfrom++;
            }
            gettimeofday (&tv4, NULL);
            pfock->timeinit += (tv4.tv_sec - tv3.tv_sec) +
                   (tv4.tv_usec - tv3.tv_usec) / 1000.0 / 1000.0;
            rowid = task/vnblks_col;
            colid = task%vnblks_col;
            // compute task
            startM = pfock->blkrowptr_sh[vsblk_row + rowid];
            endM = pfock->blkrowptr_sh[vsblk_row + rowid + 1] - 1;
            startP = pfock->blkcolptr_sh[vsblk_col + colid];
            endP = pfock->blkcolptr_sh[vsblk_col + colid + 1] - 1;

            gettimeofday (&tv3, NULL);
            fock_task (basis, pfock->erd, pfock->ncpu_f,
                       pfock->shellptr, pfock->shellvalue,
                       pfock->shellid, pfock->shellrid,
                       pfock->f_startind,
                       pfock->rowpos, pfock->colpos,
                       pfock->rowptr, pfock->colptr,
                       pfock->tolscr2,
                       vsshellrow, vsshellcol,
                       startM, endM, startP, endP,
                       D1_task, D2_task, VD3,                     
                       F1, F2, F3, F4, F5, F6,
                       ldX1, ldX2, ldX3, ldX4, ldX5, ldX6,
                       sizeX1, sizeX2, sizeX3,
                       sizeX4, sizeX5, sizeX6,
                       &(pfock->uitl), &(pfock->usq));
            gettimeofday (&tv4, NULL);
            pfock->timecomp += (tv4.tv_sec - tv3.tv_sec) +
                        (tv4.tv_usec - tv3.tv_usec) / 1000.0 / 1000.0;
            pfock->steals++;
            stealed = 1;
        }
        gettimeofday (&tv3, NULL);
        if (1 == stealed)
        {
            // reduction
            reduce_F (pfock->numF, F1, F2, F3, F4, F5, F6,
                      sizeX1, sizeX2, sizeX3,
                      sizeX4, sizeX5, sizeX6,
                      maxrowsize, maxcolsize,
                      vnfuncs_row, vnfuncs_col,
                      pfock->rowpos[vsshellrow],
                      pfock->colpos[vsshellcol],
                      ldX3, ldX4, ldX5, ldX6);
            lo[1] = 0;
            hi[1] = sizeX1 - 1;
            if (vrow != myrow)
            {
                lo[0] = vpid;
                hi[0] = vpid;
            #ifdef GA_NB    
                NGA_NbAcc (pfock->ga_F1, lo, hi, F1, &sizeX1,
                           &done, &nbhdlF1);
            #else
                NGA_Acc (pfock->ga_F1, lo, hi, F1, &sizeX1, &done);
            #endif
                pfock->ngacalls += 1;
                pfock->volumega += sizeX1 * sizeof(double);
            }
            else
            {
                lo[0] = myrank;
                hi[0] = myrank;
            #ifdef GA_NB    
                NGA_NbAcc (pfock->ga_F1, lo, hi, F1, &sizeX1,
                           &done, &nbhdlF1);
            #else
                NGA_Acc (pfock->ga_F1, lo, hi, F1, &sizeX1, &done);
            #endif
            }
            lo[1] = 0;
            hi[1] = sizeX2 - 1;
            if (vcol != mycol)
            {
                lo[0] = vpid;
                hi[0] = vpid;
            #ifdef GA_NB    
                NGA_NbAcc (pfock->ga_F2, lo, hi, F2, &sizeX2,
                           &done, &nbhdlF2);
            #else
                NGA_Acc (pfock->ga_F2, lo, hi, F2, &sizeX2, &done);
            #endif
                pfock->ngacalls += 1;
                pfock->volumega += sizeX2 * sizeof(double);
            }
            else
            {
                lo[0] = myrank;
                hi[0] = myrank;
            #ifdef GA_NB
                NGA_NbAcc (pfock->ga_F2, lo, hi, F2, &sizeX2, &done, &nbhdlF2);
            #else
                NGA_Acc (pfock->ga_F2, lo, hi, F2, &sizeX2, &done);
            #endif
            }
            lo[0] = vpid;
            hi[0] = vpid;
            lo[1] = 0;
            hi[1] = sizeX3 - 1;
        #ifdef GA_NB
            NGA_NbAcc (pfock->ga_F3, lo, hi, F3, &sizeX3, &done, &nbhdlF3);
        #else
            NGA_Acc (pfock->ga_F3, lo, hi, F3, &sizeX3, &done);
        #endif
            pfock->ngacalls += 1;
            pfock->volumega += sizeX3 * sizeof(double);
            prevrow = vrow;
            prevcol = vcol;
        }
        counter_vals[vpid] = 1;
        
        finished = 1;
        for (int i = 0; i < pfock->nprocs; i++)
        {
            if (counter_vals[i] == 0)
            {
                finished = 0;
                break;
            }
        }
        gettimeofday (&tv4, NULL);
        pfock->timereduce += (tv4.tv_sec - tv3.tv_sec) +
                        (tv4.tv_usec - tv3.tv_usec) / 1000.0 / 1000.0;
    } /* steal tasks */    
#endif /* #ifdef __DYNAMIC__ */

end:
#ifdef GA_NB
    // wait for last NbAcc F
    NGA_NbWait (&nbhdlF1);
    NGA_NbWait (&nbhdlF2);
    NGA_NbWait (&nbhdlF3);
#endif
    lo[0] = myrank;
    hi[0] = myrank;
    lo[1] = 0;
    hi[1] = sizeX1 - 1;
    NGA_Release (pfock->ga_D1, lo, hi);
    hi[1] = sizeX2 - 1;
    NGA_Release (pfock->ga_D2, lo, hi);
    hi[1] = sizeX3 - 1;
    NGA_Release (pfock->ga_D3, lo, hi);    
    
    gettimeofday (&tv2, NULL);
    pfock->timepass = (tv2.tv_sec - tv1.tv_sec) +
               (tv2.tv_usec - tv1.tv_usec) / 1000.0 / 1000.0;    
    GA_Sync ();
    
    gettimeofday (&tv3, NULL);
    store_local_bufF (pfock);
    gettimeofday (&tv4, NULL);
    pfock->timescatter = (tv4.tv_sec - tv3.tv_sec) +
               (tv4.tv_usec - tv3.tv_usec) / 1000.0 / 1000.0;

    if (myrank == 0)
    {
        PFOCK_INFO ("correct F ...\n");
    }
    
    // correct F
    GA_Symmetrize (pfock->ga_F);
    
    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_createCoreHMat ( PFock_t pfock,
                                     BasisSet_t basis,
                                     CoreH_t *hmat )
{
    int ga;
    int lo[2];
    int hi[2];
    CoreH_t h;
    int myrank;
    double *mat;
    int stride;
    double dzero = 0.0;
    
    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);    
    h = (CoreH_t)ALIGN_MALLOC (sizeof(struct CoreH), 64);
    ga = GA_Duplicate (pfock->ga_D, "core hamilton mat");
    if (0 == ga)
    {
        PFOCK_PRINTF (1, "GA allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }
    GA_Fill (ga, &dzero);
    NGA_Distribution (ga, myrank, lo, hi);
    NGA_Access (ga, lo, hi, &mat, &stride);
    
    compute_H (pfock, basis, pfock->sshell_row, pfock->eshell_row,
               pfock->sshell_col, pfock->eshell_col, mat, stride);
    NGA_Release_update (ga, lo, hi);
    GA_Sync ();
    
    h->ga = ga;
    *hmat = h;
    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_destroyCoreHMat ( CoreH_t hmat )
{
    GA_Destroy (hmat->ga);
    ALIGN_FREE (hmat);

    return PFOCK_STATUS_SUCCESS;    
}


PFockStatus_t PFock_getCoreHMatGAHandle( CoreH_t hmat,
                                         int *ga )
{
    *ga = hmat->ga;
    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_getCoreHMat( CoreH_t hmat,
                                 int rowstart,
                                 int rowend,
                                 int colstart,
                                 int colend,
                                 double *mat,
                                 int stride )
{
    int lo[2];
    int hi[2];
    int ga;

    ga = hmat->ga;
    lo[0] = rowstart;
    hi[0] = rowend;    
    lo[1] = colstart;
    hi[1] = colend;    
    NGA_Get (ga, lo, hi, mat, &stride);
    return PFOCK_STATUS_SUCCESS;    
}


PFockStatus_t PFock_createOvlMat ( PFock_t pfock,
                                   BasisSet_t basis,
                                   Ovl_t *omat )
{
    double *mat;
    int ga;
    int lo[2];
    int hi[2];
    Ovl_t o;
    int myrank;
    int stride;
    double dzero = 0.0;
    
    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);    
    o = (Ovl_t)ALIGN_MALLOC (sizeof(struct Ovl), 64);
    ga = GA_Duplicate (pfock->ga_D, "overlap mat");
    if (NULL == o)
    {
        PFOCK_PRINTF (1, "memory allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }
    if (0 == ga)
    {
        PFOCK_PRINTF (1, "GA allocation failed\n");
        return PFOCK_STATUS_ALLOC_FAILED;
    }
    GA_Fill (ga, &dzero);
    NGA_Distribution (ga, myrank, lo, hi);
    NGA_Access (ga, lo, hi, &mat, &stride);
    
    compute_S (pfock, basis, pfock->sshell_row, pfock->eshell_row,
               pfock->sshell_col, pfock->eshell_col, mat, stride);
    NGA_Release_update (ga, lo, hi);
    GA_Sync ();
    
    o->ga = ga;
    *omat = o;
    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_destroyOvlMat ( Ovl_t omat )
{
    GA_Destroy (omat->ga);
    ALIGN_FREE (omat);

    return PFOCK_STATUS_SUCCESS;    
}


PFockStatus_t PFock_getOvlMatGAHandle( Ovl_t omat,
                                       int *ga )
{
    *ga = omat->ga;
    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_getOvlMat( Ovl_t omat,
                               int rowstart,
                               int rowend,
                               int colstart,
                               int colend,
                               double *mat,
                               int stride )
{
    int lo[2];
    int hi[2];
    int ga;

    ga = omat->ga;
    lo[0] = rowstart;
    hi[0] = rowend;    
    lo[1] = colstart;
    hi[1] = colend;    
    NGA_Get (ga, lo, hi, mat, &stride);
    return PFOCK_STATUS_SUCCESS;    
}


PFockStatus_t PFock_GAInit( int nbf,
                            int nprow,
                            int npcol,
                            int numdenmat,
                            int sizeheap,
                            int sizestack )
{
    int stack;
    int heap;
    int maxrowsize;
    int maxcolsize;
    
    maxrowsize = (nbf + nprow - 1)/nprow;
    maxcolsize = (nbf + npcol - 1)/npcol;    
    heap = numdenmat * 5 * maxrowsize * maxcolsize * sizeof(double);
    stack = heap;
    heap += sizeheap;
    stack += sizestack;

#if ( _DEBUG_LEVEL_ > 2 )
    printf ("MA_init(): GA heap = %lf KB, stack = %lf KB\n",
            heap/1024.0, stack/1024.0);
#endif
    GA_Initialize ();
    if (!MA_init (C_DBL, heap, stack))
    {
        printf ("MA_init() failed\n");
        return PFOCK_STATUS_INIT_FAILED;
    }
    
    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_GAFinalize (void)
{    
    GA_Terminate ();
   
    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_getMemorySize (PFock_t pfock,
                                   double *mem_cpu,
                                   double *mem_mic,
                                   double *erd_mem_cpu,
                                   double *erd_mem_mic)
{
    *mem_cpu = pfock->mem_cpu;
    *mem_mic = pfock->mem_mic;
#ifdef __INTEL_OFFLOAD    
    if (pfock->offload == 1)
    {
        CInt_offload_getMaxMemory (pfock->erd, erd_mem_cpu, erd_mem_mic);        
    }
    else
#endif        
    {
        CInt_getMaxMemory (pfock->erd, erd_mem_cpu);
    }

    return PFOCK_STATUS_SUCCESS;
}


PFockStatus_t PFock_getStatistics (PFock_t pfock)
{
    int myrank;
    double tsq;

    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
    
    // statistics
    MPI_Gather (&pfock->steals, 1, MPI_DOUBLE,
        pfock->mpi_steals, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather (&pfock->stealfrom, 1, MPI_DOUBLE,
        pfock->mpi_stealfrom, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather (&pfock->usq, 1, MPI_DOUBLE, 
        pfock->mpi_usq, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather (&pfock->uitl, 1, MPI_DOUBLE, 
        pfock->mpi_uitl, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather (&pfock->timepass, 1, MPI_DOUBLE, 
        pfock->mpi_timepass, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather (&pfock->timecomp, 1, MPI_DOUBLE, 
        pfock->mpi_timecomp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather (&pfock->timeinit, 1, MPI_DOUBLE, 
        pfock->mpi_timeinit, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather (&pfock->timereduce, 1, MPI_DOUBLE, 
        pfock->mpi_timereduce, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather (&pfock->timegather, 1, MPI_DOUBLE, 
        pfock->mpi_timegather, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather (&pfock->timescatter, 1, MPI_DOUBLE, 
        pfock->mpi_timescatter, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather (&pfock->volumega, 1, MPI_DOUBLE, 
        pfock->mpi_volumega, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather (&pfock->ngacalls, 1, MPI_DOUBLE, 
        pfock->mpi_ngacalls, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (myrank == 0)
    {
        double total_timepass;
        double max_timepass;
        double total_timereduce;
        double total_timeinit;
        double total_timecomp;
        double total_timegather;
        double total_timescatter;
        double total_usq;
        double max_usq;
        double total_uitl;
        double max_uitl;
        double total_steals;
        double total_stealfrom;
        double total_ngacalls;
        double total_volumega;
        for (int i = 0; i < pfock->nprocs; i++)
        {
            total_timepass += pfock->mpi_timepass[i];
            max_timepass =
                max_timepass < pfock->mpi_timepass[i] ?
                    pfock->mpi_timepass[i] : max_timepass;            
            total_usq += pfock->mpi_usq[i];
            max_usq = max_usq < pfock->mpi_usq[i] ? pfock->mpi_usq[i] : max_usq;          
            total_uitl += pfock->mpi_uitl[i];
            max_uitl = max_uitl < pfock->mpi_uitl[i] ? pfock->mpi_uitl[i] : max_uitl;           
            total_steals += pfock->mpi_steals[i];
            total_stealfrom += pfock->mpi_stealfrom[i];
            total_timecomp += pfock->mpi_timecomp[i];
            total_timeinit += pfock->mpi_timeinit[i];
            total_timereduce += pfock->mpi_timereduce[i];
            total_timegather += pfock->mpi_timegather[i];
            total_timescatter += pfock->mpi_timescatter[i];
            total_ngacalls += pfock->mpi_ngacalls[i];
            total_volumega += pfock->mpi_volumega[i];
        }
        tsq = pfock->nshells;
        tsq = ((tsq + 1) * tsq/2.0 + 1) * tsq * (tsq + 1)/4.0;
        printf ("    PFock Statistics:\n");
        printf ("      average totaltime   = %.3lf\n"
                "      average timegather  = %.3lf\n"
                "      average timeinit    = %.3lf\n"
                "      average timecomp    = %.3lf\n"
                "      average timereduce  = %.3lf\n"
                "      average timescatter = %.3lf\n"
                "      comp/total = %.3lf\n",
                total_timepass/pfock->nprocs,
                total_timegather/pfock->nprocs,
                total_timeinit/pfock->nprocs,
                total_timecomp/pfock->nprocs,
                total_timereduce/pfock->nprocs,
                total_timescatter/pfock->nprocs,
                total_timecomp/total_timepass);
        #if 1
        printf ("      usq = %.4le (lb = %.3lf)\n"
                "      uitl = %.4le (lb = %.3lf)\n"
                "      nsq = %.4le (screening = %.3lf)\n",
                total_usq, max_usq/(total_usq/pfock->nprocs),
                total_uitl, max_uitl/(total_uitl/pfock->nprocs),
                tsq, total_usq/tsq);
        #endif
        printf ("      load blance = %.3lf\n",
                max_timepass/(total_timepass/pfock->nprocs));
        printf ("      steals = %.3lf (average = %.3lf)\n"
                "      stealfrom = %.3lf (average = %.3lf)\n"
                "      GAcalls = %.3lf\n"
                "      GAvolume %.3lf MB\n",
                total_steals, total_steals/pfock->nprocs,
                total_stealfrom, total_stealfrom/pfock->nprocs,
                total_ngacalls/pfock->nprocs,
                total_volumega/pfock->nprocs/1024.0/1024.0);
    }
    return PFOCK_STATUS_SUCCESS;
}
