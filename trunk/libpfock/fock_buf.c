#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>
#include <ga.h>
#include <macdecls.h>
#include <sys/time.h>

#include "config.h"
#include "taskq.h"
#include "fock_buf.h"


void load_local_bufD (PFock_t pfock)
{
    int A;
    int B;
    int lo[2];
    int hi[2];
    int posrow;
    int poscol;
    int ldD;
    int ldD1;
    int ldD2;
    int ldD3;
    int *loadrow;
    int sizerow;
    int *loadcol;
    int sizecol;
    double *D1;
    double *D2;
    double *D3;
#ifdef GA_NB
    ga_nbhdl_t nbnb;
#endif

    loadrow = pfock->loadrow;
    loadcol = pfock->loadcol;
    sizerow = pfock->sizeloadrow;
    sizecol = pfock->sizeloadcol;

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
    ldD1 = pfock->ldX1;
    ldD2 = pfock->ldX2;
    ldD3 = pfock->ldX3;
        
    /* D1 */
    lo[0] = pfock->sfunc_row;
    hi[0] = pfock->efunc_row;
    for (A = 0; A < sizerow; A++)
    {
        lo[1] = loadrow[PLEN * A + P_LO];
        hi[1] = loadrow[PLEN * A + P_HI];
        posrow = loadrow[PLEN * A + P_W];
    #ifdef GA_NB
        NGA_NbGet (pfock->ga_D, lo, hi, &(D1[posrow]), &ldD1, &nbnb);
    #else
        NGA_Get (pfock->ga_D, lo, hi, &(D1[posrow]), &ldD1);
    #endif
    }

    /* D2 */
    lo[0] = pfock->sfunc_col;
    hi[0] = pfock->efunc_col;
    for (B = 0; B < sizecol; B++)
    {
        lo[1] = loadcol[PLEN * B + P_LO];
        hi[1] = loadcol[PLEN * B + P_HI];
        poscol = loadcol[PLEN * B + P_W];
    #ifdef GA_NB    
        NGA_NbGet (pfock->ga_D, lo, hi, &(D2[poscol]), &ldD2, &nbnb);
    #else
        NGA_Get (pfock->ga_D, lo, hi, &(D2[poscol]), &ldD2);
    #endif
    }

    /* D3 */
    for (A = 0; A < sizerow; A++)
    {
        lo[0] = loadrow[PLEN * A + P_LO];
        hi[0] = loadrow[PLEN * A + P_HI];
        posrow = loadrow[PLEN * A + P_W];
        for (B = 0; B < sizecol; B++)
        {
            lo[1] = loadcol[PLEN * B + P_LO];
            hi[1] = loadcol[PLEN * B + P_HI];
            poscol = loadcol[PLEN * B + P_W];
        #ifdef GA_NB
            NGA_NbGet (pfock->ga_D, lo, hi, &(D3[posrow * ldD3 + poscol]), &ldD3, &nbnb);
        #else
            NGA_Get (pfock->ga_D, lo, hi, &(D3[posrow * ldD3 + poscol]), &ldD3);        
        #endif
        }
    }
#ifdef GA_NB
    /* Jeff: You have to wait on local completion,
     *       which is also "remote" for Get. */
    NGA_NbWait (&nbnb);
#endif

    lo[0] = myrank;
    hi[0] = myrank;
    lo[1] = 0;
    hi[1] = pfock->sizeX1 - 1;
    NGA_Release_update (pfock->ga_D1, lo, hi);
    hi[1] = pfock->sizeX2 - 1;
    NGA_Release_update (pfock->ga_D2, lo, hi);
    hi[1] = pfock->sizeX3 - 1;
    NGA_Release_update (pfock->ga_D3, lo, hi);

    /* Jeff: This may not be necessary... */
    //GA_Sync ();
}


void store_local_bufF (PFock_t pfock)
{
    int A;
    int B;
    int lo[2];
    int hi[2];
    int posrow;
    int poscol;
    double done = 1.0;
    int ldF1;
    int ldF2;
    int ldF3;
    int *loadrow;
    int sizerow;
    int *loadcol;
    int sizecol;
    double *F1;
    double *F2;
    double *F3;
#ifdef GA_NB    
    ga_nbhdl_t nbnb;
#endif
    loadrow = pfock->loadrow;
    loadcol = pfock->loadcol;
    sizerow = pfock->sizeloadrow;
    sizecol = pfock->sizeloadcol;

    int myrank;
    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);

    lo[0] = myrank;
    hi[0] = myrank;
    lo[1] = 0;
    hi[1] = pfock->sizeX1 - 1;
    int ldF;
    NGA_Access (pfock->ga_F1, lo, hi, &F1, &ldF);
    lo[1] = 0;
    hi[1] = pfock->sizeX2 - 1;
    NGA_Access (pfock->ga_F2, lo, hi, &F2, &ldF);
    lo[1] = 0;
    hi[1] = pfock->sizeX3 - 1;
    NGA_Access (pfock->ga_F3, lo, hi, &F3, &ldF);
    ldF1 = pfock->ldX1;
    ldF2 = pfock->ldX2;
    ldF3 = pfock->ldX3;
    
    /* F1 */
    lo[0] = pfock->sfunc_row;
    hi[0] = pfock->efunc_row;
    for (A = 0; A < sizerow; A++)
    {
        lo[1] = loadrow[PLEN * A + P_LO];
        hi[1] = loadrow[PLEN * A + P_HI];
        posrow = loadrow[PLEN * A + P_W];
    #ifdef GA_NB
        NGA_NbAcc (pfock->ga_F, lo, hi, &(F1[posrow]), &ldF1, &done, &nbnb);    
    #else
        NGA_Acc (pfock->ga_F, lo, hi, &(F1[posrow]), &ldF1, &done);
    #endif
    }

    /* D2 */
    lo[0] = pfock->sfunc_col;
    hi[0] = pfock->efunc_col;
    for (B = 0; B < sizecol; B++)
    {
        lo[1] = loadcol[PLEN * B + P_LO];
        hi[1] = loadcol[PLEN * B + P_HI];
        poscol = loadcol[PLEN * B + P_W];
    #ifdef GA_NB
        NGA_NbAcc (pfock->ga_F, lo, hi, &(F2[poscol]), &ldF2, &done, &nbnb);
    #else
        NGA_Acc (pfock->ga_F, lo, hi, &(F2[poscol]), &ldF2, &done);
    #endif
    }

    /* D3 */
    for (A = 0; A < sizerow; A++)
    {
        lo[0] = loadrow[PLEN * A + P_LO];
        hi[0] = loadrow[PLEN * A + P_HI];
        posrow = loadrow[PLEN * A + P_W];
        for (B = 0; B < sizecol; B++)
        {
            lo[1] = loadcol[PLEN * B + P_LO];
            hi[1] = loadcol[PLEN * B + P_HI];
            poscol = loadcol[PLEN * B + P_W];
        #ifdef GA_NB
            NGA_NbAcc (pfock->ga_F, lo, hi, 
                       &(F3[posrow * ldF3 + poscol]), &ldF3, &done, &nbnb);
        #else
            NGA_Acc (pfock->ga_F, lo, hi, 
                     &(F3[posrow * ldF3 + poscol]), &ldF3, &done);        
        #endif
        }
    }
#ifdef GA_NB
    /* Jeff: You have to wait on local completion
     * for remote completion operations to be effective. */
    NGA_NbWait (&nbnb);
    /* Jeff: Once NbWait returns, the local buffers are finished and
     *       you can call release. */
#endif

    lo[0] = myrank;
    hi[0] = myrank;
    lo[1] = 0;
    hi[1] = pfock->sizeX1 - 1;
    NGA_Release (pfock->ga_F1, lo, hi);
    lo[1] = 0;
    hi[1] = pfock->sizeX2 - 1;
    NGA_Release (pfock->ga_F2, lo, hi);
    lo[1] = 0;
    hi[1] = pfock->sizeX3 - 1;
    NGA_Release (pfock->ga_F3, lo, hi);
    
    GA_Sync ();
}


void compute_FD_ptr (PFock_t pfock, int startM, int endM,
                     int *ptrrow, int *rowsize)
{
    int A;
    int B;
    int i;
    int start;
    int end;

    for (A = 0; A < pfock->nshells; A++)
    {
        ptrrow[A] = -1;
    }    
    // init row pointers
    for (A = startM; A <= endM; A++)
    {
        start = pfock->shellptr[A];
        end = pfock->shellptr[A + 1]; 
        for (i = start; i < end; i++)
        {
            B = pfock->shellid[i];
            ptrrow[B] = 1;
        }
    }
    #if 1
    int flag;
    for (i = 0; i < pfock->natoms; i++)
    {
        start = pfock->s_startind[i];
        end = pfock->s_startind[i + 1];
        flag = -1;
        for (A = start; A < end; A++)
        {
            if (ptrrow[A] != -1)
                flag = 1;
        }
        for (A = start; A < end; A++)
        {
            ptrrow[A] = flag;
        }
    }
    #endif
    *rowsize = 0;
    for (A = 0; A < pfock->nshells; A++)
    {
        if (ptrrow[A] == 1)
        {
            ptrrow[A] = *rowsize;           
            *rowsize += pfock->f_startind[A + 1] - pfock->f_startind[A];
        }
    }
}


void init_FD_load (PFock_t pfock, int *ptrrow,
                   int **loadrow, int *loadsize)
{
    int loadcount;
    int A;
    int idx;
    int lo;
    int hi;
    
    loadcount = 0;
    for (A = 0; A < pfock->nshells; A++)
    {
        if (ptrrow[A] != -1)
        {
            while (A < pfock->nshells && ptrrow[A] != -1)
            {
                A++;
            }           
            loadcount++;
        }
    }
    *loadrow = (int *)ALIGN_MALLOC (sizeof(int) * PLEN * loadcount, 64);
    assert (NULL != *loadrow);
    *loadsize = loadcount;
    
    loadcount = 0;
    for (A = 0; A < pfock->nshells; A++)
    {
        idx = ptrrow[A];
        if (idx != -1)
        {
            lo = pfock->f_startind[A];
            while (A < pfock->nshells && ptrrow[A] != -1)
            {
                A++;
            }           
            hi = pfock->f_startind[A] - 1;
            (*loadrow)[loadcount * PLEN + P_LO] = lo;
            (*loadrow)[loadcount * PLEN + P_HI] = hi;
            (*loadrow)[loadcount * PLEN + P_W] = idx;
            loadcount++;
        }
    }
}
