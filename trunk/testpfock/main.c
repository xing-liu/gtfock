#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#include "fock_init.h"
#include "fock_task.h"
#include "fock_offload.h"


#define ALIGNED_8(size) ((((size) + 7)/8)*8)


int main (int argc, char **argv)
{
    int nnz;
    int *shellptr;
    int *shellid;
    int *shellrid;
    double *shellvalue;
    int *rowpos;
    int *colpos;
    int *rowptr;
    int *colptr;
    int rowsize;
    int colsize;
    int rowfuncs;
    int colfuncs;
    int nshells;
    int *f_startind;
    int nprow;
    int npcol;
    int nbp_p;
    int nfuncs;
    int toOffload = 0;
    int mic_numdevs = 0;
    struct timeval tv1, tv2;
    struct timeval tv3, tv4;
    double timeinit;
    double timecomp;
    double timereduce;
    double timepass;
    
    int nthreads_mic;
    double mic_fraction;
    int myrank;

    if (argc != 7)
    {
        printf ("Usage: %s <basis set> <xyz>"
                " <myrank> <nprow> <npcol> <ntasks>\n",
                argv[0]);
        return -1;
    }
    int provided;
    MPI_Init_thread (&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int nthreads = omp_get_max_threads ();
    omp_set_num_threads (nthreads);

    char *offload_str = getenv ("GTFOCK_OFFLOAD");
    if (offload_str == NULL)
    {
        mic_fraction = 0.0;
        toOffload = 0;
    }
    else
    {
        mic_fraction = atof (offload_str);
        assert (mic_fraction <= 1.0);
        if (mic_fraction == 0.0)
        {
            toOffload = 0;
        }
        else
        {
            toOffload = 1;
        }
    }

    myrank = atoi (argv[3]);
    nprow = atoi (argv[4]);
    npcol = atoi (argv[5]);
    nbp_p = atoi (argv[6]);

    // load basis set and create ERD_t
    BasisSet_t basis;
    ERD_t erd;
    if (toOffload == 1)
    {
        CInt_offload_createBasisSet (&basis);
        CInt_offload_loadBasisSet (basis, argv[1], argv[2]);
        
    }
    else
    {
        CInt_createBasisSet (&basis);
        CInt_loadBasisSet (basis, argv[1], argv[2]);
    }

    nshells = CInt_getNumShells (basis);
    nfuncs = CInt_getNumFuncs (basis);
    // functions starting positions of shells
    f_startind = (int *) _mm_malloc (sizeof (int) * (nshells + 1), 64);
    assert (NULL != f_startind);
    for (int i = 0; i < nshells; i++)
    {
        f_startind[i] = CInt_getFuncStartInd (basis, i);
    }
    f_startind[nshells] = nfuncs;

    // fock initialization
    screening (basis, &shellptr, &shellid, &shellrid, &shellvalue, &nnz);    
    int *rowptr_f;
    int *rowptr_sh;
    int *rowptr_blk;
    int *colptr_f;
    int *colptr_sh;
    int *colptr_blk;
    int *blkrowptr_sh;
    int *blkcolptr_sh;
    
    init_grid (nshells, nnz, shellptr, f_startind, nprow, npcol, nbp_p,
               &rowptr_f, &colptr_f, &rowptr_sh, &colptr_sh,
               &rowptr_blk, &colptr_blk, &blkrowptr_sh, &blkcolptr_sh);
    init_buffers (myrank, nshells, nnz, nprow, npcol,
                  shellptr, shellid, f_startind,
                  rowptr_f, colptr_f,
                  rowptr_sh, colptr_sh,
                  rowptr_blk, colptr_blk,
                  &rowpos, &colpos, &rowptr, &colptr,                 
                  &rowsize, &colsize,
                  &rowfuncs, &colfuncs);
    
    // malloc D and F
    double totalFDsize = 0.0;
    double totalFDsize_mic = 0.0;
    int sizeD1 = rowfuncs * rowsize;
    sizeD1 = ALIGNED_8 (sizeD1);
    int sizeD2 = colfuncs * colsize;
    sizeD2 = ALIGNED_8 (sizeD2);
    int sizeD3 = rowsize * colsize;
    sizeD3 = ALIGNED_8 (sizeD3);
    int sizeD4 = rowfuncs * colfuncs;
    sizeD4 = ALIGNED_8 (sizeD4);
    int sizeD5 = rowsize * colfuncs;
    sizeD5 = ALIGNED_8 (sizeD5);
    int sizeD6 = rowfuncs * colsize;
    sizeD6 = ALIGNED_8 (sizeD6);
    
    double *D1 = (double *) _mm_malloc (sizeof (double) * sizeD1, 64);
    double *D2 = (double *) _mm_malloc (sizeof (double) * sizeD2, 64);
    double *D3 = (double *) _mm_malloc (sizeof (double) * sizeD3, 64); 
    double *VD1 = (double *) _mm_malloc (sizeof (double) * sizeD1, 64);
    double *VD2 = (double *) _mm_malloc (sizeof (double) * sizeD2, 64);
    double *VD3 = (double *) _mm_malloc (sizeof (double) * sizeD3, 64);
    double *F1 = (double *) _mm_malloc (sizeof (double) * sizeD1 * nthreads, 64);
    double *F2 = (double *) _mm_malloc (sizeof (double) * sizeD2 * nthreads, 64);
    double *F3 = (double *) _mm_malloc (sizeof (double) * sizeD3, 64);
    double *F4 = (double *) _mm_malloc (sizeof (double) * sizeD4 * nthreads, 64);
    double *F5 = (double *) _mm_malloc (sizeof (double) * sizeD5 * nthreads, 64);
    double *F6 = (double *) _mm_malloc (sizeof (double) * sizeD6 * nthreads, 64);
    totalFDsize += 2.0 * sizeof (double) *
        (sizeD1 + sizeD2 + sizeD3);
    totalFDsize += 1.0 * sizeof (double) *
        ((double)(sizeD1 + sizeD2 + sizeD4 + sizeD5 + sizeD6) * nthreads + sizeD3);
    assert (D1 != NULL &&
            D2 != NULL &&
            D3 != NULL &&
            VD1 != NULL &&
            VD2 != NULL &&
            VD3 != NULL &&
            F1 != NULL &&
            F2 != NULL &&
            F3 != NULL &&
            F4 != NULL &&
            F5 != NULL &&
            F6 != NULL);

    printf ("Molecule info:\n");
    printf ("  #Atoms\t= %d\n", CInt_getNumAtoms (basis));
    printf ("  #Shells\t= %d\n", CInt_getNumShells (basis));
    printf ("  #Funcs\t= %d\n", CInt_getNumFuncs (basis));
    printf ("  #OccOrb\t= %d\n", CInt_getNumOccOrb (basis));
    printf ("  #threads\t= %d\n", nthreads);
    double *F1_offload = NULL;
    double *F2_offload = NULL;
    double *F3_offload = NULL;
    int ldX1 = rowsize;
    int ldX2 = colsize;
    int ldX3 = colsize;
    int ldX4 = colfuncs;
    int ldX5 = colsize;
    int ldX6 = colfuncs;
    if (toOffload == 1)
    {
        offload_init (nshells, nnz, shellptr, shellid, shellrid,
                      shellvalue, f_startind, rowpos, colpos, rowptr, colptr,
                      D1, D2, D3,
                      VD1, VD2, VD3,
                      &F1_offload, &F2_offload, &F3_offload,
                      ldX1, ldX2, ldX3, ldX4, ldX5, ldX6,
                      sizeD1, sizeD2, sizeD3, sizeD4, sizeD5, sizeD6,
                      &mic_numdevs, &nthreads_mic);
        CInt_offload_createERD (basis, &erd, nthreads, nthreads_mic);
        printf ("  #mic_numdevs = %d\n", mic_numdevs);
        printf ("  #threads_mic = %d\n", nthreads_mic);
        totalFDsize += 1.0 * sizeof (double) *
            (sizeD1 + sizeD2 + sizeD3) * mic_numdevs;
        offfload_get_memsize (&totalFDsize_mic);
    }
    else
    {
        CInt_createERD (basis, &erd, nthreads);
    }
    printf ("  CPU use %.3lf MB\n", totalFDsize / 1024.0 / 1024.0);
    printf ("  MIC use %.3lf MB\n", totalFDsize_mic / 1024.0 / 1024.0);
    
    // init D
    #pragma omp parallel for
    for (int j = 0; j < sizeD1; j++)
    {
        D1[j] = 1.0;
    }
    #pragma omp parallel for
    for (int j = 0; j < sizeD2; j++)
    {
        D2[j] = 1.0;
    }
    #pragma omp parallel for
    for (int j = 0; j < sizeD3; j++)
    {
        D3[j] = 1.0;
    }
    // dummy work
    reset_F (nthreads, F1, F2, F3, F4, F5, F6,
             sizeD1, sizeD2, sizeD3,
             sizeD4, sizeD5, sizeD6);
    if (toOffload)
    {
        offload_reset_F (mic_numdevs);
    }

    gettimeofday (&tv1, NULL);
    /************************************************************/

    // init
    gettimeofday (&tv3, NULL);
    gettimeofday (&tv3, NULL);
    if (toOffload == 1)
    {
        offload_copy_D (mic_numdevs, D1, sizeD1);
        offload_copy_D (mic_numdevs, D2, sizeD2);
        offload_copy_D (mic_numdevs, D3, sizeD3);
    }
    double tolscr = TOLSRC;
    double tolscr2 = tolscr * tolscr;
    // init F
    reset_F (nthreads, F1, F2, F3, F4, F5, F6,
             sizeD1, sizeD2, sizeD3,
             sizeD4, sizeD5, sizeD6);
    if (toOffload)
    {
        offload_reset_F (mic_numdevs);
    }
    gettimeofday (&tv4, NULL);
    timeinit = (tv4.tv_sec - tv3.tv_sec) + (tv4.tv_usec - tv3.tv_usec) / 1e6;
    
    int rowid;
    int colid;
    int sblk_row = rowptr_blk[myrank/npcol];
    int sblk_col = colptr_blk[myrank%npcol];
    int eblk_col = colptr_blk[myrank%npcol + 1] - 1;
    int nblks_col = eblk_col - sblk_col + 1;
    int my_sshellrow = rowptr_sh[myrank/npcol];
    int my_sshellcol = colptr_sh[myrank%npcol];
    
    // computation
    gettimeofday (&tv3, NULL);
    for (int i = 0; i < nbp_p * nbp_p; i++)
    {
        int startM;
        int startP;
        int endM;
        int endP;

        rowid = i/nblks_col;
        colid = i%nblks_col;
        startM = blkrowptr_sh[sblk_row + rowid];
        endM = blkrowptr_sh[sblk_row + rowid + 1] - 1;
        startP = blkcolptr_sh[sblk_col + colid];
        endP = blkcolptr_sh[sblk_col + colid + 1] - 1;

        printf ("Compute task %d: (%d %d) (%d %d)\n",
            i, startM, endM, startP, endP);

        if (toOffload == 1)
        {
            offload_fock_task (mic_numdevs, basis, erd, shellptr,
                               shellvalue, shellid, shellrid,
                               f_startind, rowpos, colpos, rowptr, colptr,
                               tolscr2, my_sshellrow, my_sshellcol,
                               startM, endM, startP, endP,
                               D1, D2, D3, F1, F2, F3, F4, F5, F6,
                               ldX1, ldX2, ldX3, ldX4, ldX5, ldX6,
                               sizeD1, sizeD2, sizeD3, sizeD4, sizeD5, sizeD6,
                               mic_fraction);
        }
        else
        {
            fock_task (basis, erd, shellptr,
                       shellvalue, shellid, shellrid,
                       f_startind, rowpos, colpos, rowptr, colptr,
                       tolscr2, startM, startP,
                       startM, endM, startP, endP,
                       D1, D2, D3, F1, F2, F3, F4, F5, F6,
                       ldX1, ldX2, ldX3, ldX4, ldX5, ldX6,
                       sizeD1, sizeD2, sizeD3,
                       sizeD4, sizeD5, sizeD6);
        }
    }
    gettimeofday (&tv4, NULL);
    timecomp = (tv4.tv_sec - tv3.tv_sec) + (tv4.tv_usec - tv3.tv_usec) / 1e6;

    // reduction
    gettimeofday (&tv3, NULL);
    if (toOffload == 1)
    {
        offload_reduce_mic (mic_numdevs, rowfuncs, colfuncs,
                            rowpos[my_sshellrow],
                            colpos[my_sshellcol],
                            F1_offload, F2_offload, F3_offload,
                            sizeD1, sizeD2, sizeD3);
    }
    reduce_F (nthreads, F1, F2, F3, F4, F5, F6,
              sizeD1, sizeD2, sizeD3,
              sizeD4, sizeD5, sizeD6,
              rowsize, colsize,
              rowfuncs, colfuncs,
              rowpos[my_sshellrow],
              colpos[my_sshellcol],
              ldX3, ldX4, ldX5, ldX6);
    if (toOffload == 1)
    {
        offload_wait_mic (mic_numdevs);
        offload_reduce (mic_numdevs,
                        F1, F2, F3,
                        F1_offload, F2_offload, F3_offload,
                        sizeD1, sizeD2, sizeD3);
    }
    gettimeofday (&tv4, NULL);
    timereduce = (tv4.tv_sec - tv3.tv_sec) + (tv4.tv_usec - tv3.tv_usec) / 1e6;
    /***********************************************************/

    gettimeofday (&tv2, NULL);    
    timepass = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) / 1e6;

    printf ("totally takes %.3lf secs\n", timepass);
    printf ("  reset F takes %.3lf secs\n", timeinit);
    printf ("  compute takes %.3lf secs\n", timecomp);
    printf ("  reduce takes %.3lf secs\n", timereduce);

#if 1
    FILE *fp;
    fp = fopen ("F.dat", "w+");
    assert (fp != NULL);

    for (int j = 0; j < sizeD1; j++)
        fprintf (fp, "%le\n", F1[j]);
    for (int j = 0; j < sizeD2; j++)
        fprintf (fp, "%le\n", F2[j]);
    for (int j = 0; j < sizeD3; j++)
        fprintf (fp, "%le\n", F3[j]);

    fclose (fp);
#endif

    if (toOffload == 1)
    {
        offload_deinit (mic_numdevs, shellptr, shellid, shellrid,
                        shellvalue, f_startind, rowpos, colpos,
                        rowptr, colptr, D1, D2, D3,
                        VD1, VD2, VD3, F1_offload, F2_offload, F3_offload);
        CInt_offload_destroyERD (erd);
        CInt_offload_destroyBasisSet (basis);
    }
    else
    {
        CInt_destroyERD (erd);
        CInt_destroyBasisSet (basis);
    }
    _mm_free (F1);
    _mm_free (F2);
    _mm_free (F3);
    _mm_free (F4);
    _mm_free (F5);
    _mm_free (F6);
    _mm_free (D1);
    _mm_free (D2);
    _mm_free (D3);
    _mm_free (VD1);
    _mm_free (VD2);
    _mm_free (VD3);
    _mm_free (rowpos);
    _mm_free (colpos);
    _mm_free (rowptr);
    _mm_free (colptr);
    _mm_free (shellptr);
    _mm_free (shellid);
    _mm_free (shellvalue);
    _mm_free (shellrid);
    _mm_free (f_startind);

    free (rowptr_f);
    free (colptr_f);
    free (rowptr_sh);
    free (colptr_sh);
    free (rowptr_blk);
    free (colptr_blk);
    free (blkrowptr_sh);
    free (blkcolptr_sh);

    MPI_Finalize ();
    
    return 0;
}
