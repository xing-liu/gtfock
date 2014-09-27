#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>
#include <mkl.h>
#include <string.h>
#include <mkl_scalapack.h>
#include <ga.h>
#include <math.h>

#include "putils.h"


// ring broadcast (pipelined)
static void ring_bcast (double *buf, int count, int root, MPI_Comm comm)
{
    int me;
    int np;

    MPI_Comm_rank (comm, &me);
    MPI_Comm_size (comm, &np);
    if (me != root)
    {
        MPI_Recv (buf, count, MPI_DOUBLE, (me - 1 + np) % np,
                  MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
    }
    if ((me + 1) % np != root)
    {
        MPI_Send (buf, count, MPI_DOUBLE, (me + 1) % np, 0, comm);
    }
}


void my_pdgemm (int n, int nb,
                double *A, double *B, double *C,
                int nrows, int ncols, int ldx,
                int *nr, int *nc,
                MPI_Comm comm_row, MPI_Comm comm_col,
                double *work1, double *work2, int ldw1)
{
    int myrank_row;
    int myrank_col;
    int myrow;
    int mycol;
    int kk;
    int iwrk;
    int icurrow;
    int icurcol;
    int ii;
    int jj;

    // get row and col communicator
    MPI_Comm_rank (comm_row, &myrank_row);
    MPI_Comm_rank (comm_col, &myrank_col);
    myrow = myrank_col;
    mycol = myrank_row;

    // zeros C
    memset (C, 0, sizeof (double) * nrows * ldx);

    icurrow = 0;
    icurcol = 0;
    ii = jj = 0;
    iwrk = 0;
    // main loop
    for (kk = 0; kk < n; kk += iwrk)
    {
        iwrk = MIN (nb, nr[icurrow] - ii);
        iwrk = MIN (iwrk, nc[icurcol] - jj);
        if (mycol == icurcol)
        {
            dlacpy_ ("General", &iwrk, &nrows, &A[jj], &ldx, work1, &ldw1);
        }
        if (myrow == icurrow)
        {
            dlacpy_ ("General", &ncols, &iwrk, &B[ii * ldx], &ldx, work2,
                     &ldx);
        }
        ring_bcast (work1, nrows * ldw1, icurcol, comm_row);
        ring_bcast (work2, ldx * iwrk, icurrow, comm_col);
        cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, nrows, ncols,
                     iwrk, 1.0, work1, ldw1, work2, ldx, 1.0, C, ldx);
        ii += iwrk;
        jj += iwrk;
        if (jj >= nc[icurcol])
        {
            icurcol++;
            jj = 0;
        }
        if (ii >= nr[icurrow])
        {
            icurrow++;
            ii = 0;
        }
    }
}

/* Jeff: FYI there is a GA wrapper to PDSYEV already called ga_pdsyev.
 *       See global/src/scalapack.F for details. */


void my_peig (int ga_A, int ga_B, int n, int nprow, int npcol, double *eval)
{
    int myrank;
    int ictxt;
    int myrow;
    int mycol;
    int nn;
    int mm;
    int nrows;
    int ncols;
    int nb;
    int izero = 0;
    int descA[9];
    int descZ[9];
    int info;
    int itemp;
    int i;
    int blocksize;
    double *A;
    double *Z;
    double *work;
    int lo[2];
    int hi[2];
    int ld;
    int lwork;
    int ione = 1;
    int j;
#ifdef GA_NB
    ga_nbhdl_t nbnb;
#endif

    // init blacs
    nb = MIN (n / nprow, n / npcol);
    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
    Cblacs_pinfo (&nn, &mm);
    Cblacs_get (-1, 0, &ictxt);
    Cblacs_gridinit (&ictxt, "Row", nprow, npcol);
    Cblacs_gridinfo (ictxt, &nprow, &npcol, &myrow, &mycol);

    // init matrices
    nrows = numroc_ (&n, &nb, &myrow, &izero, &nprow);
    ncols = numroc_ (&n, &nb, &mycol, &izero, &npcol);
    itemp = nrows > 1 ? nrows : 1;
    descinit_ (descA, &n, &n, &nb, &nb, &izero, &izero, &ictxt, &itemp,
               &info);
    descinit_ (descZ, &n, &n, &nb, &nb, &izero, &izero, &ictxt, &itemp,
               &info);
    blocksize = nrows * ncols;
    A = (double *) mkl_malloc (blocksize * sizeof (double), 64);
    assert (A != NULL);
    Z = (double *) mkl_malloc (blocksize * sizeof (double), 64);
    assert (Z != NULL);

    // distribute source matrix
    for (i = 1; i <= nrows; i += nb)
    {
        lo[0] = indxl2g_ (&i, &nb, &myrow, &izero, &nprow) - 1;
        hi[0] = lo[0] + nb - 1;
        hi[0] = hi[0] >= n ? n - 1 : hi[0];
        for (j = 1; j <= ncols; j += nb)
        {
            lo[1] = indxl2g_ (&j, &nb, &mycol, &izero, &npcol) - 1;
            hi[1] = lo[1] + nb - 1;
            hi[1] = hi[1] >= n ? n - 1 : hi[1];
            ld = ncols;
#ifdef GA_NB
            NGA_NbGet (ga_A, lo, hi, &(Z[(i - 1) * ncols + j - 1]), &ld,
                       &nbnb);
#else
            NGA_Get (ga_A, lo, hi, &(Z[(i - 1) * ncols + j - 1]), &ld);
#endif
        }
        /* Jeff: Location of NGA_NbWait for flow-control. */
    }
#ifdef GA_NB
    /* Jeff: If one sees flow-control problems with too many
     *       outstanding NbGet operations, then move this call
     *       to the location noted above. */
    NGA_NbWait (&nbnb);
#endif
    for (i = 0; i < nrows; i++)
    {
        for (j = 0; j < ncols; j++)
        {
            A[j * nrows + i] = Z[i * ncols + j];
        }
    }

    double t1 = MPI_Wtime ();

    // inquire working space
    work = (double *) mkl_malloc (2 * sizeof (double), 64);
    assert (work != NULL);
    lwork = -1;
#if 0
    pdsyev ("V", "U", &n, A, &ione, &ione, descA,
            eval, Z, &ione, &ione, descZ, work, &lwork, &info);
#else
    int liwork = -1;
    int *iwork = (int *)mkl_malloc(2 * sizeof (int), 64);
    assert(iwork != NULL);
    pdsyevd("V", "U", &n, A, &ione, &ione, descA,
            eval, Z, &ione, &ione, descZ,
            work, &lwork, iwork, &liwork, &info);    
#endif

    // compute eigenvalues and eigenvectors
    lwork = (int)work[0] * 2;
    mkl_free(work);
    work = (double *)mkl_malloc(lwork * sizeof (double), 64);
    assert(work != NULL);
#if 0
    pdsyev ("V", "U", &n, A, &ione, &ione, descA,
            eval, Z, &ione, &ione, descZ, work, &lwork, &info);
#else
    liwork = (int)iwork[0];
    mkl_free(iwork);
    iwork = (int *)mkl_malloc(liwork * sizeof (int), 64);
    assert(iwork != NULL);
    pdsyevd("V", "U", &n, A, &ione, &ione, descA,
            eval, Z, &ione, &ione, descZ,
            work, &lwork, iwork, &liwork, &info); 
#endif

    double t2 = MPI_Wtime ();
    if (myrank == 0)
    {
        printf ("  pdsyev_ takes %.3lf secs\n", t2 - t1);
    }

    // store desination matrix
    for (i = 0; i < nrows; i++)
    {
        for (j = 0; j < ncols; j++)
        {
            A[i * ncols + j] = Z[j * nrows + i];
        }
    }
    for (i = 1; i <= nrows; i += nb)
    {
        lo[0] = indxl2g_ (&i, &nb, &myrow, &izero, &nprow) - 1;
        hi[0] = lo[0] + nb - 1;
        hi[0] = hi[0] >= n ? n - 1 : hi[0];
        for (j = 1; j <= ncols; j += nb)
        {
            lo[1] = indxl2g_ (&j, &nb, &mycol, &izero, &npcol) - 1;
            hi[1] = lo[1] + nb - 1;
            hi[1] = hi[1] >= n ? n - 1 : hi[1];
            ld = ncols;
#ifdef GA_NB
            NGA_NbPut (ga_B, lo, hi, &(A[(i - 1) * ncols + j - 1]), &ld,
                       &nbnb);
#else
            NGA_Put (ga_B, lo, hi, &(A[(i - 1) * ncols + j - 1]), &ld);
#endif
        }
        /* Jeff: Location of NGA_NbWait for flow-control. */
    }
#ifdef GA_NB
    /* Jeff: If one sees flow-control problems with too many
     *       outstanding NbPut operations, then move this call
     *       to the location noted above. */
    NGA_NbWait (&nbnb);
#endif
    GA_Sync ();

    mkl_free (A);
    mkl_free (Z);
    mkl_free (work);

    Cblacs_gridexit (ictxt);
}

void init_oedmat (BasisSet_t basis, PFock_t pfock, purif_t * purif, int nprow,
                  int npcol)
{
    int srow_purif;
    int scol_purif;
    int nrows_purif;
    int ncols_purif;
    int erow_purif;
    int ecol_purif;
    int nfuncs_row;
    int nfuncs_col;
    int lo[2];
    int hi[2];
    int ld;
    double *eval;
    int nbf;
    int myrank;
    double *blocktmp;
    double *blockS;
    int ga_S;
    int ga_X;
    int ga_tmp;
    CoreH_t hmat;
    Ovl_t smat;
    double t1;
    double t2;
    int ldx = purif->ldx;

    MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
    if (myrank == 0)
    {
        printf ("Preprocessing one electron matrices ...\n");
    }
    srow_purif = purif->srow_purif;
    scol_purif = purif->scol_purif;
    nrows_purif = purif->nrows_purif;
    ncols_purif = purif->ncols_purif;
    erow_purif = srow_purif + nrows_purif - 1;
    ecol_purif = scol_purif + ncols_purif - 1;

    /* Compute X */
    if (myrank == 0)
    {
        printf ("  computing X\n");
    }
    // compute S
    PFock_createOvlMat (pfock, basis, &smat);
    PFock_getOvlMatGAHandle (smat, &ga_S);
    if (purif->runpurif == 1)
    {
        lo[0] = srow_purif;
        hi[0] = erow_purif;
        lo[1] = scol_purif;
        hi[1] = ecol_purif;
        ld = ldx;
        NGA_Get (ga_S, lo, hi, purif->S_block, &ld);
    }
    PFock_getMatGAHandle (pfock, PFOCK_MAT_TYPE_D, &ga_tmp);
    PFock_getMatGAHandle (pfock, PFOCK_MAT_TYPE_F, &ga_X);
    // eigensolve         
    nbf = CInt_getNumFuncs (basis);
    eval = (double *) mkl_malloc (nbf * sizeof (double), 64);
    assert (eval != NULL);

    t1 = MPI_Wtime ();
    //GA_Diag_std (ga_S, ga_tmp, eval);
    my_peig (ga_S, ga_tmp, nbf, nprow, npcol, eval);

    t2 = MPI_Wtime ();
    if (myrank == 0)
    {
        printf ("  my_peig(): %.3lf secs\n", t2 - t1);
    }

    NGA_Distribution (ga_tmp, myrank, lo, hi);
    nfuncs_row = hi[0] - lo[0] + 1;
    nfuncs_col = hi[1] - lo[1] + 1;
    NGA_Access (ga_tmp, lo, hi, &blocktmp, &ld);
    NGA_Access (ga_S, lo, hi, &blockS, &ld);

    /* Jeff: I hoisted the lambda evaluation out for two reasons.
     *       First, I think it will vectorize better if it is on its own.
     *       Second, it makes contiguous memory access in the double loops trivial. */

    /* Jeff: I try to remove all indirection when I want the compiler to do SIMD. */
    int lo1tmp = lo[1];

    double *lambda_vector = malloc (nfuncs_col * sizeof (double));
    assert (lambda_vector != NULL);

    #pragma omp parallel for
    #pragma simd
    #pragma ivdep
    for (int j = 0; j < nfuncs_col; j++)
    {
        lambda_vector[j] = 1.0 / sqrt (eval[j + lo1tmp]);
    }
    /* Jeff: Am I crazy for thinking these loops are just DGEMV? */
    #pragma omp parallel for
    for (int i = 0; i < nfuncs_row; i++)
    {
        #pragma simd
        #pragma ivdep
        for (int j = 0; j < nfuncs_col; j++)
        {
            blockS[i * nfuncs_col + j] =
                blocktmp[i * nfuncs_col + j] * lambda_vector[j];
        }
    }

    free (lambda_vector);

    NGA_Release (ga_tmp, lo, hi);
    NGA_Release_update (ga_S, lo, hi);

    // get X
    GA_Dgemm ('N', 'T', nbf, nbf, nbf, 1.0, ga_S, ga_tmp, 0.0, ga_X);
    t1 = MPI_Wtime ();
    if (myrank == 0)
    {
        printf ("  GA_Dgemm(): %.3lf secs\n", t1 - t2);
    }

    if (purif->runpurif == 1)
    {
        lo[0] = srow_purif;
        hi[0] = erow_purif;
        lo[1] = scol_purif;
        hi[1] = ecol_purif;
        ld = ldx;
        NGA_Get (ga_X, lo, hi, purif->X_block, &ld);
    }
    PFock_destroyOvlMat (smat);
    mkl_free (eval);

    t1 = MPI_Wtime ();
    /* Compute H */
    if (myrank == 0)
    {
        printf ("  computing H\n");
    }
    PFock_createCoreHMat (pfock, basis, &hmat);
    if (purif->runpurif == 1)
    {
        PFock_getCoreHMat (hmat, srow_purif, erow_purif,
                           scol_purif, ecol_purif, purif->H_block, ldx);
    }
    PFock_destroyCoreHMat (hmat);

    t2 = MPI_Wtime ();
    if (myrank == 0)
    {
        printf ("  takes %.3lf secs\n", t2 - t1);
        printf ("  Done\n");
    }
}


void print_dmat (char *name, double *A, int nrows, int ncols, int ldA)
{
    printf ("%s\n", name);
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            printf ("%lf ", A[i * ldA + j]);
        }
        printf ("\n");
    }
}
