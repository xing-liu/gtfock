#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "one_electron.h"


inline void matrix_block_write (double *matrix, int startrow,
                                int startcol, int ldm,
                                double *block, int nrows, int ncols)
{
    int i;
    int j;
    int k;
    int l;

    for (k = 0; k < nrows; k++)
    {
        for (l = 0; l < ncols; l++)
        {
            i = startrow + k;
            j = startcol + l;
            matrix[i * ldm + j] = block[k + nrows * l];
        }
    }
}


void compute_S (PFock_t pfock, BasisSet_t basis,
                int startshellrow, int endshellrow,
                int startshellcol, int endshellcol,
                double *S, int ldS)
{
    int nthreads = omp_get_max_threads ();
    OED_t *oed = (OED_t *)malloc (sizeof(OED_t) * nthreads);
    assert (oed != NULL);
    for (int i = 0; i < nthreads; i++)
    {
        CInt_createOED (basis, &(oed[i]));
    }
    int start_row_id = pfock->f_startind[startshellrow];
    int start_col_id = pfock->f_startind[startshellcol];

    #pragma omp parallel
    {
        int tid = omp_get_thread_num ();
        #pragma omp for
        for (int A = startshellrow; A <= endshellrow; A++)
        {
            int row_id_1 = pfock->f_startind[A];
            int row_id_2 = pfock->f_startind[A + 1] - 1;
            int startrow = row_id_1 - start_row_id;
            int nrows = row_id_2 - row_id_1 + 1;
            for (int B = startshellcol; B <= endshellcol; B++)
            {
                int col_id_1 = pfock->f_startind[B];
                int col_id_2 = pfock->f_startind[B + 1] - 1;
                int startcol = col_id_1 - start_col_id;
                int ncols = col_id_2 - col_id_1 + 1;
                int nints;
                double *integrals;
                CInt_computePairOvl (basis, oed[tid], A, B, &integrals, &nints);
                if (nints != 0)
                {
                    matrix_block_write (S, startrow, startcol, ldS,
                                        integrals, nrows, ncols);
                }
            }
        }
    }

    for (int i = 0; i < nthreads; i++)
    {
        CInt_destroyOED (oed[i]);
    }
    free (oed);
}


void compute_H (PFock_t pfock, BasisSet_t basis,
                int startshellrow, int endshellrow,
                int startshellcol, int endshellcol,
                double *H, int ldH)
{

    int nthreads = omp_get_max_threads ();
    OED_t *oed = (OED_t *)malloc (sizeof(OED_t) * nthreads);
    assert (oed != NULL);
    for (int i = 0; i < nthreads; i++)
    {
        CInt_createOED (basis, &(oed[i]));
    }
    
    int start_row_id = pfock->f_startind[startshellrow];
    int start_col_id = pfock->f_startind[startshellcol];
    #pragma omp parallel
    {
        int tid = omp_get_thread_num ();
        #pragma omp for
        for (int A = startshellrow; A <= endshellrow; A++)
        {
            int row_id_1 = pfock->f_startind[A];
            int row_id_2 = pfock->f_startind[A + 1] - 1;
            int startrow = row_id_1 - start_row_id;
            int nrows = row_id_2 - row_id_1 + 1;
            for (int B = startshellcol; B <= endshellcol; B++)
            {
                int col_id_1 = pfock->f_startind[B];
                int col_id_2 = pfock->f_startind[B + 1] - 1;
                int startcol = col_id_1 - start_col_id;
                int ncols = col_id_2 - col_id_1 + 1;
                int nints;
                double *integrals;
                CInt_computePairCoreH (basis, oed[tid], A, B, &integrals, &nints);
                if (nints != 0)
                {
                    matrix_block_write (H, startrow, startcol, ldH,
                                        integrals, nrows, ncols);
                }
            }
        }
    }

    for (int i = 0; i < nthreads; i++)
    {
        CInt_destroyOED (oed[i]);
    }
    free (oed);
}
