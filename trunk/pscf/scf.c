#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>
#include <mpi.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>
#include <mkl.h>
#include <mkl_trans.h>
#include <ga.h>
#include <macdecls.h>
#include <sys/time.h>
#ifdef __INTEL_OFFLOAD
#include <offload.h>
#endif
#include <libgen.h>

#include "PFock.h"
#include "CInt.h"
#include "purif.h"
#include "putils.h"


static void usage(char *call)
{
    printf("Usage: %s <basis> <xyz> "
           "<#nodes per row> "
           "<#nodes per column> "
           "<np for purification> "
           "<ntasks> " "<#iters>\n", call);
}


/// compute initial guesses for the density matrix
static void initial_guess(PFock_t pfock, BasisSet_t basis, int ispurif,
                          int rowstart, int rowend,
                          int colstart, int colend,
                          double *D_block, int ldD)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int ga;
    double fzero = 0.0;
    PFock_getMatGAHandle(pfock, PFOCK_MAT_TYPE_D, &ga);
    GA_Fill(ga, &fzero);

    // load initial guess, only process 0
    if (myrank == 0) {
        int num_atoms = CInt_getNumAtoms(basis);
        for (int i = 0; i < num_atoms; i++) {
            double *guess;
            int spos;
            int epos;
            CInt_getInitialGuess(basis, i, &guess, &spos, &epos);
            int ld = epos - spos + 1;
            PFock_putDenMat(pfock, spos, epos, spos, epos, guess, ld);
        }
    }
    NGA_Sync ();

    if (1 == ispurif) {
        PFock_getMat(pfock, PFOCK_MAT_TYPE_D,
                     rowstart, rowend, colstart, colend,
                     D_block, ldD);
    }
}


/// compute Hartree-Fock energy
static double compute_energy(purif_t * purif, double *F_block, double *D_block)
{
    double etmp = 0.0;
    double energy = 0.0;
    double *H_block = purif->H_block;
    int nrows = purif->nrows_purif;
    int ncols = purif->ncols_purif;
    int ldx = purif->ldx;

    if (1 == purif->runpurif) {
        #pragma omp parallel for reduction(+: etmp)
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                F_block[i * ldx + j] += H_block[i * ldx + j];
                etmp += D_block[i * ldx + j] *
                    (H_block[i * ldx + j] + F_block[i * ldx + j]);
            }
        }
    }
    MPI_Allreduce(&etmp, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return energy;
}


/// build a Fock matrix
static void fock_build(PFock_t pfock, BasisSet_t basis,
                       int ispurif, int rowstart, int rowend,
                       int colstart, int colend, int stride,
                       double *D_block, double *F_block)
{
    // put density matrix
    if (1 == ispurif) {
        PFock_putDenMat(pfock, rowstart, rowend,
                        colstart, colend, D_block, stride);
    }
    PFock_commitDenMats (pfock);

    // compute Fock matrix
    PFock_computeFock(pfock, basis);

    // get Fock matrix
    if (1 == ispurif) {
        PFock_getMat(pfock, PFOCK_MAT_TYPE_F,
                     rowstart, rowend, colstart, colend,
                     F_block, stride);
    }
}


/// main for SCF
int main (int argc, char **argv)
{
    // init MPI
    int myrank;
    int nprocs;
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    if (myrank == 0)  {
        printf("MPI level: %d\n", provided);
    }
#if 0
    char hostname[1024];
    gethostname (hostname, 1024);
    printf ("Rank %d of %d running on node %s\n", myrank, nprocs, hostname);
#endif

#ifdef __INTEL_OFFLOAD
    double offload_fraction = 0.0;
    int purif_offload = 0;
    char *offload_str = getenv("GTFOCK_OFFLOAD");
    char *offload_str2 = getenv("PURF_OFFLOAD");
    if (offload_str != NULL) {
        offload_fraction = atof(offload_str);
        assert(offload_fraction <= 1.0);
    }
    if (offload_str2 != NULL) {
        purif_offload = atoi(offload_str2);
        assert(purif_offload == 0 ||
               purif_offload == 1);
    }
    if (myrank == 0 && purif_offload == 1) {
        printf("Purification uses MIC!\n");
    }
#endif /* __INTEL_OFFLOAD */

    // create basis set    
    BasisSet_t basis;
#ifdef __INTEL_OFFLOAD
    if (offload_fraction != 0.0) {
        CInt_offload_createBasisSet(&basis);
    }
    else
#endif
    {
        CInt_createBasisSet(&basis);
    }

    // input parameters and load basis set
    int nprow_fock;
    int npcol_fock;
    int nblks_fock;
    int nprow_purif;
    int nshells;
    int natoms;
    int nfunctions;
    int niters;
    if (myrank == 0) {
        if (argc != 8) {
            usage(argv[0]);
            MPI_Finalize();
            exit(0);
        }
        // init parameters
        nprow_fock = atoi(argv[3]);
        npcol_fock = atoi(argv[4]);
        nprow_purif = atoi(argv[5]);
        nblks_fock = atoi(argv[6]);
        niters = atoi(argv[7]);
        assert(nprow_fock * npcol_fock == nprocs);
        assert(nprow_purif * nprow_purif * nprow_purif  <= nprocs);
        assert(niters > 0);
    #ifdef __INTEL_OFFLOAD
        // load basis set
        if (offload_fraction != 0.0) {
            CInt_offload_loadBasisSet(basis, argv[1], argv[2]);
        }  
        else
    #endif        
        {
            CInt_loadBasisSet(basis, argv[1], argv[2]);
        }
        nshells = CInt_getNumShells(basis);
        natoms = CInt_getNumAtoms(basis);
        nfunctions = CInt_getNumFuncs(basis);
        assert(nprow_fock <= nshells && npcol_fock <= nshells);
        assert(nprow_purif <= nfunctions && nprow_purif <= nfunctions);
        printf("Job information:\n");
        char *fname;
        fname = basename(argv[2]);
        printf("  molecule:  %s\n", fname);
        fname = basename(argv[1]);
        printf("  basisset:  %s\n", fname);
        printf("  #atoms     = %d\n", natoms);
        printf("  #shells    = %d\n", nshells);
        printf("  #functions = %d\n", nfunctions);
        printf("  fock build uses   %d (%dx%d) nodes\n",
               nprow_fock * npcol_fock, nprow_fock, npcol_fock);
        printf("  purification uses %d (%dx%dx%d) nodes\n",
               nprow_purif * nprow_purif * nprow_purif,
               nprow_purif, nprow_purif, nprow_purif);
        printf("  #tasks = %d (%dx%d)\n",
               nblks_fock * nblks_fock * nprow_fock * nprow_fock,
               nblks_fock * nprow_fock, nblks_fock * nprow_fock);
        int nthreads = omp_get_max_threads();
        printf("  #nthreads_cpu = %d\n", nthreads);
    #ifdef __INTEL_OFFLOAD    
        if (offload_fraction != 0.0) {
            int mic_numdevs = _Offload_number_of_devices();
            printf("  #mic_devices = %d\n", mic_numdevs);
            if (mic_numdevs != 0) {
                int nthreads_mic =
                    omp_get_max_threads_target(TARGET_MIC, 0);
                printf("  #nthreads_mic = %d\n", nthreads_mic);
            }
        }
    #endif    
    }
    int btmp[8];
    btmp[0] = nprow_fock;
    btmp[1] = npcol_fock;
    btmp[2] = nprow_purif;
    btmp[3] = nblks_fock;
    btmp[4] = niters;
    btmp[5] = natoms;
    btmp[6] = nshells;
    btmp[7] = nfunctions;
    MPI_Bcast(btmp, 8, MPI_INT, 0, MPI_COMM_WORLD);
    nprow_fock = btmp[0];
    npcol_fock = btmp[1];
    nprow_purif = btmp[2];
    nblks_fock = btmp[3];
    niters = btmp[4];
    natoms = btmp[5];
    nshells = btmp[6];
    nfunctions = btmp[7];

    // broadcast basis set
    void *bsbuf;
    int bsbufsize;
    if (myrank == 0) {
        CInt_packBasisSet(basis, &bsbuf, &bsbufsize);
        MPI_Bcast(&bsbufsize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(bsbuf, bsbufsize, MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast(&bsbufsize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        bsbuf = (void *)malloc(bsbufsize);
        assert(bsbuf != NULL);
        MPI_Bcast(bsbuf, bsbufsize, MPI_CHAR, 0, MPI_COMM_WORLD);
        CInt_unpackBasisSet(basis, bsbuf);
    #ifdef __INTEL_OFFLOAD   
        if (offload_fraction != 0.0) {
            CInt_offload_pushBasisSet(basis);
        }
    #endif    
        free(bsbuf);
    }

    // init PFock
    if (myrank == 0) {
        printf("Initializing pfock ...\n");
    }
    double t1, t2, t3, t4;
    t1 = MPI_Wtime();
    PFock_t pfock;
    PFock_GAInit(nfunctions, nprow_fock, npcol_fock, 1, 0, 0);
#ifdef __INTEL_OFFLOAD
    PFock_create(basis, nprow_fock, npcol_fock, nblks_fock,
                 1e-10, offload_fraction, &pfock);
#else
    PFock_create(basis, nprow_fock, npcol_fock, nblks_fock,
                 1e-10, &pfock);
#endif
    t2 = MPI_Wtime ();
    if (myrank == 0) {
        double mem_cpu;
        double mem_mic;
        double erd_mem_cpu;
        double erd_mem_mic;
        PFock_getMemorySize(pfock, &mem_cpu, &mem_mic,
                            &erd_mem_cpu, &erd_mem_mic);
        printf("  CPU uses %.3lf MB\n"
               "  MIC uses %.3lf MB\n",
               (mem_cpu + erd_mem_cpu) / 1024.0 / 1024.0,
               (mem_mic + erd_mem_mic) / 1024.0 / 1024.0);
        printf("  takes %.3lf secs\n", t2 - t1);
        printf("  Done\n");
    }

    // init purif
#ifdef __INTEL_OFFLOAD    
    purif_t *purif = create_purif(basis, nprow_purif, nprow_purif,
                                  nprow_purif, purif_offload);
#else
    purif_t *purif = create_purif(basis, nprow_purif, nprow_purif, nprow_purif);
#endif
    init_oedmat(basis, pfock, purif, nprow_fock, npcol_fock);

    // compute SCF
    if (myrank == 0) {
        printf("Computing SCF ...\n");
    }
    int rowstart = purif->srow_purif;
    int rowend = purif->nrows_purif + rowstart - 1;
    int colstart = purif->scol_purif;
    int colend = purif->ncols_purif + colstart - 1;
    double energy0 = -1.0;
    double totaltime = 0.0;
    double purif_flops = 2.0 * nfunctions * nfunctions * nfunctions;
    double diis_flops;

    // set initial guess
    if (myrank == 0) {
        printf("  initialing D ...\n");
    }
    initial_guess(pfock, basis, purif->runpurif,
                  rowstart, rowend, colstart, colend,
                  purif->D_block, purif->ldx);

    double ene_nuc = CInt_getNucEnergy(basis);
    if (myrank == 0) {
        printf("  nuc energy = %.8lf\n", ene_nuc);

    }

    MPI_Barrier(MPI_COMM_WORLD);
    // main loop
    for (int iter = 0; iter < niters; iter++) {
        if (myrank == 0) {
            printf("  iter %d\n", iter);
        }
        t3 = MPI_Wtime();

        // fock matrix construction
        t1 = MPI_Wtime();
        fock_build(pfock, basis, purif->runpurif,
                   rowstart, rowend, colstart, colend,
                   purif->ldx, purif->D_block, purif->F_block);
        // compute energy
        double energy = compute_energy(purif, purif->F_block, purif->D_block);
        t2 = MPI_Wtime();
        if (myrank == 0) {
            printf("    fock build takes %.3lf secs\n", t2 - t1);
            if (iter > 0) {
                printf("    energy %.8lf (%.8lf), %le\n",
                       energy + ene_nuc, energy, fabs (energy - energy0));
            }
            else {
                printf("    energy %.8lf (%.8lf)\n", energy + ene_nuc,
                       energy);
            }
        }
        if (iter > 0 && fabs (energy - energy0) < 1e-8) {
            niters = iter + 1;
            break;
        }
        energy0 = energy;

        // compute DIIS
        t1 = MPI_Wtime();
        compute_diis(pfock, purif, purif->D_block, purif->F_block, iter);
        t2 = MPI_Wtime();

        if (myrank == 0) {
            if (iter > 1) {
                diis_flops = purif_flops * 6.0;
            } else {
                diis_flops = purif_flops * 2.0;
            }
            printf("    diis takes %.3lf secs, %.3lf Gflops\n",
                   t2 - t1, diis_flops / (t2 - t1) / 1e9);
        }
        
    #ifdef SCF_OUT
        if (myrank == 0) {
            double outbuf[nfunctions];
            char fname[1024];
            sprintf(fname, "XFX_%d_%d.dat", nfunctions, iter);
            FILE *fp = fopen(fname, "w+");
            assert(fp != NULL);
            for (int i = 0; i < nfunctions; i++) {
                PFock_getMat(pfock, PFOCK_MAT_TYPE_F,
                             i, i, 0, nfunctions - 1,
                             outbuf, nfunctions);
                for (int j = 0; j < nfunctions; j++) {
                    fprintf(fp, "%le\n", outbuf[j]);
                }
            }
            fclose(fp);
        }
    #endif
    
        // purification
        MPI_Barrier(MPI_COMM_WORLD);
        t1 = MPI_Wtime();
        int it = compute_purification(purif, purif->F_block, purif->D_block);
        t2 = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);

#if 0
        int ga_tmp;
        PFock_getMatGAHandle(pfock, PFOCK_MAT_TYPE_D, &ga_tmp);
        compute_eigensolve(ga_tmp, purif, purif->F_block,
                           nprow_fock, npcol_fock);
#endif

        if (myrank == 0) {
            printf("    purification takes %.3lf secs,"
                   " %d iterations, %.3lf Gflops\n",
                   t2 - t1, it,
                   (it * 2.0 + 4.0) * purif_flops / (t2 - t1) / 1e9);
        }

        t4 = MPI_Wtime ();
        totaltime += t4 - t3;

#ifdef SCF_TIMING
        PFock_getStatistics(pfock);
        double purif_timedgemm;
        double purif_timepass;
        double purif_timetr;
        MPI_Reduce(&purif->timedgemm, &purif_timedgemm,
                   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&purif->timepass, &purif_timepass,
                   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&purif->timetr, &purif_timetr,
                   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (myrank == 0) {
            printf("    Purification Statistics:\n");
            printf("      average totaltime = %.3lf\n"
                   "      average timetr    = %.3lf\n"
                   "      average timedgemm = %.3lf, %.3lf Gflops\n",
                   purif_timepass / purif->np_purif,
                   purif_timetr / purif->np_purif,
                   purif_timedgemm / purif->np_purif,
                   (it * 2.0 + 4.0) *
                   purif_flops / (purif_timedgemm / purif->np_purif) / 1e9);
        }
#endif
    } /* for (iter = 0; iter < NITERATIONS; iter++) */

    if (myrank == 0) {
        printf("  totally takes %.3lf secs: %.3lf secs/iters\n",
               totaltime, totaltime / niters);
        printf("  Done\n");
    }

    destroy_purif(purif);
    PFock_destroy(pfock);
    CInt_destroyBasisSet(basis);

    MPI_Finalize();

    return 0;
}
