#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <omp.h>

#include "CInt.h"


typedef struct _atom_t
{
    char name[4];
    double x;
    double y;
    double z;
} atom_t;


#define TOLSRC 1e-10


static void atom_screening (BasisSet_t basis, int **atomptrOut,
                            int **atomidOut, int **atomridOut,
                            double **atomvalueOut, int *nnzOut)
{
    int nthreads;

    nthreads = omp_get_max_threads ();
    ERD_t erd;
    CInt_createERD (basis, &erd, nthreads);
    const int natoms = CInt_getNumAtoms (basis);

    double *vpairs = (double *) malloc (sizeof (double) * natoms * natoms);
    assert (vpairs != NULL);

    double allmax = 0.0;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num ();
        #pragma omp for reduction(max:allmax)
        for (int i = 0; i < natoms; i++)
        {
            const int start_sh0 = CInt_getAtomStartInd (basis, i);
            const int end_sh0 = CInt_getAtomStartInd (basis, i + 1);
            double mvalue;
            int j;
            for (int M = start_sh0; M < end_sh0; M++)
            {
                const int dimM = CInt_getShellDim (basis, M);
                for (j = 0; j < natoms; j++)
                {
                    const int start_sh1 = CInt_getAtomStartInd (basis, j);
                    const int end_sh1 = CInt_getAtomStartInd (basis, j + 1);
                    mvalue = 0.0;
                    for (int N = start_sh1; N < end_sh1; N++)
                    {
                        const int dimN = CInt_getShellDim (basis, N);
                        int nints;
                        double *integrals;

                        CInt_computeShellQuartet (basis, erd, tid, M, N, M, N, &integrals,
                                                  &nints);                       
                        if (nints != 0)
                        {
                            for (int iM = 0; iM < dimM; iM++)
                            {
                                for (int iN = 0; iN < dimN; iN++)
                                {
                                    const int index = iM * (dimN * dimM * dimN + dimN) +
                                        iN * (dimM * dimN + 1);
                                    if (mvalue < fabs (integrals[index]))
                                    {
                                        mvalue = fabs (integrals[index]);
                                    }
                                }
                            }
                        }
                    }
                    if (mvalue > allmax)
                    {
                        allmax = mvalue;
                    }
                    vpairs[i * natoms + j] = mvalue;
                } /* for j */              
            }
        } /* for i */      
    } /* #pragma omp parallel */

    // init atomptr
    int nnz = 0;
    const double eta = TOLSRC * TOLSRC / allmax;
    int *atomptr = (int *) _mm_malloc (sizeof (int) * (natoms + 1), 64);
    assert (atomptr != NULL);
    memset (atomptr, 0, sizeof (int) * (natoms + 1));

    for (int i = 0; i < natoms; i++)
    {
        for (int j = 0; j < natoms; j++)
        {
            double mvalue = vpairs[i * natoms + j];
            if (mvalue > eta)
            {
                nnz++;
            }
        }
        atomptr[i + 1] = nnz;
    }

    double *atomvalue = (double *) _mm_malloc (sizeof (double) * nnz, 64);
    int *atomid = (int *) _mm_malloc (sizeof (int) * nnz, 64);
    int *atomrid = (int *) _mm_malloc (sizeof (int) * nnz, 64);
    assert ((atomvalue != NULL) && (atomid != NULL) && (atomrid != NULL));
    nnz = 0;
    for (int i = 0; i < natoms; i++)
    {
        for (int j = 0; j < natoms; j++)
        {
            const double mvalue = vpairs[i * natoms + j];
            if (mvalue > eta)
            {
                atomvalue[nnz] = mvalue;                       
                atomid[nnz] = j;
                atomrid[nnz] = i;
                nnz++;
            }
        }
    }
    
    *nnzOut = nnz;
    free (vpairs);
    CInt_destroyERD (erd);

    *atomidOut = atomid;
    *atomridOut = atomrid;
    *atomptrOut = atomptr;
    *atomvalueOut = atomvalue;
}


#if 0
static int compare_x (const void *a, const void *b)
{
    atom_t *aa = (atom_t *)a;
    atom_t *bb = (atom_t *)b;
    if ((aa->x - bb->x) > 0.0)
        return 1;
    else
        return 0;
}


static int compare_y (const void *a, const void *b)
{
    atom_t *aa = (atom_t *)a;
    atom_t *bb = (atom_t *)b;
    if ((aa->y - bb->y) > 0.0)
        return 1;
    else
        return 0;
}


static int compare_z (const void *a, const void *b)
{
    atom_t *aa = (atom_t *)a;
    atom_t *bb = (atom_t *)b;
    if ((aa->z - bb->z) > 0.0)
        return 1;
    else
        return 0;
}
#endif

static void write_xyz (char *file, int natoms, atom_t *atoms)
{
    int i;
    FILE *fp;

    fp = fopen (file, "w+");
    assert (fp != NULL);

    fprintf (fp, "%d\n", natoms);    
    fprintf (fp, "\n");
    
    for (i = 0; i < natoms; i++)
    {
        fprintf (fp, "%s %24.16f %24.16f %24.16f\n",
                 atoms[i].name, atoms[i].x, atoms[i].y, atoms[i].z);
    }
    
    fclose (fp);
}


static void read_xyz (char *file, int *_natoms, atom_t **_atoms)
{
    FILE *fp;
    char line[1024];
    int nsc;
    atom_t *atoms;
    int natoms;
    int count = 0;

    fp = fopen (file, "r");
    assert (fp != NULL);

    // number of atoms    
    assert (fgets (line, 1024, fp) != NULL);    
    sscanf (line, "%d", &natoms);    
    assert (natoms > 0);
        
    // skip comment line
    assert (fgets (line, 1024, fp) != NULL);
    atoms = (atom_t *)malloc (sizeof(atom_t) * natoms);
    assert (atoms != NULL);

    // read x, y and z
    count = 0;
    while (fgets (line, 1024, fp) != NULL)
    {
        nsc = sscanf (line, "%s %lf %lf %lf",
                      &(atoms[count].name[0]), &(atoms[count].x), 
                      &(atoms[count].y), &(atoms[count].z));
        count++;
    }
    assert (count == natoms);

    *_natoms = natoms;
    *_atoms = atoms;
    
    fclose (fp);
}


int main (int argc, char **argv)
{
    atom_t *atoms;
    int natoms;
    BasisSet_t basis;
    int *atomptr;
    int *atomid;
    int *atomrid;
    double *atomvalue;
    int nnz;
    double max_dist;
    double min_dist;

    if (argc != 3)
    {
        printf ("Usage: %s <basis set> <xyz file>\n", argv[0]);
        return -1;
    }
    
    printf ("read %s\n", argv[2]);
    read_xyz (argv[2], &natoms, &atoms);
    CInt_createBasisSet (&basis);
    CInt_loadBasisSet (basis, argv[1], argv[2]);   

    printf ("Molecule info:\n");
    printf ("  #Atoms\t= %d\n", CInt_getNumAtoms (basis));
    printf ("  #Shells\t= %d\n", CInt_getNumShells (basis));
    printf ("  #Funcs\t= %d\n", CInt_getNumFuncs (basis));
    printf ("  #OccOrb\t= %d\n", CInt_getNumOccOrb (basis));

    double max_x = atoms[0].x;
    double min_x = max_x;
    double max_y = atoms[0].y;
    double min_y = max_y;
    double max_z = atoms[0].z;
    double min_z = max_z;
    int i;
    for (i = 0; i < natoms; i++)
    {
        max_x = max_x > atoms[i].x ? max_x : atoms[i].x;
        max_y = max_y > atoms[i].y ? max_y : atoms[i].y;
        max_z = max_z > atoms[i].z ? max_z : atoms[i].z;
        min_x = min_x < atoms[i].x ? min_x : atoms[i].x;
        min_y = min_y < atoms[i].y ? min_y : atoms[i].y;
        min_z = min_z < atoms[i].z ? min_z : atoms[i].z;
    }    
    printf ("X (%.4lf, %.4lf), Y (%.4lf, %.4lf), Z (%.4lf, %.4lf)\n",
        min_x, max_x, min_y, max_y, min_z, max_z);

    atom_screening (basis, &atomptr, &atomid, &atomrid,
                    &atomvalue, &nnz);
    max_dist = 0.0;
    min_dist = sqrt((max_x-min_x)*(max_x-min_x) +
                    (max_y-min_y)*(max_y-min_y) +
                    (max_z-min_z)*(max_z-min_z));
    FILE *fp;
    fp = fopen ("map.dat", "w+");
    assert (fp != NULL);
    for (i = 0; i < nnz; i++)
    {
        double x1;
        double y1;
        double z1;
        double x2;
        double y2;
        double z2;
        int id1;
        int id2;
        id1 = atomid[i];
        id2 = atomrid[i];
        fprintf (fp, "%d %d %lf\n", id1 + 1, id2 + 1, atomvalue[i]);
        x1 = atoms[id1].x;
        y1 = atoms[id1].y;
        z1 = atoms[id1].z;
        x2 = atoms[id2].x;
        y2 = atoms[id2].y;
        z2 = atoms[id2].z;
           
        double dist = sqrt((x1-x2)*(x1-x2) +
                      (y1-y2)*(y1-y2) +
                      (z1-z2)*(z1-z2));
        max_dist = max_dist > dist ? max_dist : dist;
        if (dist != 0.0)
        {
            min_dist = min_dist < dist ? min_dist : dist;
        }
    }
    fclose (fp);
    printf ("max interacting distance %.4lf\n", max_dist);
    printf ("min distance %.4lf\n", min_dist);
    
    printf ("sorting molecules ...\n");

#if 0
    qsort (atoms, natoms, sizeof(atom_t), compare_x);   
    int k = 0;
    int j;
    int startx;
    int starty;
    double x0;
    double y0;
    startx = 0;    
    x0 = min_dist;
    while (min_x + x0 -  min_dist <= max_x)
    {
        while (k < natoms)
        {
            if (atoms[k].x > min_x + x0)
                break;
            k++;
        }
        printf ("sort for %lf %lf, %d %d\n",
            min_x + x0 - min_dist, min_x + x0, startx, k);
        qsort(&atoms[startx], k - startx, sizeof(atom_t), compare_y);

        j = startx;
        y0 = min_dist;
        starty = startx;
        while (min_y + y0 -  min_dist <= max_y)
        {
            while (j < k)
            {
                if (atoms[j].y > min_y + y0)
                    break;
                j++;
            }
            printf ("  sort for %lf %lf, %d %d\n",
                min_y + y0 - min_dist, min_y + y0, starty, j);
            qsort(&atoms[starty], j - starty, sizeof(atom_t), compare_z);
            starty = j;
            y0 += min_dist;
        }
        startx = k;
        x0 += min_dist;
    }
    
    write_xyz ("new.xyz", natoms, atoms);
    printf ("write new.xyz\n");
    free (atoms);
#endif

    CInt_destroyBasisSet (basis);
    _mm_free (atomvalue);
    _mm_free (atomid);
    _mm_free (atomrid);
    _mm_free (atomptr);
    
    return 0;
}
