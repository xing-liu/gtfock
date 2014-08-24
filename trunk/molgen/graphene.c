#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>


typedef struct _atom_t
{
    int idx;
    char name;
    double x;
    double y;
    double z;
} atom_t;


typedef struct _edge_t
{
    atom_t *atom;
    double phi;
    int type;
} edge_t;


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
    if ((aa->y - bb->y) < 0.0)
        return 1;
    else
        return 0;
}


static void output_nwchem (int natoms, atom_t *atoms, int nc)
{
    char str[1000];
    int i;
    FILE *fp;
    
    sprintf (str, "./data/graphene_%d.nw", natoms);

    fp = fopen (str, "w+");

    sprintf (str, "graphene_%d", natoms);
    
    fprintf (fp, "title %s\n", str);

    fprintf (fp, "print medium \"screening statistics\"\n");
    fprintf (fp, "set scf:variable logical .true.\n");
    fprintf (fp, "set fock:replicated logical .false.\n");

    fprintf (fp, "geometry units an\n");
    for (i = 0; i < nc; i++)
    {
        if (atoms[i].idx != -1)
        {
            fprintf (fp, "%c %24.16f %24.16f %24.16f\n",
                     atoms[i].name, atoms[i].x, atoms[i].y, atoms[i].z);
        }
    }
    fprintf (fp, "end\n");

    fprintf (fp, "basis\n");
    fprintf (fp, "    H library cc-pVDZ\n");
    fprintf (fp, "    C library cc-pVDZ\n");
    fprintf (fp, "end\n");

    fprintf (fp, "scf\n");
    fprintf (fp, "    direct\n");
    fprintf (fp, "    sym off\n");
    fprintf (fp, "    tol2e 1e-10\n");
    fprintf (fp, "    profile\n");
    fprintf (fp, "    maxiter 1\n");
    fprintf (fp, "    thresh 100.0\n");
    fprintf (fp, "end\n");
    fprintf (fp, "task scf\n");


    fclose (fp);
}


static void output_screening (int natoms, atom_t *atoms, int nc)
{
    char str[1000];
    int i;
    FILE *fp;
    
    sprintf (str, "./data/graphene_%d_screening.dat", natoms);

    fp = fopen (str, "w+");

    fprintf (fp, "molecule {\n\n");
    for (i = 0; i < nc; i++)
    {
        if (atoms[i].idx != -1)
        {
            fprintf (fp, "%c %24.16f %24.16f %24.16f\n",
                     atoms[i].name, atoms[i].x, atoms[i].y, atoms[i].z);
        }
    }
    fprintf (fp, "\n}\n");

    fprintf (fp, "memory 4000 mb\n\n");

    fprintf (fp, "plugin_load(\"./screening.so\")\n\n");
    
    fprintf (fp, "set {\n");  
    fprintf (fp, "    basis cc-pVDZ\n");
    fprintf (fp, "}\n\n");

    fprintf (fp, "set screening {\n");
    fprintf (fp, "    print 1\n");
    fprintf (fp, "}\n\n");

    fprintf (fp, "plugin(\"./screening.so\")\n");

    fclose (fp);
}


static void output_config (int natoms, atom_t *atoms, int nc)
{
    char str[1000];
    int i;
    FILE *fp;
    
    sprintf (str, "./data/graphene_%d_config.dat", natoms);

    fp = fopen (str, "w+");

    fprintf (fp, "molecule {\n\n");
    for (i = 0; i < nc; i++)
    {
        if (atoms[i].idx != -1)
        {
            fprintf (fp, "%c %24.16f %24.16f %24.16f\n",
                     atoms[i].name, atoms[i].x, atoms[i].y, atoms[i].z);
        }
    }
    fprintf (fp, "\n}\n");

    fprintf (fp, "memory 4000 mb\n\n");

    fprintf (fp, "plugin_load(\"./test_plugin.so\")\n\n");
    
    fprintf (fp, "set {\n");  
    fprintf (fp, "    basis cc-pVDZ\n");
    fprintf (fp, "    scf_type pk\n");
 
    fprintf (fp, "}\n\n");

    fprintf (fp, "set test_plugin {\n");
    fprintf (fp, "    print 1\n");
    fprintf (fp, "}\n\n");

    fprintf (fp, "plugin(\"./test_plugin.so\")\n");

    fclose (fp);
}
#endif


static void output_xyz (int natoms, atom_t *atoms, int nc)
{
    char str[1000];
    int i;
    FILE *fp;
    sprintf (str, "./graphene_%d.xyz", natoms);

    fp = fopen (str, "w+");
    assert (fp != NULL);

    sprintf (str, "graphene_%d", natoms);

    fprintf (fp, "%d\n", natoms);
    
    fprintf (fp, "  title %s\n", str);
    for (i = 0; i < nc; i++)
    {
        if (atoms[i].idx != -1)
        {
            fprintf (fp, "%c %24.16f %24.16f %24.16f\n",
                     atoms[i].name, atoms[i].x, atoms[i].y, atoms[i].z);
        }
    }
    
    fclose (fp);
}


int main (int argc, char **argv)
{
    int N;
    int nb;
    double RCC;
    double RCH;
    double TH;
    atom_t *atoms;
    edge_t *edges;
    edge_t *edges2;
    int natoms;
    int nedges;
    int nc;
    int nh;
    int k;
    int i;
    
    if (argc < 2)
    {
        printf ("Need to specify N\n");
        exit (-1);
    }
    N = atoi (argv[1]);
    assert (N > 0 && N <= 25);

    nb = 1;
    RCC = 1.4;
    RCH = 1.1;
    if (argc > 2)
    {
        RCC = atof (argv[2]);
        RCH = atof (argv[3]);

    }
    assert (RCC > 0);
    assert (RCH > 0);

    natoms = 6 * N * N;
    nedges = 6 * N;
    printf ("Generating Graphene with %d Cabons %d Hydrogens\n",
            natoms, nedges);
    natoms += nedges;

    TH = M_PI / 3.0;
    nc = nh = 6;
    // initialization
    atoms = (atom_t *) malloc (sizeof (atom_t) * natoms);
    assert (atoms != NULL);
    atoms[0].name = 'C';
    atoms[0].x = RCC;
    atoms[0].y = 0.0;
    atoms[0].z = 0.0;
    atoms[0].idx = 0;
    atoms[1].name = 'C';
    atoms[1].x = -RCC;
    atoms[1].y = 0.0;
    atoms[1].z = 0.0;
    atoms[1].idx = 1;
    atoms[2].name = 'C';
    atoms[2].x = RCC * cos (TH);
    atoms[2].y = RCC * sin (TH);
    atoms[2].z = 0.0;
    atoms[2].idx = 2;
    atoms[3].name = 'C';
    atoms[3].x = -RCC * cos (TH);
    atoms[3].y = RCC * sin (TH);
    atoms[3].z = 0.0;
    atoms[3].idx = 3;
    atoms[4].name = 'C';
    atoms[4].x = RCC * cos (TH);
    atoms[4].y = -RCC * sin (TH);
    atoms[4].z = 0.0;
    atoms[4].idx = 4;
    atoms[5].name = 'C';
    atoms[5].x = -RCC * cos (TH);
    atoms[5].y = -RCC * sin (TH);
    atoms[5].z = 0.0;
    atoms[5].idx = 5;
    edges = (edge_t *) malloc (sizeof (edge_t) * nedges);
    assert (edges != NULL);
    edges[0].atom = &atoms[0];
    edges[0].phi = 0.0 * M_PI / 3.0;
    edges[0].type = 1;
    edges[1].atom = &atoms[1];
    edges[1].phi = 3.0 * M_PI / 3.0;
    edges[1].type = 1;
    edges[2].atom = &atoms[2];
    edges[2].phi = 1.0 * M_PI / 3.0;
    edges[2].type = 1;
    edges[3].atom = &atoms[3];
    edges[3].phi = 2.0 * M_PI / 3.0;
    edges[3].type = 1;
    edges[4].atom = &atoms[4];
    edges[4].phi = 5.0 * M_PI / 3.0;
    edges[4].type = 1;
    edges[5].atom = &atoms[5];
    edges[5].phi = 4.0 * M_PI / 3.0;
    edges[5].type = 1;
    
    printf ("Constructing Carbon rings ...\n");
    edges2 = (edge_t *) malloc (sizeof (edge_t) * nedges);
    assert (edges2 != NULL);

    for (k = 0; k < N - 1; k++)
    {
        int nh2;
        nh2 = 0;
        for (i = 0; i < nh; i++)
        {
            atom_t *a;
            double phi;
            double phi2;
            double x;
            double x1;
            double y;
            double y1;
            double x2;
            double y2;
            double x3;
            double y3;
            double phi3;
            int type;

            a = edges[i].atom;
            phi = edges[i].phi;
            type = edges[i].type;
            x = a->x;
            y = a->y;
            x1 = x + RCC * cos (phi);
            y1 = y + RCC * sin (phi);
            phi2 = phi + M_PI / 3.0;
            x2 = x1 + RCC * cos (phi2);
            y2 = y1 + RCC * sin (phi2);
            phi3 = phi - M_PI / 3.0;
            x3 = x1 + RCC * cos (phi3);
            y3 = y1 + RCC * sin (phi3);

            atoms[nc].name = 'C';
            atoms[nc].x = x1;
            atoms[nc].y = y1;
            atoms[nc].z = 0.0;
            atoms[nc].idx = nc;
            nc++;
            atoms[nc].name = 'C';
            atoms[nc].x = x2;
            atoms[nc].y = y2;
            atoms[nc].z = 0.0;
            atoms[nc].idx = nc;
            edges2[nh2].atom = &atoms[nc];
            edges2[nh2].phi = phi;
            edges2[nh2].type = 0;
            nc++;
            nh2++;
            if (type == 1)
            {
                atoms[nc].name = 'C';
                atoms[nc].x = x3;
                atoms[nc].y = y3;
                atoms[nc].z = 0.0;
                atoms[nc].idx = nc;         
                edges2[nh2].atom = &atoms[nc];
                edges2[nh2].phi = phi;
                edges2[nh2].type = 1;
                nh2++;
                nc++;
            }
        }
        nh = nh2;
        memcpy (edges, edges2, sizeof (edge_t) * nh);
    }
    assert (nh == nedges);
    printf ("Done\n");

    printf ("Adding Hydrogens ...\n");
    // Hydrogens
    for (i = 0; i < nh; i++)
    {
        atom_t *a;
        double phi;
        double x;
        double x1;
        double y;
        double y1;
        a = edges[i].atom;
        phi = edges[i].phi;
        x = a->x;
        y = a->y;
        x1 = x + RCH * cos (phi);
        y1 = y + RCH * sin (phi);
        atoms[nc].name = 'H';
        atoms[nc].x = x1;
        atoms[nc].y = y1;
        atoms[nc].z = 0.0;
        atoms[nc].idx = nc;
        nc++;
    }
    assert (nc == natoms);
    printf ("Done\n");
    printf ("Got %d atoms\n", nc);

    // Write XYZ
    output_xyz (natoms, atoms, nc);
    
    return 0;
}
