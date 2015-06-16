## API Reference ##
GTFock provides a flexible interface for constructing Fock matrices on distributed systems. A typical invocation of GTFock is composed of the following five steps.

  1. Creating a PFock computing engine
```
/**
 * Create a PFock computing engine on 3x3 MPI processes.
 * Each MPI process owns 16 (4x4) tasks.
 * The screening threshold is 1e-10. The maximum number
 * of density matrices can be computed is 5. The input
 * density
 * matrices can be non-symmetric.
 **/ 
PFock_t pfock;
PFock_create(basis, 3, 3, 4, 1e-10, 5, 0, &pfock);
```
  1. Setting the number of density matrices to be computed
```
// The number of Fock matrices to be computed is 3.
PFock_setNumDenMat(3, pfock);
```
  1. Putting local data into the global densities matrices
```
/**
  * Put the local data onto the range (rowstart:rowend, colstart:colend)
  * of the global density matrix 2.
 **/
PFock_putDenMat(rowstart, rowend, colstart, colend, ld, localmat, 2, pfock);
PFock_commitDenMats(pfock);
```
  1. Computing Fock matrices
```
PFock_computeFock(basis, pfock);
```
  1. Getting data from the global Fock matrices
```
/**
  * Get the local data from the range (rowstart:rowend, colstart:colend)
  * of the global Fock matrix 1.
 **/
PFock_getMat(pfock, PFOCK_MAT_TYPE_F, 1, rowstart, rowend, colstart, colend, stride, F_block);
```

The detailed API reference can found [here](http://www.cc.gatech.edu/~xliu66/gtfock/doc/html/pfock_8h.html).

## Example SCF code ##
GTFock also includes an example SCF code, which demonstrates how to use GTFock to build a non-trivial quantum chemistry application. The example SCF code uses a variant of McWeeny purification, called canonical purification, to compute the density matrix, which scales better than diagonalization approaches. In the example SCF code, a 3D matrix-multiply kernel is implemented for efficient purification calculations.

The example SCF code can be run as
```
mpirun -np <nprocs> ./scf <basis> <xyz> <nprow> <npcol> <np2> <ntasks> <niters>
```
  * **nprocs**: the number of MPI processes
  * **basis**:  the basis file
  * **xyz**:    the xyz file
  * **nprow**:  the number of MPI processes per row
  * **npcol**:  the number of MPI processes per col
  * **np2**: the number of MPI processes per one dimension for purification (eigenvalue solve)
  * **ntasks**: the each MPI process has ntasks x ntasks tasks
  * **niters**: the number of SCF iterations

**NOTE**
  1. **nprow** x **npcol** must be equal to nprocs
  1. **np2** x **np2 x**np2**must be smaller than nprocs
  1. suggested values for**ntasks**: 3, 4, 5**

For example, the following command run the SCF code on graphene\_12\_54\_114.xyz with 12 MPI processes for 10 iterations. The number of processes used for purification is 2 x 2 x 2 = 8.
```
mpirun -np 12 ./pscf/scf data/guess/cc-pvdz.gbs data/graphene/graphene_12_54_114.xyz 3 4 2 4 10
```