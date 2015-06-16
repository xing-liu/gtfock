On this page, a step-by-step description of the build process and necessary and optional environment variables is outlined.

## Setting up the proper environment variables ##
  * WORK\_TOP defines the work directory.
```
export WORK_TOP=$PWD
```
  * GA\_TOP defines where the Global Arrays library is (going to be) installed.
```
export GA_TOP=$WORK_TOP/GAlib
```
  * ARMCI\_TOP defines where the ARMCI library is (going to be) installed.
```
export ARMCI_TOP=$WORK_TOP/external-armci
```
  * ERD\_TOP defines where the OptERD library is (going to be) installed.
```
export ERD_TOP=$WORK_TOP/ERDlib
```

## Installing Global Arrays on MPI-3 or ARMCI ##
Skip this step if Global Arrays has already been installed.

### Installing Global Arrays on MPI-3 ###

  * Installing armci-mpi
```
cd $WORK_TOP
git clone git://git.mpich.org/armci-mpi.git || git clone http://git.mpich.org/armci-mpi.git
cd armci-mpi
git checkout mpi3rma
./autogen.sh
mkdir build
cd build
../configure CC=mpiicc --prefix=$WORK_TOP/external-armci
make -j12
make install
```

  * Installing Global Arrays
```
cd $WORK_TOP
wget http://hpc.pnl.gov/globalarrays/download/ga-5-3.tgz
tar xzf ga-5-3.tgz
cd ga-5-3
# carefully set the mpi executables that you want to use
./configure CC=mpiicc MPICC=mpiicc CXX=mpiicpc MPICXX=mpiicpc \
F77=mpiifort MPIF77=mpiifort FC=mpiifort MPIFC=mpiifort\
--with-mpi --with-armci=$WORK_TOP/external-armci --prefix=$WORK_TOP/GAlib
make -j12 install
```

### Installing Global Arrays on ARMCI ###
```
cp config_ga.py $WORK_TOP
# openib means using Infiniband
python config_ga.py download openib
```

## Installing the OptERD library ##
The OptERD library will been installed in $WORK\_TOP/ERDlib.
```
cd $WORK_TOP
svn co https://github.com/Maratyszcza/OptErd/trunk OptErd
cd $WORK_TOP/OptErd
make -j12
prefix=$WORK_TOP/ERDlib make install
```

## Installing GTFock ##
The GTFock libraries will be installed in $WORK\_TOP/gtfock/install/. The example SCF code can be found in $WORK\_TOP/gtfock/pscf/.

```
cd $WORK_TOP
svn co http://gtfock.googlecode.com/svn/trunk gtfock
cd $WORK_TOP/gtfock/

# Change the following variables in make.in
BLAS_INCDIR      = /opt/intel/mkl/include/
BLAS_LIBDIR      = /opt/intel/mkl/lib/intel64/
BLAS_LIBS        = -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm
SCALAPACK_INCDIR = /opt/intel/mkl/include/
SCALAPACK_LIBDIR = /opt/intel/mkl/lib/intel64/
SCALAPACK_LIBS   = -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64
MPI_LIBDIR = /opt/mpich2/lib/
MPI_LIBS =
MPI_INCDIR = /opt/mpich2/include

make
```