include ../make.in

EXEC = scf

SRC = $(wildcard *.c)
OBJS := $(addsuffix .o, $(basename $(SRC)))

CFLAGS = -O3 -Wall -openmp -mkl -std=gnu99 -fasm-blocks
CFLAGS += -Wunknown-pragmas -Wunused-variable #-Wunused-function
CFLAGS += ${OPTFLAGS}
#CFLAGS += -DSCF_TIMING

INC = -I./                  \
      -I../install          \
      -I${MPI_INCDIR}       \
      -I${SCALAPACK_INCDIR} \
      -I${GA_INCDIR}        \
      -I${ERD_INCDIR}

LDFLAGS = -L${MPI_LIBDIR}       \
          -L${GA_LIBDIR}        \
          -L${ARMCI_LIBDIR}     \
          -L${SCALAPACK_LIBDIR} \
          -L${ERD_LIBDIR}

PFOCK_LIBS = ../install/libpfock.a

LIBS = ${MPI_LIBS}       \
       ${GA_LIBS}        \
       ${ARMCI_LIBS}     \
       ${SCALAPACK_LIBS} \
       ${PFOCK_LIBS}     \
       ${ERD_LIBS}       \
       -lifcore -limf

all: ${EXEC} 

$(EXEC): $(OBJS) Makefile ${CINT_LIBS} ${PFOCK_LIBS}
	$(MPICC) ${CFLAGS} ${LDFLAGS} $(OBJS) -o $(EXEC) ${LIBS}

%.o : %.c Makefile
	$(MPICC) ${CFLAGS} ${INC} -c $< -o $@ 

clean:
	rm -f *.o *.s *.d *~ ${EXEC}
