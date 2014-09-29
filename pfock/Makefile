include ../make.in

DYNAMIC=1

LIBPFOCK = libpfock.a

SRC = $(wildcard *.c)
OBJS := $(addsuffix .o, $(basename $(SRC)))

CFLAGS = -O3 -Wall -openmp -w2 -std=gnu99 -mkl 
CFLAGS += -Wunknown-pragmas -Wunused-variable
CFLAGS += -D__ALIGNLEN__=${alignlen}
CFLAGS += ${OPTFLAGS}

ifeq "${DYNAMIC}" "1"
CFLAGS += -D__DYNAMIC__
endif

INC = -I./                  \
      -I../include          \
      -I${MPI_INCDIR}       \
      -I${GA_INCDIR}        \
      -I${ERD_INCDIR}

all: ${LIBPFOCK}
	mkdir -p ../install
	cp -r ${LIBPFOCK} ../install/
	cp -r pfock.h ../install/        

${LIBPFOCK}: ${OBJS}
	${AR} $@ $^

%.o : %.c Makefile
	$(MPICC) ${CFLAGS} ${INC} -c $< -o $@ 

clean:
	rm -f *.o *.s *.d *~ *.a
