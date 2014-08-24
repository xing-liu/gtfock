include ../make.in

EXEC = testpfock

SRC = $(wildcard *.c)
OBJS := $(addsuffix .o, $(basename $(SRC)))

CFLAGS = -O3 -Wall -openmp -mkl -std=gnu99 
CFLAGS += -Wunknown-pragmas -Wunused-variable #-Wunused-function
CFLAGS += ${OPTFLAGS}

INC = -I./                  \
      -I../include          \
      -I../libpfock         \
      -I${ERD_INCDIR}

LDFLAGS = -L${ERD_LIBDIR}

PFOCK_LIBS = ../lib/libpfock.a

LIBS = ${PFOCK_LIBS}     \
       ${ERD_LIBS}       \
       -lifcore -limf

all: ${EXEC} 

$(EXEC): $(OBJS) Makefile ${CINT_LIBS} ${PFOCK_LIBS}
	$(MPICC) ${CFLAGS} ${LDFLAGS} $(OBJS) -o $(EXEC) ${LIBS}

%.o : %.c Makefile
	$(MPICC) ${CFLAGS} ${INC} -c $< -o $@ 

clean:
	rm -f *.o *.s *.d *~ ${EXEC}
