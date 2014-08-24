include ../make.in

CFLAGS = -O3 -Wall -openmp -std=gnu99
CFLAGS += -Wunknown-pragmas -Wunused-variable

INC = -I./ \
      -I${ERD_INCDIR}

LDFLAGS = -L${ERD_LIBDIR}

LIBS = ${ERD_LIBS}

all: graphene sort_mol

graphene: graphene.c Makefile
	$(CC) ${INC} ${CFLAGS} ${LDFLAGS} $< -o $@ ${LIBS}

sort_mol: sort_mol.c Makefile
	$(CC) ${INC} ${CFLAGS} ${LDFLAGS} $< -o $@ ${LIBS}

%.o : %.c Makefile
	$(CC) ${CFLAGS} ${INC} -c $< -o $@ 

clean:
	rm -f *.o *.s *.d *~ graphene sort_mol
