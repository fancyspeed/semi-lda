CC=g++
MPICC=mpicxx

#CFLAGS=-ggdb3 -O3 -Wall -Wno-sign-compare -I/home/michael/install/mpich2/include/
CFLAGS=-O3 -Wall -Wno-sign-compare -I/usr/local/include/
MPIFLAGS=-I/usr/local/include/ /usr/local/lib/libmpich.so 

EXEC=infer mpi_slda

all: $(EXEC)

clean:
	rm -f *.o
	rm -f $(EXEC) 

cmd_flags.o: cmd_flags.cc cmd_flags.h
	$(CC) -c $(CFLAGS)  cmd_flags.cc -o cmd_flags.o

common.o: common.cc common.h
	$(CC) -c $(CFLAGS)  common.cc -o common.o

document.o: document.cc document.h common.o
	$(CC) -c $(CFLAGS)  document.cc -o document.o

model.o: model.cc model.h common.o
	$(CC) -c $(CFLAGS)  model.cc -o model.o

accumulative_model.o: accumulative_model.cc accumulative_model.h common.o model.o
	$(CC) -c $(CFLAGS)  accumulative_model.cc -o accumulative_model.o

sampler.o: sampler.cc sampler.h common.o document.o model.o accumulative_model.o
	$(CC) -c $(CFLAGS)  sampler.cc -o sampler.o

slda: slda.cc cmd_flags.o common.o document.o model.o accumulative_model.o sampler.o
	$(CC) $(CFLAGS) slda.cc cmd_flags.o common.o document.o model.o accumulative_model.o sampler.o -o slda

infer: infer.cc cmd_flags.o common.o document.o model.o accumulative_model.o sampler.o
	$(CC) $(CFLAGS) infer.cc cmd_flags.o common.o document.o model.o accumulative_model.o sampler.o -o infer

mpi_slda: mpi_slda.cc cmd_flags.o common.o document.o model.o accumulative_model.o sampler.o
	$(MPICC) $(CFLAGS) $(MPIFLAGS) mpi_slda.cc cmd_flags.o common.o document.o model.o accumulative_model.o sampler.o -o mpi_slda

