
# Directorio de trabajo:
DIR=milenet5

# NOMENCLATURA
# - SEQ: Sequential
# - COA: Coarse-grained
# - FIN: Fine-grained
# - GPU: Graphics processing unit

# Directorio de cada versión
SEQDIR=sequential
COADIR=coarse-grained
FINDIR=fine-grained
GPUDIR=gpu

# Objetivos de cada versión
SEQTAR=$(SEQDIR)/sequential
COATAR=$(COADIR)/coarse_grained
FINTAR=$(FINDIR)/fine_grained
GPUTAR=$(GPUDIR)/gpu

# Dependencias de cada versión
# NOTA: La versión COA hace uso del fichero 'lenet.c' de SEQ.
SEQDEP=$(SEQDIR)/main.c $(SEQDIR)/lenet.c lenet.h measure.h -std=c11 -lm
COADEP=$(COADIR)/main.c $(SEQDIR)/lenet.c lenet.h measure.h -std=c11 -lm -fopenmp
FINDEP=$(FINDIR)/main.c $(FINDIR)/lenet.c lenet.h measure.h -std=c11 -lm -fopenmp
GPUDEP=$(GPUDIR)/main.cu -lm

# Parámetros de compilación de cada versión
SEQCC=gcc $(SEQDEP) -o $(SEQTAR)
COACC=gcc $(COADEP) -o $(COATAR)
FINCC=gcc $(FINDEP) -o $(FINTAR)
GPUCC=nvcc $(GPUDEP) -o $(GPUTAR)

# Hilos a utilizar en las versiones OMP
THREADS ?= 4

# Órdenes para el profiling
SEQPROFDEP=$(SEQDEP) -pg
SEQPROFCC=gcc $(SEQPROFDEP) -o $(SEQTAR)

# Órdenes de compilación de cada versión
build_seq:
	cd $(DIR); $(SEQCC); cd ..
build_coa:
	cd $(DIR); $(COACC); cd ..
build_fin:
	cd $(DIR); $(FINCC); cd ..
build_gpu:
	cd $(DIR); $(GPUCC); cd ..

build_seqprof:
	cd $(DIR); $(SEQPROFCC); cd ..

profiling:
	cd $(DIR); gprof $(SEQTAR) gmon.out > prof_output; cd ..

# Órdenes de ejecución de cada versión
run_seq: build_seq
	cd $(DIR); $(SEQTAR); cd ..
run_coa: build_coa
	cd $(DIR); $(COATAR) $(THREADS); cd ..
run_fin: build_fin
	cd $(DIR); $(FINTAR) $(THREADS); cd ..
run_gpu: build_gpu
	cd $(DIR); $(GPUTAR); cd ..


run_all: run_seq run_fin run_coa run_gpu

# Generación de ficheros de entrada para la versión Verilog:
verilog_in:
	cd util; gcc main.c verilog.h floatToFixed.h -o verilog_in -lm; ./verilog_in

