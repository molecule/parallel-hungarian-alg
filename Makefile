# Load CUDA using the following command
# module load cuda
#
CC = nvcc
CFLAGS = -O3 -arch=compute_35 -code=sm_35
#NVCCFLAGS = -O3 -arch=compute_35 -code=sm_35
GPP = g++
NVCCFLAGS = -O3 -g
LIBS = 

TARGETS = HungarianAlg

all:	$(TARGETS)

HungarianAlg: HungarianAlg.o 
	$(CC) -o $@ $(LIBS) HungarianAlg.o

HungarianAlg.o: HungarianAlg.cu HungarianAlg.h
	$(CC) -c $(NVCCFLAGS) HungarianAlg.cu

clean:
	rm -f *.o $(TARGETS) *.stdout *.txt
