COMMON	= ../common

DBG      ?=
NVCC     ?= nvcc
CUDA_HOME?= /usr/local/cuda
NVFLAGS  = -I$(CUDA_HOME)/include --ptxas-options="-v" -gencode=arch=compute_35,code=\"sm_35,compute_35\" -Xcompiler -fopenmp
CXXFLAGS = -O3 -I. -I$(COMMON) $(DBG)

EXEC = nbody3

all: $(EXEC)

OBJS = $(EXEC:=.o)
DEPS = $(OBJS:.o=.d)

-include $(DEPS)

# Load common make options
include $(COMMON)/Makefile.common
LDFLAGS	= $(COMMON_LIBS) -lcudart -L$(CUDA_HOME)/lib64

%.o: %.cu
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -c $<
#$(NVCC) -MM $(CXXFLAGS) $< > $*.d

nbody3: nbody3.o $(COMMON_OBJS)
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -o nbody3 $^ $(LDFLAGS)

clean: clean_common
	/bin/rm -fv $(EXEC) *.d *.o *.optrpt
