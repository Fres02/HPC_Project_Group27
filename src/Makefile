.PHONY: all utils serial openmp pthreads cuda mpi hybrid clean

all: utils serial openmp pthreads cuda mpi hybrid

utils:
	$(MAKE) -C utils

serial: utils
	$(MAKE) -C serial

openmp: utils
	$(MAKE) -C openmp

pthreads: utils
	$(MAKE) -C pthreads

cuda: utils
	$(MAKE) -C cuda

mpi: utils
	$(MAKE) -C mpi

hybrid: utils
	$(MAKE) -C hybrid

clean:
	$(MAKE) -C utils clean
	$(MAKE) -C serial clean
	$(MAKE) -C openmp clean
	$(MAKE) -C pthreads clean
	$(MAKE) -C cuda clean
	$(MAKE) -C mpi clean
	$(MAKE) -C hybrid clean
