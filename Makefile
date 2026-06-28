MPICC ?= mpicc
CFLAGS ?= -O2 -g

all: libscicomm_pmpi.so

libscicomm_pmpi.so: scicomm_pmpi.c
	$(MPICC) $(CFLAGS) -std=c99 -Wall -Wextra -fPIC -shared -o $@ $< -pthread

clean:
	rm -f libscicomm_pmpi.so

.PHONY: all clean

