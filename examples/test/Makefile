#
# Makefile for TinyEKF GPS example
#
# Copyright (C) 2015 Simon D. Levy
#
# MIT License

SRC = ../../src

all: test

run: test
	./test

test: test.c $(SRC)/tinyekf.h
	gcc -Wall -I. -I$(SRC) -o test test.c -lm

edit:
	vim test.c

clean:
	rm -f test *.o *~ ekf.csv
