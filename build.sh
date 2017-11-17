#! /bin/sh
# make sure you have icc, mkl, JDK installed
export MKLROOT=/opt/intel/mkl
export JDK=''
javac CBLAS.java
javah CBLAS
icc -shared -fPIC -o libmkl_java_stubs.so CBLAS.c -I. -I$MKLROOT/include -I$JDK/include/linux -I$JDK/include/ -Wl,--start-group $MKLROOT/lib/intel64/libmkl_intel_lp64.a $MKLROOT/lib/intel64/libmkl_intel_thread.a $MKLROOT/lib/intel64/libmkl_core.a -Wl,--end-group -openmp -lpthread -lm -ldl
sbt clean package
