#!/bin/bash


CC = gcc
CFLAGS = -Wall -I /hri/sit/latest/External/OpenCV/2.1/include/ -I /hri/sit/latest/External/python/2.6.6/lucid32/include/python2.6 -I /hri/sit/latest/External/pymodules2.6/numpy/1.5.1/lib/lucid32/numpy/core/include/numpy/

LDFLAGS = -L /hri/sit/latest/External/OpenCV/2.1/lib/lucid32/

PATHFLAGS = -DCOILDATABASE=\"/hri/localdisk/jqueisse/database/coil-100/obj\"


GRLVQLIBSOURCE = grlvq_c/src/Network_c.c grlvq_c/src/Prototype_c.c
GRLVQLIBINCLUDE = grlvq_c/src/Network_c.h  grlvq_c/src/Prototype_c.h

PYTHONINTERFACESOURCE = pythoninterface_c/src/intf.c

CCP = g++
GRLVQTESTCSOURCE = grlvqtest_c/src/grlvqtest.cpp
GRLVQTESTCLDFLAGS = -lpthread -lgrlvqc -lopencv_core -lopencv_imgproc -lopencv_highgui

std: clean grlvqc pythoninterface

all: std grlvqtestc

clean:
	rm -f *.o
	rm -f *.so
	rm -f *.bin
	rm -f *.sh
	rm -f *~



grlvqc: $(GRLVQLIBSOURCE) $(GRLVQLIBINCLUDE)
	$(CC) $(CFLAGS) -fPIC -c $(GRLVQLIBSOURCE)
	$(CC) -shared -o libgrlvqc.so $(notdir $(GRLVQLIBSOURCE:.c=.o))

pythoninterface: grlvqc $(PYTHONINTERFACESOURCE)
	$(CC) $(CFLAGS) -fPIC -c $(PYTHONINTERFACESOURCE) -I $(dir $(firstword $(GRLVQLIBINCLUDE)))
	$(CC) -shared -o libpythoninterface.so $(notdir $(PYTHONINTERFACESOURCE:.c=.o))
	echo "#!/bin/bash" > run_pythoninterface_test.sh
	echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:." >> run_pythoninterface_test.sh
	echo "python ./grlvqPythonTest/src/testDistribution.py" >> run_pythoninterface_test.sh
	echo "" >> run_pythoninterface_test.sh
	chmod +x run_pythoninterface_test.sh


grlvqtestc: grlvqc $(GRLVQTESTCSOURCE) $(PYTHONINTERFACESOURCE)
	$(CCP) $(PATHFLAGS) $(CFLAGS) -o grlvqtestc.bin $(GRLVQTESTCLDFLAGS) $(LDFLAGS) -L $(CURDIR) -I $(dir $(firstword $(GRLVQLIBINCLUDE))) $(GRLVQTESTCSOURCE)
	echo "#!/bin/bash" > run_grlvq_ctest.sh
	echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:." >> run_grlvq_ctest.sh
	echo "exec ./grlvqtestc.bin" >> run_grlvq_ctest.sh
	echo "" >> run_grlvq_ctest.sh
	chmod +x run_grlvq_ctest.sh







