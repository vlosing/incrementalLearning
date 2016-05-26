import numpy, scipy.misc
import random
import matplotlib
import matplotlib.pyplot as plt
import json
import generateNetworkValues

import libpythoninterface


if __name__ == '__main__':
    generateNetworkValues.genNetworkValuesToFile(libpythoninterface.NETTYPE_GLVQ, False, 'GLVQ_LRFALSE')
    generateNetworkValues.genNetworkValuesToFile(libpythoninterface.NETTYPE_GLVQ, True, 'GLVQ_LRTRUE')
    generateNetworkValues.genNetworkValuesToFile(libpythoninterface.NETTYPE_GRLVQ, False, 'GRLVQ_LRFALSE')
    generateNetworkValues.genNetworkValuesToFile(libpythoninterface.NETTYPE_GRLVQ, True, 'GRLVQ_LRTRUE')
    generateNetworkValues.genNetworkValuesToFile(libpythoninterface.NETTYPE_GMLVQ, False, 'GMLVQ_LRFALSE')
    generateNetworkValues.genNetworkValuesToFile(libpythoninterface.NETTYPE_GMLVQ, True, 'GMLVQ_LRTRUE')
    generateNetworkValues.genNetworkValuesToFile(libpythoninterface.NETTYPE_LGRLVQ, False, 'LGRLVQ_LRFALSE')
    generateNetworkValues.genNetworkValuesToFile(libpythoninterface.NETTYPE_LGRLVQ, True, 'LGRLVQ_LRTRUE')
    generateNetworkValues.genNetworkValuesToFile(libpythoninterface.NETTYPE_LGMLVQ, False, 'LGMLVQ_LRFALSE')
    generateNetworkValues.genNetworkValuesToFile(libpythoninterface.NETTYPE_LGMLVQ, True, 'LGMLVQ_LRTRUE')