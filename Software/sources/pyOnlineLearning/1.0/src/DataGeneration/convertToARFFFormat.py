__author__ = 'vlosing'


import numpy as np

def convertToARFFFormat(datasetName, featuresFileName, labelsFileName, dstARFFFileName, featuresDelimiter=' '):
    features = np.loadtxt(featuresFileName, delimiter=featuresDelimiter, skiprows=1)
    labels = np.atleast_2d(np.loadtxt(labelsFileName, skiprows=1)).T
    data = np.hstack([features, labels])
    header = '@relation %s \n' % (datasetName)
    for i in np.arange(features.shape[1]):
        header += '@attribute attribute%d Numeric \n' % (i)
    header += '@attribute class {%s}\n' % (','.join(str(p) for p in np.sort(np.unique(labels).astype(int))))
    header += '@data'
    np.savetxt(dstARFFFileName, data, header=header, comments='', fmt='%.10g')

if __name__ == '__main__':
    convertToARFFFormat('borderOriginal', '/home/vlosing/storage/Features/stationary/Border/border-train.data', '/home/vlosing/storage/Features/stationary/Border/border-train.labels', '/home/vlosing/storage/Features/stationary/Border/borderTrain.arff')
    convertToARFFFormat('borderOriginal', '/home/vlosing/storage/Features/stationary/Border/border-test.data', '/home/vlosing/storage/Features/stationary/Border/border-test.labels', '/home/vlosing/storage/Features/stationary/Border/borderTest.arff')