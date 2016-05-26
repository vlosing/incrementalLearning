__author__ = 'viktor'
import numpy as np

def transformFromOneFile(srcFileName, samplesDstFileName, labelsDstFileName, delimiter=','):
  data = np.loadtxt(srcFileName, delimiter=delimiter)
  samples = data[:,:-1]
  labels = data[:,-1]
  np.savetxt(samplesDstFileName, samples, fmt='%.6g')
  np.savetxt(labelsDstFileName, labels, fmt='%i')

if __name__ == '__main__':
    #transformFromOneFile('/home/viktor/Downloads/NonStationaryData/benchmark/GEARS_2C_2D.txt', '/home/viktor/svn/storage/Features/nonStationary/souza/GEARS_2C_2D.data', '/home/viktor/svn/storage/Features/nonStationary/souza/GEARS_2C_2D.labels')
    #transformFromOneFile('/home/viktor/Downloads/NonStationaryData/benchmark/FG_2C_2D.txt', '/home/viktor/svn/storage/Features/nonStationary/souza/FG_2C_2D.data', '/home/viktor/svn/storage/Features/nonStationary/souza/FG_2C_2D.labels')
    #transformFromOneFile('/home/viktor/Downloads/NonStationaryData/powersupply.arff', '/home/viktor/svn/storage/Features/nonStationary/powerSupply/powersupply.data', '/home/viktor/svn/storage/Features/nonStationary/powerSupply/powersupply.labels')
    #transformFromOneFile('/homes/vlosing/RBFFast.arff', '/homes/vlosing/svn/storage/Features/nonStationary/rbf/rbfFast.data', '/homes/vlosing/svn/storage/Features/nonStationary/rbf/rbfFast.labels')
    #transformFromOneFile('/homes/vlosing/RBFSlow.arff', '/homes/vlosing/svn/storage/Features/nonStationary/rbf/rbfSlow.data', '/homes/vlosing/svn/storage/Features/nonStationary/rbf/rbfSlow.labels')
    #transformFromOneFile('/homes/vlosing/HypFast.arff', '/homes/vlosing/svn/storage/Features/nonStationary/hyperplane/hypFast.data', '/homes/vlosing/svn/storage/Features/nonStationary/hyperplane/hypFast.labels')
    transformFromOneFile('/homes/vlosing/hypSlowLarge.arff', '/homes/vlosing/svn/storage/Features/nonStationary/hyperplane/hypSlowLarge.data', '/homes/vlosing/svn/storage/Features/nonStationary/hyperplane/hypSlowLarge.labels')
    transformFromOneFile('/homes/vlosing/rbfSlowLarge.arff', '/homes/vlosing/svn/storage/Features/nonStationary/rbf/rbfSlowLarge.data', '/homes/vlosing/svn/storage/Features/nonStationary/rbf/rbfSlowLarge.labels')

