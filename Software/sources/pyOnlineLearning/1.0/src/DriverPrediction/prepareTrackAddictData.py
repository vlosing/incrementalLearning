__author__ = 'vlosing'
import numpy as np
import os
import matplotlib.pyplot as plt

def filterNoGPSDataTrackAddictData(srcFileDir, dstFileDir, keepColumns=range(12), srcFilePath='', maximumGPSDelay = 1):
    allSrcFilePaths = []
    if srcFilePath != '':
        allSrcFilePaths.append(srcFilePath)
    else:
        for file in os.listdir(srcFileDir):
            if file.endswith(".csv"):
                allSrcFilePaths.append(srcFileDir + file)



    for filePath in allSrcFilePaths:
        data = np.loadtxt(filePath, delimiter=',', skiprows=2)
        fileName = os.path.basename(filePath)
        dstFileDir = os.path.normpath(dstFileDir) + os.sep
        data = data[data[:, 3] <= maximumGPSDelay]
        data = data[data[:, 2] == 1][:, keepColumns]
        columnNames = np.array(['Time', 'Lap', 'GPS_Update', 'GPS_Delay', 'Accuracy (m)', 'Latitude', 'Longitude', 'Altitude (m)', 'Speed (KM/H)', 'Heading', 'X', 'Y', 'Z'])
        header = ', '.join(columnNames[keepColumns])
        np.savetxt(dstFileDir + fileName, data, delimiter=',', fmt='%10.7f', header=header, comments='')

def getGPSFieldsOnly(srcFileDir, dstFileDir, srcFilePath=''):
    filterNoGPSDataTrackAddictData(srcFileDir, dstFileDir=dstFileDir, keepColumns=[5, 6], srcFilePath=srcFilePath)

def getFeatureRelevantFields(srcFileDir, dstFileDir, srcFilePath=''):
    filterNoGPSDataTrackAddictData(srcFileDir, dstFileDir=dstFileDir, keepColumns=[0, 5, 6, 8, 9], srcFilePath=srcFilePath)

if __name__ == '__main__':
    sourceDirPrefix = '/hri/storage/user/PASS/Data/TrackAddict/martina/'
    directionDirName = 'fromHonda'
    #directionDirName = 'toHonda'
    srcDir = sourceDirPrefix + 'processed/originalRenamed/' + directionDirName +'/'
    dstDir = sourceDirPrefix + 'processed/featureRelevant/' + directionDirName +'/'
    #getFeatureRelevantFields(srcDir, dstDir)

    srcDir = sourceDirPrefix + 'processed/originalRenamed/' + directionDirName +'/'
    dstDir = sourceDirPrefix + 'processed/GPSOnly/' + directionDirName +'/'
    #getGPSFieldsOnly(srcDir, dstDir)

    #filterNoGPSDataTrackAddictData('', '/hri/storage/user/PASS/Data/TrackAddict/viktor/Comparison/filtered/all/', srcFilePath='/hri/storage/user/PASS/Data/TrackAddict/viktor/Comparison/filtered/IPhone3.csv')
    #filterNoGPSDataTrackAddictData('', '/hri/storage/user/PASS/Data/TrackAddict/viktor/Comparison/filtered/all/', srcFilePath='/hri/storage/user/PASS/Data/TrackAddict/viktor/Comparison/filtered/XGPS3.csv')

    #getGPSFieldsOnly('', '/hri/storage/user/PASS/Data/TrackAddict/viktor/Comparison/filtered/gps/', srcFilePath='/hri/storage/user/PASS/Data/TrackAddict/viktor/Comparison/filtered/IPhone2.csv')
    #getGPSFieldsOnly('', '/hri/storage/user/PASS/Data/TrackAddict/viktor/Comparison/filtered/gps/', srcFilePath='/hri/storage/user/PASS/Data/TrackAddict/viktor/Comparison/filtered/XGPS2.csv')


    srcDir = '/hri/storage/user/PASS/Data/TrackAddict/viktor/Comparison/filtered/'
    dstDir = '/hri/storage/user/PASS/Data/TrackAddict/viktor/Comparison/filtered/all/'
    getFeatureRelevantFields(srcDir, dstDir)


    dstDir = '/hri/storage/user/PASS/Data/TrackAddict/viktor/Comparison/filtered/gps/'
    getGPSFieldsOnly(srcDir, dstDir)




    iphone = np.loadtxt('/hri/storage/user/PASS/Data/TrackAddict/viktor/Comparison/filtered/all/IPhone3.csv', delimiter=',', skiprows=1)
    xgps = np.loadtxt('/hri/storage/user/PASS/Data/TrackAddict/viktor/Comparison/filtered/all/XGPS3.csv', delimiter=',', skiprows=1)

    iphone4 = np.loadtxt('/hri/storage/user/PASS/Data/TrackAddict/viktor/Comparison/filtered/all/IPhone4.csv', delimiter=',', skiprows=1)
    xgps4 = np.loadtxt('/hri/storage/user/PASS/Data/TrackAddict/viktor/Comparison/filtered/all/XGPS4.csv', delimiter=',', skiprows=1)

    offSet = 1
    fig, ax1 = plt.subplots(1, 1, sharex=True)
    fig, ax2 = plt.subplots(1, 1, sharex=True)
    ax1.plot(iphone[:,0]+offSet, iphone[:,3], c='r')
    ax1.plot(xgps[:,0], xgps[:,3], c='b')

    ax2.scatter(iphone[:,0]+offSet, iphone[:,3], c='r')
    ax2.scatter(xgps[:,0], xgps[:,3], c='b')


    offSet = 14.5
    fig, ax3 = plt.subplots(1, 1, sharex=True)
    fig, ax4 = plt.subplots(1, 1, sharex=True)

    ax3.plot(iphone4[:,0]+offSet, iphone4[:,3], c='r')
    #plt.plot(xgps[:,0]+2.5, xgps[:,3], c='b')
    ax3.plot(xgps4[:,0], xgps4[:,3], c='b')

    ax4.scatter(iphone4[:,0]+offSet, iphone4[:,3], c='r')
    ax4.scatter(xgps4[:,0], xgps4[:,3], c='b')


    plt.show()