__author__ = 'viktor'
import os
import subprocess

def compress(srcFileDir, frameRate=10, size='360x190'):
    srcFileDir = os.path.normpath(srcFileDir) + os.sep
    outPutFileDir = srcFileDir + 'downsampled/'
    if not os.path.exists(outPutFileDir):
        os.makedirs(outPutFileDir)

    for fileName in os.listdir(srcFileDir):
        if fileName.endswith(".avi"):
            filePath = os.path.join(srcFileDir, fileName)
            outputFilePath = os.path.join(outPutFileDir, fileName)
            cmd = ["ffmpeg -i %s -r %d -s %s %s" % (filePath, frameRate, size, outputFilePath)]
            subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    #compress('/home/viktor/Downloads')
    compress('/hri/storage/user/PASS/Heiko')

    compress('/hri/storage/user/PASS/Martina')
    compress('/hri/storage/user/PASS/Viktor')
