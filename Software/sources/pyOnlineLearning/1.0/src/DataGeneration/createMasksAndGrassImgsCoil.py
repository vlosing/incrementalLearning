import numpy

from Base import Constants as Const
from Base import Serialization as serial
from Base import Paths


if __name__ == '__main__':
    fileIterator = serial.SimpleFileIterator(Paths.coilOrgImagesPath(), Const.coilImageFilesSuffix())
    coilImageHeight = 128
    coilImageWidth = 128
    thresh = 33
    starCol = (Const.imageWidth() - coilImageWidth) / 2
    starRow = (Const.imageHeight() - coilImageHeight) / 2
    grassImageOrig = serial.loadImage(Const.imageWidth(), Const.imageHeight(), Paths.coilEmptyTemplateFileName())
    while True:
        imageFileName = fileIterator.getNextFileName()
        if imageFileName is None:
            break

        coilImage = serial.loadImage(coilImageWidth, coilImageHeight, Paths.coilOrgImagesPath() + imageFileName)
        grassImage = grassImageOrig.copy()
        mask = numpy.zeros((Const.imageHeight(), Const.imageWidth()), dtype='uint8')
        for row in range(coilImageHeight):
            for col in range(coilImageWidth):
                if coilImage[row][col][0] > thresh or \
                                coilImage[row][col][0] > thresh or coilImage[row][col][0] > thresh:
                    grassImage[starRow + row][starCol + col][0] = coilImage[row][col][0]
                    grassImage[starRow + row][starCol + col][1] = coilImage[row][col][1]
                    grassImage[starRow + row][starCol + col][2] = coilImage[row][col][2]
                    mask[starRow + row][starCol + col] = 255

        serial.saveImage(grassImage, Paths.coilOrgImagesPath() + "/new2/" + imageFileName, True, Const.imageHeight(),
                         Const.imageWidth())
        serial.saveImage(mask, Paths.coilOrgImagesPath() + "/mask2/" + imageFileName)


