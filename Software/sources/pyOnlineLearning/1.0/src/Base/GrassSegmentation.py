import numpy

import AlmClib
from Base import Constants as Const


class GrassSegmentation:
    instance = None

    @classmethod
    def getInst(cls):
        if cls.instance == None:
            cls.instance = GrassSegmentation(Const.imageWidth(), Const.imageHeight())
        return cls.instance

    # # define grass segmentation parameters
    MGS = 750;
    MOS = 1500;
    # MOS  = 2500;


    # indoor
    '''muC  = 0.38;
    sigC = 0.3;
    muS  = 0.5;
    sigS = 0.3;'''

    #outdoor
    muC = 0.27
    #muC  = 0.15
    sigC = 0.3

    muS = 0.6
    sigS = 0.3

    gs = None
    width = 0
    height = 0

    def __init__(self, width, height):
        self.width = width
        self.height = height
        sz = AlmClib.Type2DSize()
        sz.width = width
        sz.height = height

        self.gs = AlmClib.GrassSegC_new();
        AlmClib.GrassSegC_init(self.gs, sz);

        # set grass segmentation parameters
        AlmClib.GrassSegC_setMinGrassSize(self.gs, self.MGS);
        AlmClib.GrassSegC_setMinObstacleSize(self.gs, self.MOS);

        AlmClib.GrassSegC_setColorMean(self.gs, self.muC);
        AlmClib.GrassSegC_setColorSigma(self.gs, self.sigC);
        AlmClib.GrassSegC_setSaturationMean(self.gs, self.muS);
        AlmClib.GrassSegC_setSaturationSigma(self.gs, self.sigS);

        # indoor settings
        '''AlmClib.GrassSegC_setMinHue(                    self.gs, 0.33)
        AlmClib.GrassSegC_setMaxHue(                    self.gs, 0.5)

        AlmClib.GrassSegC_setMinSaturation(             self.gs, 0.45)
        AlmClib.GrassSegC_setMinIntensityValue(         self.gs, 0.06)
        AlmClib.GrassSegC_setMaxGrayValue(              self.gs, 0.99);
        AlmClib.GrassSegC_setSigmaEnvironmentColor(     self.gs, 3);
        AlmClib.GrassSegC_setSigmaEnvironmentSaturation(self.gs, 3);
        AlmClib.GrassSegC_setMinSigmaColor(             self.gs, 0.05);
        AlmClib.GrassSegC_setMinSigmaSaturation(        self.gs, 0.1);'''

        '''#outdoorDataSet settings
        AlmClib.GrassSegC_setMinHue(                    self.gs, 0.22)
        AlmClib.GrassSegC_setMaxHue(                    self.gs, 0.33)
        AlmClib.GrassSegC_setMinSaturation(             self.gs, 0.5)
        AlmClib.GrassSegC_setMinIntensityValue(         self.gs, 0.03)
        AlmClib.GrassSegC_setMaxGrayValue(              self.gs, 0.99);

        AlmClib.GrassSegC_setSigmaEnvironmentColor(     self.gs, 3);
        AlmClib.GrassSegC_setSigmaEnvironmentSaturation(self.gs, 3);
        AlmClib.GrassSegC_setMinSigmaColor(             self.gs, 0.05);
        AlmClib.GrassSegC_setMinSigmaSaturation(        self.gs , 0.1);'''

        #Demo settings
        AlmClib.GrassSegC_setMinHue(self.gs, 0.17)
        AlmClib.GrassSegC_setMaxHue(self.gs, 0.36)
        AlmClib.GrassSegC_setMinSaturation(self.gs, 0.5)
        AlmClib.GrassSegC_setMinIntensityValue(self.gs, 0.03)
        AlmClib.GrassSegC_setMaxGrayValue(self.gs, 0.99);

        AlmClib.GrassSegC_setSigmaEnvironmentColor(self.gs, 3);
        AlmClib.GrassSegC_setSigmaEnvironmentSaturation(self.gs, 3);
        AlmClib.GrassSegC_setMinSigmaColor(self.gs, 0.05);
        AlmClib.GrassSegC_setMinSigmaSaturation(self.gs, 0.1);

        '''#studRoom settings
        AlmClib.GrassSegC_setMinHue(                    self.gs, 0.18)
        AlmClib.GrassSegC_setMaxHue(                    self.gs, 0.3)

        AlmClib.GrassSegC_setMinSaturation(             self.gs, 0.6)
        AlmClib.GrassSegC_setMinIntensityValue(         self.gs, 0.03)
        AlmClib.GrassSegC_setMaxGrayValue(              self.gs, 1.0);

        AlmClib.GrassSegC_setSigmaEnvironmentColor(     self.gs, 3);
        AlmClib.GrassSegC_setSigmaEnvironmentSaturation(self.gs, 3);
        AlmClib.GrassSegC_setMinSigmaColor(             self.gs, 0.05);
        AlmClib.GrassSegC_setMinSigmaSaturation(        self.gs, 0.1);'''

    def createMask(self):
        return numpy.zeros((self.height, self.width), dtype='uint8')

    ## can be triggered on command line by pressing  "r" if color segmentation mis-adapts to anything other than grass
    def reset(self):
        AlmClib.GrassSegC_setColorMean(self.gs, self.muC);
        AlmClib.GrassSegC_setColorSigma(self.gs, self.sigC);
        AlmClib.GrassSegC_setSaturationMean(self.gs, self.muS);
        AlmClib.GrassSegC_setSaturationSigma(self.gs, self.sigS);

    def processImage(self, image, mask):
        AlmClib.GrassSegC_processAdaptive(self.gs, mask,
                                          image);  # call grass segmentation on rgbSegmentImage, store results in mask. Specs item 4-5

    def getColorMean(self):
        return AlmClib.GrassSegC_getColorMean(self.gs)

    def getSaturationMean(self):
        return AlmClib.GrassSegC_getSaturationMean(self.gs)

    def getColorSigma(self):
        return AlmClib.GrassSegC_getColorSigma(self.gs)

    def getSaturationSigma(self):
        return AlmClib.GrassSegC_getSaturationSigma(self.gs)

    def clear(self):
        # deallocation
        AlmClib.GrassSegC_clear(self.gs);
        AlmClib.GrassSegC_delete(self.gs);




