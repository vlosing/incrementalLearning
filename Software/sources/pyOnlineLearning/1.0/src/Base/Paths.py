import os
import socket

def projectDirectory():
    return os.environ['ONLINE_LEARNING_DIR']

def moaDirectory():
    return storageDir() + '/moa'

def orfDirectory():
    #return os.path.join(projectDirectory(), 'Software/sources/OnlineRandomForest/online-multiclass-lpboost-master')
    return storageDir() + '/ORF/' + socket.gethostname() + '/online-multiclass-lpboost-master'

def orfConfFilePath():
    return os.path.join(orfDirectory(),'conf/omcb.conf')

def storageDir():
    return os.environ['ONLINE_LEARNING_STORAGE_DIR']

def trackAddictDir():
    return os.environ['TRACK_ADDICT_STORAGE_DIR']

def ImagesDirPrefix():
    return "../../../../Data/Images/"

def FeaturesJSONDirPrefix():
    #return getProjectDirectory() + 'Software/versionedData/Features/JSON/'
    return storageDir() + '/Features/JSON/'

def CutInFeaturesDir():
    #return getProjectDirectory() + 'Software/versionedData/Features/JSON/'
    return FeaturesJSONDirPrefix() + 'CutIn/'

def CutInTestFeaturesPath():
    return CutInFeaturesDir() + 'TestFeatures.json'

def CutInTestLabelsPath():
    return CutInFeaturesDir() + 'TestLabels.json'

def HystereseCutInDir():
    return storageDir() + '/HystereseCutIn/'

def HystereseCutInResultsDir():
    return storageDir() + '/HystereseCutInResults/'

def stationaryFeaturesDirPrefix():
    #return getProjectDirectory() + 'Software/versionedData/Features/ORF/'
    return storageDir() +  '/Features/stationary/'
    #return os.path.join(storageDir(), '/Features/ORF/')

def nonStationaryFeaturesDirPrefix():
    #return getProjectDirectory() + 'Software/versionedData/Features/ORF/'
    return storageDir() +  '/Features/nonStationary/'

def FeatureTmpsDirPrefix():
    #return getProjectDirectory() + 'Software/versionedData/Features/ORF/'
    return storageDir() +  '/Features/Tmp/'
    #return storageDir() +  '/Features/Tmp2/'
    #return storageDir() +  '/Features/Tmp3/'
    #return storageDir() +  '/Features/Tmp4/'

def IndoorImagesPath():
    return ImagesDirPrefix() + 'IndoorObjects/'

def OutdoorCloudedImagesPath():
    return ImagesDirPrefix() + 'Outdoor/Clouded/WithoutFragmentsFinal/'


def OutdoorSunnyImagesPath():
    return ImagesDirPrefix() + 'Outdoor/Sunny/WithoutFragmentsFinal/'


def demoStreamDir():
    return ImagesDirPrefix() + 'Demo/RecordedImages_demo/Images_Smaller/'


def demoPreTrainImagesPath():
    return ImagesDirPrefix() + 'Demo/DemoPreTrain/'


def USPSTrainFeaturesPath():
    return stationaryFeaturesDirPrefix() + 'USPS/usps-train.data'


def USPSTrainLabelsPath():
    return stationaryFeaturesDirPrefix() + 'USPS/usps-train.labels'


def USPSTestFeaturesPath():
    return stationaryFeaturesDirPrefix() + 'USPS/usps-test.data'


def USPSTestLabelsPath():
    return stationaryFeaturesDirPrefix() + 'USPS/usps-test.labels'


def DNATrainFeaturesPath():
    return stationaryFeaturesDirPrefix() + 'DNA/dna-train.data'


def DNATrainLabelsPath():
    return stationaryFeaturesDirPrefix() + 'DNA/dna-train.labels'


def DNATestFeaturesPath():
    return stationaryFeaturesDirPrefix() + 'DNA/dna-test.data'


def DNATestLabelsPath():
    return stationaryFeaturesDirPrefix() + 'DNA/dna-test.labels'


def LetterTrainFeaturesPath():
    return stationaryFeaturesDirPrefix() + 'Letter/letter-train.data'


def LetterTrainLabelsPath():
    return stationaryFeaturesDirPrefix() + 'Letter/letter-train.labels'


def LetterTestFeaturesPath():
    return stationaryFeaturesDirPrefix() + 'Letter/letter-test.data'


def LetterTestLabelsPath():
    return stationaryFeaturesDirPrefix() + 'Letter/letter-test.labels'


def GisetteTrainFeaturesPath():
    return stationaryFeaturesDirPrefix() + 'GISETTE/gisette_train.data'

def GisetteTrainLabelsPath():
    return stationaryFeaturesDirPrefix() + 'GISETTE/gisette_train.labels'

def GisetteTestFeaturesPath():
    return stationaryFeaturesDirPrefix() + 'GISETTE/gisette_valid.data'

def GisetteTestLabelsPath():
    return stationaryFeaturesDirPrefix() + 'GISETTE/gisette_valid.labels'


def GisetteSmallTrainFeaturesPath():
    return stationaryFeaturesDirPrefix() + 'GISETTE/gisette_train_small.data'

def GisetteSmallTrainLabelsPath():
    return stationaryFeaturesDirPrefix() + 'GISETTE/gisette_train_small.labels'

def GisetteSmallTestFeaturesPath():
    return stationaryFeaturesDirPrefix() + 'GISETTE/gisette_valid_small.data'

def GisetteSmallTestLabelsPath():
    return stationaryFeaturesDirPrefix() + 'GISETTE/gisette_valid_small.labels'


def isoletTrainFeaturesPath():
    return stationaryFeaturesDirPrefix() + 'ISOLET/isolet1+2+3+4_train.data'

def isoletTrainLabelsPath():
    return stationaryFeaturesDirPrefix() + 'ISOLET/isolet1+2+3+4_train.labels'

def isoletTestFeaturesPath():
    return stationaryFeaturesDirPrefix() + 'ISOLET/isolet5_test.data'

def isoletTestLabelsPath():
    return stationaryFeaturesDirPrefix() + 'ISOLET/isolet5_test.labels'


def elecTrainFeaturesPath():
    return nonStationaryFeaturesDirPrefix() + 'Elec2/elec2_data.dat'

def elecTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'Elec2/elec2_label.dat'

def news20TrainPath():
    return stationaryFeaturesDirPrefix() + 'news20/news20'

def news20TestPath():
    return stationaryFeaturesDirPrefix() + 'news20/news20.t'

def seaTrainFeaturesPath():
    return nonStationaryFeaturesDirPrefix() + 'sea/SEA_training_data.csv'

def seaTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'sea/SEA_training_class.csv'

def seaTestFeaturesPath():
    return nonStationaryFeaturesDirPrefix() + 'sea/SEA_testing_data.csv'

def seaTestLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'sea/SEA_testing_class.csv'

def MNistTrainSamplesPath():
    return stationaryFeaturesDirPrefix() + 'MNIST/mnistTrainSamples.data'

def MNistTrainLabelsPath():
    return stationaryFeaturesDirPrefix() + 'MNIST/mnistTrainLabels.data'

def MNistTestSamplesPath():
    return stationaryFeaturesDirPrefix() + 'MNIST/mnistTestSamples.data'

def MNistTestLabelsPath():
    return stationaryFeaturesDirPrefix() + 'MNIST/mnistTestLabels.data'


def MNistSmallTrainSamplesPath():
    return stationaryFeaturesDirPrefix() + 'MNIST/mnistSmallTrainSamples.data'

def MNistSmallTrainLabelsPath():
    return stationaryFeaturesDirPrefix() + 'MNIST/mnistSmallTrainLabels.data'

def MNistSmallTestSamplesPath():
    return stationaryFeaturesDirPrefix() + 'MNIST/mnistSmallTestSamples.data'

def MNistSmallTestLabelsPath():
    return stationaryFeaturesDirPrefix() + 'MNIST/mnistSmallTestLabels.data'


def outdoorTrainSamplesPath():
    return stationaryFeaturesDirPrefix() + 'Outdoor/Outdoor-train.data'

def outdoorTrainLabelsPath():
    return stationaryFeaturesDirPrefix() + 'Outdoor/Outdoor-train.labels'

def outdoorTestSamplesPath():
    return stationaryFeaturesDirPrefix() + 'Outdoor/Outdoor-test.data'

def outdoorTestLabelsPath():
    return stationaryFeaturesDirPrefix() + 'Outdoor/Outdoor-test.labels'


def outdoorStreamSamplesPath():
    return stationaryFeaturesDirPrefix() + 'Outdoor/outdoorStream.data'

def outdoorStreamLabelsPath():
    return stationaryFeaturesDirPrefix() + 'Outdoor/outdoorStream.labels'

def coilTrainSamplesPath():
    return stationaryFeaturesDirPrefix() + 'COIL/COIL-train.data'

def coilTrainLabelsPath():
    return stationaryFeaturesDirPrefix() + 'COIL/COIL-train.labels'

def coilTestSamplesPath():
    return stationaryFeaturesDirPrefix() + 'COIL/COIL-test.data'

def coilTestLabelsPath():
    return stationaryFeaturesDirPrefix() + 'COIL/COIL-test.labels'


def borderTrainSamplesPath():
    return stationaryFeaturesDirPrefix() + 'Border/border-train.data'

def borderTrainLabelsPath():
    return stationaryFeaturesDirPrefix() + 'Border/border-train.labels'

def borderTestSamplesPath():
    return stationaryFeaturesDirPrefix() + 'Border/border-test.data'

def borderTestLabelsPath():
    return stationaryFeaturesDirPrefix() + 'Border/border-test.labels'

def borderTrainARFFPath():
    return stationaryFeaturesDirPrefix() + 'Border/borderTrain.arff'

def borderTestARFFPath():
    return stationaryFeaturesDirPrefix() + 'Border/borderTest.arff'

def overlapTrainSamplesPath():
    return stationaryFeaturesDirPrefix() + 'Overlap/overlap-train.data'

def overlapTrainLabelsPath():
    return stationaryFeaturesDirPrefix() + 'Overlap/overlap-train.labels'

def overlapTestSamplesPath():
    return stationaryFeaturesDirPrefix() + 'Overlap/overlap-test.data'

def overlapTestLabelsPath():
    return stationaryFeaturesDirPrefix() + 'Overlap/overlap-test.labels'

def noiseTrainSamplesPath():
    return stationaryFeaturesDirPrefix() + 'noise/noise-train.data'

def noiseTrainLabelsPath():
    return stationaryFeaturesDirPrefix() + 'noise/noise-train.labels'

def noiseTestSamplesPath():
    return stationaryFeaturesDirPrefix() + 'noise/noise-test.data'

def noiseTestLabelsPath():
    return stationaryFeaturesDirPrefix() + 'noise/noise-test.labels'

def satImageTrainSamplesPath():
    return stationaryFeaturesDirPrefix() + 'satImage/satImage-train.data'

def satImageTrainLabelsPath():
    return stationaryFeaturesDirPrefix() + 'satImage/satImage-train.labels'

def satImageTestSamplesPath():
    return stationaryFeaturesDirPrefix() + 'satImage/satImage-test.data'

def satImageTestLabelsPath():
    return stationaryFeaturesDirPrefix() + 'satImage/satImage-test.labels'

def HARTrainSamplesPath():
    return stationaryFeaturesDirPrefix() + 'HAR/HAR_train.data'

def HARTrainLabelsPath():
    return stationaryFeaturesDirPrefix() + 'HAR/HAR_train.labels'

def HARTestSamplesPath():
    return stationaryFeaturesDirPrefix() + 'HAR/HAR_test.data'

def HARTestLabelsPath():
    return stationaryFeaturesDirPrefix() + 'HAR/HAR_test.labels'

def optDigitsTrainSamplesPath():
    return stationaryFeaturesDirPrefix() + 'optDigits/optDigits_train.data'

def optDigitsTrainLabelsPath():
    return stationaryFeaturesDirPrefix() + 'optDigits/optDigits_train.labels'

def optDigitsTestSamplesPath():
    return stationaryFeaturesDirPrefix() + 'optDigits/optDigits_test.data'

def optDigitsTestLabelsPath():
    return stationaryFeaturesDirPrefix() + 'optDigits/optDigits_test.labels'

def weatherTrainFeaturesPath():
    return nonStationaryFeaturesDirPrefix() + 'weather/NEweather_data.csv'

def weatherTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'weather/NEweather_class.csv'

def spamTrainFeaturesPath():
    return nonStationaryFeaturesDirPrefix() + 'spam/spam.data'

def spamTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'spam/spam.labels'


def PenDigitsTrainFeaturesPath():
    return stationaryFeaturesDirPrefix() + 'PenDigits/pendigits-train.data'

def PenDigitsTrainLabelsPath():
    return stationaryFeaturesDirPrefix() + 'PenDigits/pendigits-train.labels'


def PenDigitsTestFeaturesPath():
    return stationaryFeaturesDirPrefix() + 'PenDigits/pendigits-test.data'


def PenDigitsTestLabelsPath():
    return stationaryFeaturesDirPrefix() + 'PenDigits/pendigits-test.labels'


def demoPreTrainFeaturesPath():
    return FeaturesJSONDirPrefix() + 'DemoPreTrainFeatures.json'


def demoPreTrainLabelsPath():
    return FeaturesJSONDirPrefix() + 'DemoPreTrainLabels.json'


def demoPreTrainFeatureFileNamesFileName():
    return FeaturesJSONDirPrefix() + 'DemoPreTrainFeatureFileNames.json'


def getDemoPreTrainPaths():
    return demoPreTrainImagesPath(), demoPreTrainFeaturesPath(), demoPreTrainLabelsPath(), demoPreTrainFeatureFileNamesFileName()


def demoAppDataMissClassifiedImagesDir():
    return ImagesDirPrefix() + 'AppMissClassified/'


def demoAppDataMissClassifiedImagesStatFileName():
    return ImagesDirPrefix() + 'AppMissClassified/stat.json'


def OutdoorImagesDir():
    return ImagesDirPrefix() + 'Outdoor/Sunny+Clouded/'


def outdoorFeaturesFileName():
    return FeaturesJSONDirPrefix() + 'Outdoor/OutdoorFeatures.json'


def outdoorLabelsFileName():
    return FeaturesJSONDirPrefix() + 'Outdoor/OutdoorLabels.json'


def outdoorFeatureFileNamesFileName():
    return FeaturesJSONDirPrefix() + 'Outdoor/OutdoorFeatureFileNames.json'


def outdoorMissClassifiedImagesDir():
    return ImagesDirPrefix() + 'OutdoorMissClassified/'


def outdoorMissClassifiedImagesStatFileName():
    return ImagesDirPrefix() + 'OutdoorMissClassified/stat.json'


def getOutdoorAllPaths():
    return OutdoorImagesDir(), outdoorFeaturesFileName(), outdoorLabelsFileName(), outdoorFeatureFileNamesFileName()


def OutdoorEasyImagesDir():
    return ImagesDirPrefix() + 'Outdoor/SunnyCloudedMedium2/'


def outdoorEasyFeaturesFileName():
    return FeaturesJSONDirPrefix() + 'Outdoor/OutdoorEasyFeatures.json'


def outdoorEasyFeaturesFileNameRGB():
    return FeaturesJSONDirPrefix() + 'Outdoor/OutdoorEasyFeaturesRGB.json'


def outdoorEasyLabelsFileName():
    return FeaturesJSONDirPrefix() + 'Outdoor/OutdoorEasyLabels.json'


def outdoorEasyLabelsFileNameRGB():
    return FeaturesJSONDirPrefix() + 'Outdoor/OutdoorEasyLabelsRGB.json'


def outdoorEasyFeatureFileNamesFileName():
    return FeaturesJSONDirPrefix() + 'Outdoor/OutdoorEasyFeatureFileNames.json'


def outdoorEasyFeatureFileNamesFileNameRGB():
    return FeaturesJSONDirPrefix() + 'Outdoor/OutdoorEasyFeatureFileNamesRGB.json'


def outdoorEasyMissClassifiedImagesDir():
    return ImagesDirPrefix() + 'OutdoorEasyMissClassified/'


def outdoorEasyMissClassifiedImagesStatFileName():
    return ImagesDirPrefix() + 'OutdoorEasyMissClassified/stat.json'


def OutdoorTestImageDir():
    return ImagesDirPrefix() + 'Outdoor/Test/'


def OutdoorSequenceIndicesPath():
    return '../Statistics/Classifications/Indices'


def getOutdoorEasyPaths():
    return OutdoorEasyImagesDir(), outdoorEasyFeaturesFileName(), outdoorEasyLabelsFileName(), outdoorEasyFeatureFileNamesFileName()


def getOutdoorEasyPathsRGB():
    return OutdoorEasyImagesDir(), outdoorEasyFeaturesFileNameRGB(), outdoorEasyLabelsFileNameRGB(), outdoorEasyFeatureFileNamesFileNameRGB()


def optDigitsFeaturesFileName():
    return FeaturesJSONDirPrefix() + 'OptDigitsFeatures.json'


def optDigitsLabelsFileName():
    return FeaturesJSONDirPrefix() + 'OptDigitsLabels.json'


def getOptDigitsPaths():
    return optDigitsFeaturesFileName(), optDigitsLabelsFileName()


def coilEmptyTemplateFileName():
    return ImagesDirPrefix() + 'emptyTemplate.png'


def coilOrgImagesPath():
    return ImagesDirPrefix() + 'coil-100/'


def coilImagesDir():
    return ImagesDirPrefix() + 'coil-100/new/'


def coilImagesMaskDir():
    return ImagesDirPrefix() + 'coil-100/mask/'


def coilFeaturesFileName():
    return FeaturesJSONDirPrefix() + 'COILFeatures.json'


def coilFeaturesFileNameRGB():
    return FeaturesJSONDirPrefix() + 'COILFeaturesRGB.json'


def coilLabelsFileName():
    return FeaturesJSONDirPrefix() + 'COILLabels.json'


def coilLabelsFileNameRGB():
    return FeaturesJSONDirPrefix() + 'COILLabelsRGB.json'


def coilFeatureFileNamesFileName():
    return FeaturesJSONDirPrefix() + 'COILFeatureFileNames.json'


def coilFeatureFileNamesFileNameRGB():
    return FeaturesJSONDirPrefix() + 'COILFeatureFileNamesRGB.json'


def coilMissClassifiedImagesDir():
    return ImagesDirPrefix() + 'coilMissClassified/'


def coilMissClassifiedImagesStatFileName():
    return ImagesDirPrefix() + 'coilMissClassified/stat.json'


def getCoilAllPaths():
    return coilImagesDir(), coilImagesMaskDir(), coilFeaturesFileName(), coilLabelsFileName(), coilFeatureFileNamesFileName()


def getCoilAllRGBPaths():
    return coilImagesDir(), coilImagesMaskDir(), coilFeaturesFileNameRGB(), coilLabelsFileNameRGB(), coilFeatureFileNamesFileNameRGB()


def recordedImagesPath():
    return "../../Data/RecordedImages/"


def ALLMAppDataDir():
    return "./../ALM_v2/appData/"


def ALMSetDataDir():
    return "./../ALM_v2/appData/"


def ALMSetMappingDir():
    return "./../ALM_v2/appData/setups/"


def ALMDemoTrainSetDataDir():
    return "./../ALM_v2/appDataDemoTrain/"


def ALMDemoTrainSetMappingDir():
    return "./../ALM_v2/appDataDemoTrain/setups/"


def ALMDefaultClassesFileName():
    return "./../ALM_v2/appData/setups/DefaultClasses.json"


def StatisticsClassificationDir():
    return '../../../../Data/Results/'


def testPath():
    return '/hri/localdisk/home/pioneer/vlosing/OnlineLearning/Images/Tmp/'


def SpeakerSamplesDir():
    return './SpeakerSamples/'


def souza2CDTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'souza/2CDT.data'

def souza2CDTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'souza/2CDT.labels'

def souza4CREV1TrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'souza/4CRE-V1.data'

def souza4CREV1TrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'souza/4CRE-V1.labels'

def souzaGEARS2C2DTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'souza/GEARS_2C_2D.data'

def souzaGEARS2C2DTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'souza/GEARS_2C_2D.labels'

def souzaFG2C2DTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'souza/FG_2C_2D.data'

def souzaFG2C2DTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'souza/FG_2C_2D.labels'

def souza2CHTTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'souza/2CHT.data'

def souza2CHTTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'souza/2CHT.labels'

def keystrokeTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'souza/keystroke.data'

def keystrokeTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'souza/keystroke.labels'

def covTypeTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'covType/covtype.data'

def covTypeTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'covType/covtype.labels'

def rbfFastTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'rbf/rbfFast.data'

def rbfFastTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'rbf/rbfFast.labels'

def hyperplaneFastTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'hyperplane/hyperplaneFast.data'

def hyperplaneFastTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'hyperplane/hyperplaneFast.labels'

def hyperplaneSlowTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'hyperplane/hyperplaneSlow.data'

def hyperplaneSlowTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'hyperplane/hyperplaneSlow.labels'

def hyperplaneSlowLargeTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'hyperplane/hypSlowLarge.data'

def hyperplaneSlowLargeTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'hyperplane/hypSlowLarge.labels'

def rbfFast2DTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'rbf2D/rbfFast.data'

def rbfFast2DTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'rbf2D/rbfFast.labels'

def rbfSlowTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'rbf/rbfSlow.data'

def rbfSlowTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'rbf/rbfSlow.labels'

def rbfSlowLargeTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'rbf/rbfSlowLarge.data'

def rbfSlowLargeTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'rbf/rbfSlowLarge.labels'

def rbfSlow2DTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'rbf2D/rbfSlow.data'

def rbfSlow2DTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'rbf2D/rbfSlow.labels'

def cbConstTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'checkerboard/CBconstant_training_data.csv'

def cbConstTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'checkerboard/CBconstant_training_labels.csv'

def cbSinusTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'checkerboard/CBsinusoidal_training_data.csv'

def cbSinusTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'checkerboard/CBsinusoidal_training_labels.csv'

def rbfAbruptXXLTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'rbf/rbfAbruptXXL.data'

def rbfAbruptXXLTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'rbf/rbfAbruptXXL.labels'

def rbfAbruptSmallTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'rbf/rbfAbruptSmall.data'

def rbfAbruptSmallTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'rbf/rbfAbruptSmall.labels'

def rialtoTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'rialto/rialto.data'

def rialtoTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'rialto/rialto.labels'

def squaresIncrTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'squaresIncr/squaresIncr.data'

def squaresIncrTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'squaresIncr/squaresIncr.labels'

def squaresIncrXXLTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'squaresIncr/squaresIncrXXL.data'

def squaresIncrXXLTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'squaresIncr/squaresIncrXXL.labels'

def chessVirtualTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'chess/chessVirtual.data'

def chessVirtualTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'chess/chessVirtual.labels'

def chessVirtualXXLTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'chess/chessVirtualXXL.data'

def chessVirtualXXLTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'chess/chessVirtualXXL.labels'

def chessIIDXXLTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'chess/chessIIDXXL.data'

def chessIIDXXLTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'chess/chessIIDXXL.labels'

def chessFieldsTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'chess/chessFields.data'

def chessFieldsTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'chess/chessFields.labels'

def allDriftTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'allDrift/allDrift.data'

def allDriftTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'allDrift/allDrift.labels'

def allDriftXXLTrainSamplesPath():
    return nonStationaryFeaturesDirPrefix() + 'allDrift/allDriftXXL.data'

def allDriftXXLTrainLabelsPath():
    return nonStationaryFeaturesDirPrefix() + 'allDrift/allDriftXXL.labels'