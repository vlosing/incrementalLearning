import AlmClib


# setuup of camera control (exposure, gain)
def initCamCtrl(v4lHdl):
    # # define camera control parameters
    ## target value for mean gray value of recognized grass area
    targetMeanGray = 0.35
    ## initial exposure
    startExposure = 20
    ## initial gain
    startGain = 0
    cc = AlmClib.CamCtrlC_new()
    AlmClib.CamCtrlC_init(cc, v4lHdl, startExposure, startGain)

    # set camera control parameters for PSEye
    AlmClib.CamCtrlC_setMinExposure(cc, 5)
    AlmClib.CamCtrlC_setMaxExposure(cc, 255)  # rbargain constrain, normal 255
    AlmClib.CamCtrlC_setMinGain(cc, 0)
    AlmClib.CamCtrlC_setMaxGain(cc, 30)  # rbargain constrain, normal 15
    AlmClib.CamCtrlC_setMaxExposureChange(cc, 3)  # rbargain constrain, normal 50
    AlmClib.CamCtrlC_setMinFrac(cc, 0.01)
    AlmClib.CamCtrlC_setMinDiff(cc, 0.04)
    AlmClib.CamCtrlC_setTargetMeanGray(cc, targetMeanGray)
    AlmClib.CamCtrlC_setMaskWeight(cc, 0.01)
    #  AlmClib.CamCtrlC_setDampFactor(        cc, 0.00)
    cc.m_expDampFactor = 0.00
    return cc