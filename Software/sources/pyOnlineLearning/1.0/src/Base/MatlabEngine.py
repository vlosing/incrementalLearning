__author__ = 'viktor'
import matlab.engine


class MatlabEngine:
    engine = None
    @staticmethod
    def getEng():
        if MatlabEngine.engine is None:
            print 'start'
            MatlabEngine.engine = matlab.engine.start_matlab()
            print 'got it'
        return MatlabEngine.engine

    @staticmethod
    def stop():
        if MatlabEngine.engine is not None:
            print 'stop'
            MatlabEngine.engine.quit()
            MatlabEngine.engine = None
            print 'stopped'


