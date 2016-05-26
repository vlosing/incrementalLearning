import numpy as np
import os
from Base import Paths

class JunctionGPSInformation(object):
    def __init__(self, route):
        self.route = route
        self.junctionCenters, self.stopLines, self.trafficLights = self.getByUserAndDirection(self.route.driverName, self.route.direction)

    def getByUserAndDirection(self, driverName, direction):
        if driverName == 'martina' and direction == 'fromHonda':
            junctionCenter = np.array([[50.083648, 8.839027],
                                     [50.080915, 8.837988],
                                     [50.079778, 8.845376],
                                     [50.084624, 8.846949],
                                     [50.096896, 8.804485],
                                     [50.093521, 8.798956],
                                     [50.094756, 8.796385],
                                     [50.096944, 8.791619],
                                     [50.097285, 8.790963],
                                     [50.098322, 8.788719],
                                     [50.099325, 8.785505],
                                     [50.100419, 8.782010],
                                     [50.100575, 8.781535],
                                     [50.102002, 8.776856],
                                     [50.102474, 8.775339],
                                     [50.101523, 8.770383],
                                     [50.101007, 8.767875],
                                     [50.102895, 8.767279],
                                     [50.103115, 8.768733]])

            stopLines = np.array([[50.083661, 8.839030],
                         #[50.081064, 8.838034],
                          [50.081094, 8.837992],
                         [50.079734, 8.845207],
                         [50.084425, 8.846918],
                         [50.097049, 8.804614],
                         [50.093602, 8.799103],
                         [50.094751, 8.796467],
                         [50.096829, 8.791952],
                         [50.097199, 8.791175],
                         [50.098178, 8.789026],
                         [50.099274, 8.785806],
                         [50.100376, 8.782173],
                         [50.100518, 8.781726],
                         [50.101994, 8.776995],
                         [50.102474, 8.775339],
                         [50.101536, 8.770520],
                         [50.101032, 8.768005],
                         [50.102815, 8.767288],
                         [50.103115, 8.768733]])

            trafficLights = np.array([[-1,-1],
                             [50.081064, 8.838034],
                             [50.079734, 8.845207],
                             [50.084425, 8.846918],
                             [50.097049, 8.804614],
                             [50.093602, 8.799103],
                             [50.094751, 8.796467],
                             [50.096829, 8.791952],
                             [50.097199, 8.791175],
                             [50.098178, 8.789026],
                             [50.099274, 8.785806],
                             [50.100376, 8.782173],
                             [50.100518, 8.781726],
                             [50.101994, 8.776995],
                             [-1,-1],
                             [50.101536, 8.770520],
                             [50.101032, 8.768005],
                             [-1,-1],
                             [-1,-1]])

            return junctionCenter, stopLines, trafficLights
            #return np.atleast_2d(junctionCenter[13]), np.atleast_2d(stopLines[13]), np.atleast_2d(trafficLights[13])
        raise ValueError('unknown username %s or direction %s' % (driverName, direction))

class JunctionLabelData(object):
    def __init__(self, route, streamDate):
        self.route = route
        self.streamDate = streamDate
        self.filePath = route.featuresDir + '/labeled/junctionFeatures_' + streamDate + '.csv'
        self.data = np.loadtxt(self.filePath, delimiter=',', skiprows=1)

class Route(object):
    def __init__(self, driverName, direction, prefix=Paths.trackAddictDir()):
        self.driverName = driverName
        self.direction = direction
        self.prefix = prefix
        self.featuresDir = os.path.join(prefix, driverName + '/processed/featureRelevant/' + direction + '/')

