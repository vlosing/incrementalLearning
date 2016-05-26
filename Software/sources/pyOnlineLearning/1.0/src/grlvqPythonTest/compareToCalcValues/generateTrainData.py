import json

import numpy


if __name__ == '__main__':
    datasamples = 10000

    mean = (1, 1)
    cov = [[1, -0.5],
           [0.7, 1]]
    x1 = numpy.random.multivariate_normal(mean, cov, (datasamples))

    mean = (5, 5)
    cov = [[1, 0.8],
           [-0.9, 0.1]]
    x2 = numpy.random.multivariate_normal(mean, cov, (datasamples))

    json.encoder.FLOAT_REPR = lambda o: format(o, '.3f')
    f = open('data/traindataClass1.json', 'w')
    json.dump(x1.tolist(), f)
    f.close()

    f = open('data/traindataClass2.json', 'w')
    json.dump(x2.tolist(), f)
    f.close()


