import math
import numpy as np



def getRotationMatrix(rotation):
    sinrot = np.sin(rotation)
    cosrot = np.cos(rotation)
    return np.array([[cosrot, -sinrot], [sinrot, cosrot]])

def randomCov(n):
    Q = np.random.rand(n, n) * 2 - 1
    A = np.dot(Q.T, Q)

    return A

def getChessSquareSamples(row, col, edgeLength, nSamples, sampleRange):
    currentX = sampleRange[0] + col * edgeLength
    currentY = sampleRange[0] - row * edgeLength
    samples = getRectSamples(currentX, currentY, edgeLength, edgeLength, nSamples)
    labels = np.empty([nSamples]).astype(int)
    if (row % 2 == 0 and col % 2 == 0) or (row % 2 == 1 and col % 2 == 1):
        labels.fill(0)
    else:
        labels.fill(1)
    return samples, labels

def getChessSquareSamplesMultiClass(row, col, edgeLength, nSamples, sampleRangeX, sampleRangeY, nTiles=8):
    border=0.
    currentX = sampleRangeX[0] + col * (edgeLength + border*edgeLength)
    currentY = sampleRangeY[0] + row * (edgeLength + border*edgeLength)
    samples = getRectSamples(currentX, currentY, edgeLength, edgeLength, nSamples)
    labels = np.empty([nSamples]).astype(int)
    labels.fill((col+2*row) % nTiles)

    return samples, labels

def getChessVirtual(nSamplesPerField, numValidationSamples, nTiles=8, sampleRange=[0, 1]):
    allSamples = np.empty(shape=(0, 2))
    allLabels = np.empty(shape=(0, 1))
    edgeLength = (sampleRange[1] - sampleRange[0]) / float(nTiles)
    for row in range(nTiles):
        colRange = range(nTiles)
        '''if (row%2 == 0):
            colRange = range(cols)
        else:
            colRange = range(cols-1, -1, -1)'''

        for col in colRange:
            samples, labels = getChessSquareSamplesMultiClass(row, col, edgeLength, nSamplesPerField, sampleRange)
            allSamples = np.vstack([allSamples, samples])
            allLabels = np.append(allLabels, labels)

        for i in range(numValidationSamples):
            rndRow = np.random.randint(0, row+1)
            if rndRow == row:
                rndCol = np.random.randint(0, col+1)
            else:
                rndCol = np.random.randint(0, nTiles)
            samples, labels = getChessSquareSamplesMultiClass(rndRow, rndCol, edgeLength, 1, sampleRange)
            allSamples = np.vstack([allSamples, samples])
            allLabels = np.append(allLabels, labels)
    return allSamples, allLabels

def getChessIID(nSamplesPerField, nTiles=8, sampleRangeX=[0, 1], sampleRangeY = [0, 1]):
    allSamples = np.empty(shape=(0, 2))
    allLabels = np.empty(shape=(0, 1))
    edgeLength = (sampleRangeX[1] - sampleRangeX[0]) / float(nTiles)
    for row in range(nTiles):
        for col in range(nTiles):
            samples, labels = getChessSquareSamplesMultiClass(row, col, edgeLength, nSamplesPerField, sampleRangeX, sampleRangeY)
            allSamples = np.vstack([allSamples, samples])
            allLabels = np.append(allLabels, labels)
    permIndices = np.random.permutation(len(allLabels))
    allSamples = allSamples[permIndices, :]
    allLabels = allLabels[permIndices]
    return allSamples, allLabels

def getMixedDataset(samplesList, labelsList):
    maxLen = 0
    for samples in samplesList:
        if samples.shape[0] > maxLen:
            maxLen = samples.shape[0]

    mixedSamples = np.empty(shape=(0, samplesList[0].shape[1]))
    mixedLabels = np.empty(shape=(0, samplesList[0].shape[1]))

    for i in range(maxLen):
        for samples, labels in zip(samplesList, labelsList):
            if samples.shape[0] > i:
                mixedSamples = np.vstack([mixedSamples, samples[i,:]])
                mixedLabels = np.append(mixedLabels, labels[i])
    return mixedSamples, mixedLabels

def getChessRandomVirtual(nSamplesPerField, numValidationSamples, nTiles=8, repetitions=1, sampleRangeX=[0, 1], sampleRangeY = [0, 1]):
    allSamples = np.empty(shape=(0, 2))
    allLabels = np.empty(shape=(0, 1))
    edgeLength = (sampleRangeX[1] - sampleRangeX[0]) / float(nTiles)
    for i in range(repetitions):
        fieldIndices =[]
        for j in range(nTiles**2):
            if (j) % 4 == 0:
                for i in range(numValidationSamples):
                    rndFieldIdx = np.random.randint(0, nTiles**2)
                    rndRow = rndFieldIdx / nTiles
                    rndCol = rndFieldIdx % nTiles
                    samples, labels = getChessSquareSamplesMultiClass(rndRow, rndCol, edgeLength, 1, sampleRangeX, sampleRangeY)
                    allSamples = np.vstack([allSamples, samples])
                    allLabels = np.append(allLabels, labels)
            while True:
                fieldIdx = np.random.randint(0, nTiles**2)
                if fieldIdx not in fieldIndices:
                    fieldIndices.append(fieldIdx)
                    break
            row = fieldIdx / nTiles
            col = fieldIdx % nTiles
            samples, labels = getChessSquareSamplesMultiClass(row, col, edgeLength, nSamplesPerField, sampleRangeX, sampleRangeY)
            allSamples = np.vstack([allSamples, samples])
            allLabels = np.append(allLabels, labels)
    return allSamples, allLabels


'''def getChessRandomVirtual2(nSamplesPerField, rounds=3, nTiles=8, sampleRange=[0, 1]):
    allSamples = np.empty(shape=(0, 2))
    allLabels = np.empty(shape=(0, 1))
    edgeLength = (sampleRange[1] - sampleRange[0]) / float(nTiles)
    for i in range(rounds):
        fieldIndices =[]
        for j in range(nTiles**2):
            while True:
                fieldIdx = np.random.randint(0, nTiles**2)
                if fieldIdx not in fieldIndices:
                    fieldIndices.append(fieldIdx)
                    break
            row = fieldIdx / nTiles
            col = fieldIdx % nTiles
            samples, labels = getChessSquareSamples2(row, col, edgeLength, nSamplesPerField, sampleRange)
            allSamples = np.vstack([allSamples, samples])
            allLabels = np.append(allLabels, labels)
    return allSamples, allLabels
'''

def rectGradual(numRects, numSamplesPerRect, numConcepts, distBetween = 0, numDimensions=2, sampleRange=[0, 1]):
    canvasWidth = sampleRange[1] - sampleRange[0]
    squareLength = canvasWidth * 0.1

    allSamples = np.empty(shape=(0, numDimensions))
    allLabels = np.empty(shape=(0, 1))

    squarePos = np.zeros(shape=(numRects, numDimensions))
    distBetweenSquares = squareLength * distBetween
    for i in range(numRects):
        squarePos[i, 0] = sampleRange[0] + (numRects-1-i) * (squareLength + distBetweenSquares)

    for i in range(numConcepts):
        newSquarePos = squarePos[::-1]
        conceptSamples = np.empty(shape=(0, numDimensions))
        conceptLabels = np.empty(shape=(0, 1))
        for trans in np.arange(0, 1.0, 0.1):
            concept1Samples = np.empty(shape=(0, numDimensions))
            concept1Labels = np.empty(shape=(0, 1))
            concept2Samples = np.empty(shape=(0, numDimensions))
            concept2Labels = np.empty(shape=(0, 1))

            numSamplesConcept2 = int(np.floor(numSamplesPerRect * trans))
            numSamplesConcept1 = int(numSamplesPerRect - numSamplesConcept2)

            for idx in range(numRects):
                #samples = createNormDistData(numSamplesPerConceptPerCentroid, mean=(X[idx], Y[idx]), var=[[varX[idx], varX2[idx]], [varX2[idx], varY2[idx]]])

                if numSamplesConcept2 > 0:
                    samples = newSquarePos[idx, :] + np.random.rand(numSamplesConcept2, numDimensions) * squareLength
                    labels = idx * np.ones(shape=numSamplesConcept2)
                    concept2Samples = np.vstack([concept2Samples, samples])
                    concept2Labels = np.append(concept2Labels, labels)

                if numSamplesConcept1 > 0:
                    samples = squarePos[idx, :] + np.random.rand(numSamplesConcept1, numDimensions) * squareLength
                    labels = idx * np.ones(shape=numSamplesConcept1)
                    concept1Samples = np.vstack([concept1Samples, samples])
                    concept1Labels = np.append(concept1Labels, labels)
            permIndices = np.random.permutation(len(concept1Labels))
            concept1Samples = concept1Samples[permIndices, :]
            concept1Labels = concept1Labels[permIndices]
            permIndices = np.random.permutation(len(concept2Labels))
            concept2Samples = concept2Samples[permIndices, :]
            concept2Labels = concept2Labels[permIndices]

            conceptSamples = np.vstack([conceptSamples, concept1Samples])
            conceptSamples = np.vstack([conceptSamples, concept2Samples])
            conceptLabels = np.append(conceptLabels, concept1Labels)
            conceptLabels = np.append(conceptLabels, concept2Labels)

        allSamples = np.vstack([allSamples, conceptSamples])
        allLabels = np.append(allLabels, conceptLabels)
        squarePos = newSquarePos
    return allSamples, allLabels

def rbfGradual(numClasses, numCentroidsPerClass, numSamplesPerConceptPerCentroid, numConcepts, numDimensions=2, sampleRange=[0, 1]):
    cov = []
    for i in range(numClasses * numCentroidsPerClass):
        cov.append(randomCov(numDimensions) * 0.0005)

    allSamples = np.empty(shape=(0, numDimensions))
    allLabels = np.empty(shape=(0, 1))
    X = np.random.rand(numClasses * numCentroidsPerClass, numDimensions) * (sampleRange[1] - sampleRange[0]) + sampleRange[0]
    for i in range(numConcepts):
        newX = X[::-1]
        conceptSamples = np.empty(shape=(0, numDimensions))
        conceptLabels = np.empty(shape=(0, 1))
        for trans in np.arange(0, 1.0, 0.1):
            concept1Samples = np.empty(shape=(0, numDimensions))
            concept1Labels = np.empty(shape=(0, 1))
            concept2Samples = np.empty(shape=(0, numDimensions))
            concept2Labels = np.empty(shape=(0, 1))

            numSamplesConcept2 = int(np.floor(numSamplesPerConceptPerCentroid * trans))
            numSamplesConcept1 = int(numSamplesPerConceptPerCentroid - numSamplesConcept2)

            for cls in range(numClasses):
                for centroid in range(numCentroidsPerClass):
                    idx = cls * numCentroidsPerClass + centroid
                    #samples = createNormDistData(numSamplesPerConceptPerCentroid, mean=(X[idx], Y[idx]), var=[[varX[idx], varX2[idx]], [varX2[idx], varY2[idx]]])

                    if numSamplesConcept2 > 0:
                        samples = createNormDistData(numSamplesConcept2, mean=newX[idx, :], var=cov[idx])
                        labels = cls * np.ones(shape=numSamplesConcept2)
                        concept2Samples = np.vstack([concept2Samples, samples])
                        concept2Labels = np.append(concept2Labels, labels)

                    if numSamplesConcept1 > 0:
                        samples = createNormDistData(numSamplesConcept1, mean=X[idx, :], var=cov[idx])
                        labels = cls * np.ones(shape=numSamplesConcept1)
                        concept1Samples = np.vstack([concept1Samples, samples])
                        concept1Labels = np.append(concept1Labels, labels)
            permIndices = np.random.permutation(len(concept1Labels))
            concept1Samples = concept1Samples[permIndices, :]
            concept1Labels = concept1Labels[permIndices]
            permIndices = np.random.permutation(len(concept2Labels))
            concept2Samples = concept2Samples[permIndices, :]
            concept2Labels = concept2Labels[permIndices]

            conceptSamples = np.vstack([conceptSamples, concept1Samples])
            conceptSamples = np.vstack([conceptSamples, concept2Samples])
            conceptLabels = np.append(conceptLabels, concept1Labels)
            conceptLabels = np.append(conceptLabels, concept2Labels)

        allSamples = np.vstack([allSamples, conceptSamples])
        allLabels = np.append(allLabels, conceptLabels)
        X = newX
    return allSamples, allLabels

def rbfAbrupt(numClasses, numCentroidsPerClass, numSamplesPerConceptPerCentroid, numConcepts, numDimensions=2, sampleRange=[0, 1]):
    cov = []
    for i in range(numClasses * numCentroidsPerClass):
        cov.append(randomCov(numDimensions) * 0.0005)


    allSamples = np.empty(shape=(0, numDimensions))
    allLabels = np.empty(shape=(0, 1))
    for i in range(numConcepts):
        X = np.random.rand(numClasses * numCentroidsPerClass, numDimensions) * (sampleRange[1] - sampleRange[0]) + sampleRange[0]



        conceptSamples = np.empty(shape=(0, numDimensions))
        conceptLabels = np.empty(shape=(0, 1))
        for cls in range(numClasses):
            for centroid in range(numCentroidsPerClass):
                idx = cls * numCentroidsPerClass + centroid
                #samples = createNormDistData(numSamplesPerConceptPerCentroid, mean=(X[idx], Y[idx]), var=[[varX[idx], varX2[idx]], [varX2[idx], varY2[idx]]])

                samples = createNormDistData(numSamplesPerConceptPerCentroid, mean=X[idx, :], var=cov[idx])
                labels = cls * np.ones(shape=(numSamplesPerConceptPerCentroid))
                conceptSamples = np.vstack([conceptSamples, samples])
                conceptLabels = np.append(conceptLabels, labels)
        permIndices = np.random.permutation(len(conceptLabels))
        conceptSamples = conceptSamples[permIndices, :]
        conceptLabels = conceptLabels[permIndices]

        allSamples = np.vstack([allSamples, conceptSamples])
        allLabels = np.append(allLabels, conceptLabels)
    return allSamples, allLabels

def rbfAbruptIncreased(numClasses, numCentroidsPerClass, numSamplesPerConceptPerCentroid, numConcepts, numDimensions=2, sampleRange=[0, 1]):
    cov = []
    numCentroids = numClasses * numCentroidsPerClass
    for i in range(numCentroids):
        cov.append(randomCov(numDimensions) * 0.0005)


    allSamples = np.empty(shape=(0, numDimensions))
    allLabels = np.empty(shape=(0, 1))

    X = np.random.rand(numCentroids, numDimensions) * (sampleRange[1] - sampleRange[0]) + sampleRange[0]
    for i in range(numConcepts+1):
        conceptSamples = np.empty(shape=(0, numDimensions))
        conceptLabels = np.empty(shape=(0, 1))
        for cls in range(numClasses):
            for centroid in range(numCentroidsPerClass):
                idx = cls * numCentroidsPerClass + centroid
                samples = createNormDistData(numSamplesPerConceptPerCentroid, mean=X[idx, :], var=cov[idx])
                labels = cls * np.ones(shape=(numSamplesPerConceptPerCentroid))
                conceptSamples = np.vstack([conceptSamples, samples])
                conceptLabels = np.append(conceptLabels, labels)
        permIndices = np.random.permutation(len(conceptLabels))
        conceptSamples = conceptSamples[permIndices, :]
        conceptLabels = conceptLabels[permIndices]

        allSamples = np.vstack([allSamples, conceptSamples])
        allLabels = np.append(allLabels, conceptLabels)
        changedCentroids = numCentroids/float(numConcepts) * (i+1)
        if i < numConcepts:
            X[:changedCentroids, :] = np.random.rand(changedCentroids, numDimensions) * (sampleRange[1] - sampleRange[0]) + sampleRange[0]
    return allSamples, allLabels


def rbfAbrupt2(numClasses, numCentroidsPerClass, numSamplesPerConceptPerCentroid, numConcepts, numDimensions=2, sampleRange=[0, 1]):
    cov = []
    numCentroids = numClasses * numCentroidsPerClass
    for i in range(numCentroids):
        cov.append(randomCov(numDimensions) * 0.0005)


    allSamples = np.empty(shape=(0, numDimensions))
    allLabels = np.empty(shape=(0, 1))

    X = np.random.rand(numCentroids, numDimensions) * (sampleRange[1] - sampleRange[0]) + sampleRange[0]
    for i in range(numConcepts):
        conceptSamples = np.empty(shape=(0, numDimensions))
        conceptLabels = np.empty(shape=(0, 1))
        for cls in range(numClasses):
            for centroid in range(numCentroidsPerClass):
                idx = cls * numCentroidsPerClass + centroid
                samples = createNormDistData(numSamplesPerConceptPerCentroid, mean=X[idx, :], var=cov[idx])
                labels = cls * np.ones(shape=(numSamplesPerConceptPerCentroid))
                conceptSamples = np.vstack([conceptSamples, samples])
                conceptLabels = np.append(conceptLabels, labels)
        permIndices = np.random.permutation(len(conceptLabels))
        conceptSamples = conceptSamples[permIndices, :]
        conceptLabels = conceptLabels[permIndices]

        allSamples = np.vstack([allSamples, conceptSamples])
        allLabels = np.append(allLabels, conceptLabels)
        permutedCentroids = numCentroids
        permIndices = np.random.permutation(numCentroids)
        X = X[permIndices, :]
    return allSamples, allLabels

def rbfAbrupt2Increased(numClasses, numCentroidsPerClass, numSamplesPerConceptPerCentroid, numConcepts, numDimensions=2, sampleRange=[0, 1]):
    cov = []
    numCentroids = numClasses * numCentroidsPerClass
    for i in range(numCentroids):
        cov.append(randomCov(numDimensions) * 0.0005)


    allSamples = np.empty(shape=(0, numDimensions))
    allLabels = np.empty(shape=(0, 1))

    X = np.random.rand(numCentroids, numDimensions) * (sampleRange[1] - sampleRange[0]) + sampleRange[0]
    for i in range(numConcepts):
        conceptSamples = np.empty(shape=(0, numDimensions))
        conceptLabels = np.empty(shape=(0, 1))
        for cls in range(numClasses):
            for centroid in range(numCentroidsPerClass):
                idx = cls * numCentroidsPerClass + centroid
                samples = createNormDistData(numSamplesPerConceptPerCentroid, mean=X[idx, :], var=cov[idx])
                labels = cls * np.ones(shape=(numSamplesPerConceptPerCentroid))
                conceptSamples = np.vstack([conceptSamples, samples])
                conceptLabels = np.append(conceptLabels, labels)
        permIndices = np.random.permutation(len(conceptLabels))
        conceptSamples = conceptSamples[permIndices, :]
        conceptLabels = conceptLabels[permIndices]

        allSamples = np.vstack([allSamples, conceptSamples])
        allLabels = np.append(allLabels, conceptLabels)


        if i != numConcepts-1:
            permIndices = np.random.permutation(int(numCentroids/float(numConcepts) * (i+2)))
            print permIndices
            permIndices = np.append(permIndices, np.arange(i+2, numCentroids))
            X = X[permIndices, :]
    return allSamples, allLabels

def squaresIncr(numRects, iterations, sampleRange, numDimensions, distBetween=0, velocity = 0.005):
    allSamples = np.empty(shape=(0, numDimensions))
    allLabels = np.empty(shape=(0, 1))
    canvasWidth = sampleRange[1] - sampleRange[0]
    squareLength = canvasWidth * 0.05
    #pos is topLeftCorner of square
    distBetweenSquares = squareLength * distBetween
    squarePos = np.zeros(shape=(numRects, numDimensions))
    for i in range(numRects):
        squarePos[i, 0] = sampleRange[0] + (numRects-1-i) * (squareLength + distBetweenSquares)

    direction = 1
    for i in range(iterations):
        samples = np.random.rand(numRects, numDimensions) * squareLength
        allSamples = np.vstack([allSamples, samples + squarePos])
        allLabels = np.append(allLabels, np.arange(numRects))
        squarePos[:, 0] += velocity * squareLength * direction
        if (squarePos[0, 0] + squareLength) >= sampleRange[1] or squarePos[-1, 0] <= sampleRange[0]:
            direction *= -1
    return allSamples, allLabels

def rbfIncr(numClasses, numCentroidsPerClass, numSamplesPerConceptPerCentroid, numConcepts, numDimensions=2, xRange=[0, 1], yRange=[0, 1], velocity = 0.05):
    X = np.random.random(numClasses * numCentroidsPerClass) * (xRange[1] - xRange[0]) + xRange[0]
    Y = np.random.random(numClasses * numCentroidsPerClass) * (yRange[1] - yRange[0]) + yRange[0]

    cov = []
    for i in range(numClasses * numCentroidsPerClass):
        cov.append(randomCov(numDimensions) * 0.0005)

    driftVelocity = np.random.random(numClasses * numCentroidsPerClass) * (xRange[1] - xRange[0]) * velocity
    allSamples = np.empty(shape=(0, numDimensions))
    allLabels = np.empty(shape=(0, 1))
    for i in range(numConcepts):

        conceptSamples = np.empty(shape=(0, numDimensions))
        conceptLabels = np.empty(shape=(0, 1))
        for cls in range(numClasses):
            for centroid in range(numCentroidsPerClass):
                idx = cls * numCentroidsPerClass + centroid
                #samples = createNormDistData(numSamplesPerConceptPerCentroid, mean=(X[idx], Y[idx]), var=[[varX[idx], varX2[idx]], [varX2[idx], varY2[idx]]])

                samples = createNormDistData(numSamplesPerConceptPerCentroid, mean=(X[idx], Y[idx]), var=cov[idx])
                labels = cls * np.ones(shape=(numSamplesPerConceptPerCentroid))
                conceptSamples = np.vstack([conceptSamples, samples])
                conceptLabels = np.append(conceptLabels, labels)
        permIndices = np.random.permutation(len(conceptLabels))
        conceptSamples = conceptSamples[permIndices, :]
        conceptLabels = conceptLabels[permIndices]
        angles = np.random.random(numClasses * numCentroidsPerClass) * 2 * math.pi
        X += np.sin(angles) * driftVelocity
        X = np.minimum(np.maximum(X, xRange[0]), xRange[1])
        Y += np.cos(angles) * driftVelocity
        Y = np.minimum(np.maximum(Y, yRange[0]), yRange[1])

        allSamples = np.vstack([allSamples, conceptSamples])
        allLabels = np.append(allLabels, conceptLabels)

    return allSamples, allLabels

def createNormDistData(samples, mean, var=[[1,0], [0,1]], rotation=0):
    #cov = np.array([[var[0], -0],
    #                [-0, var[1]]])

    cov = var
    if not rotation == 0:
        points = np.random.multivariate_normal((0, 0), cov, samples)
        return np.dot(getRotationMatrix(rotation), points.T).T + mean
    else:
        return np.random.multivariate_normal(mean, cov, samples)

def getManyClasses(numClasses, numParts, numSamples, XRange=[-18.5, 18.5], YRange=[-18.5, 18.5]):
    distance = 9
    offset = 0
    startMean = [XRange[0] + offset + distance, YRange[1] - distance - offset]
    numPerRow = 3
    currRow = 0
    currCol = 0
    dataAll = np.array([])
    labelsAll = np.array([])
    for i in range(numParts):
        for j in range(numClasses):
            currMean = [startMean[0] + currCol * distance, startMean[1] - currRow * distance]
            cov = np.array([[6, 0],
                            [0, 6]])
            data = np.random.multivariate_normal(currMean, cov, numSamples)
            labels = np.empty([numSamples]).astype(int)
            labels.fill(j)
            if len(dataAll) == 0:
                dataAll = data
                labelsAll = labels
            else:
                dataAll = np.vstack([dataAll, data])
                labelsAll = np.hstack([labelsAll, labels])
            if currCol < numPerRow - 1:
                currCol += 1
            else:
                currCol = 0
                currRow += 1

    return dataAll, labelsAll


def getChess(xLeft, yTop, fieldWidth, fieldSamples, rows=8, cols=8):
    dataAll = np.array([])
    labelsAll = np.array([])
    for row in range(rows):
        for col in range(cols):
            currentX = xLeft + col * fieldWidth
            currentY = yTop - row * fieldWidth
            data = getRectSamples(currentX, currentY, fieldWidth, fieldWidth, fieldSamples)
            labels = np.empty([fieldSamples]).astype(int)
            if (row % 2 == 0 and col % 2 == 0) or (row % 2 == 1 and col % 2 == 1):
                labels.fill(0)
            else:
                labels.fill(1)
            if len(dataAll) == 0:
                dataAll = data
                labelsAll = labels
            else:
                dataAll = np.vstack([dataAll, data])
                labelsAll = np.hstack([labelsAll, labels])
    return dataAll, labelsAll


def getLeavesSamples(centerX, centerY, parts, numSamples, varLeaves, dist):
    samples = np.empty(shape=(0, 2))
    for p in range(parts):
        rot = p * (2 * np.pi / parts)
        center = (np.sin(rot) * dist + centerX, np.cos(rot) * dist + centerY)
        samples = np.vstack([samples, createNormDistData(numSamples, mean=center, var=varLeaves, rotation=-rot)])
    return samples


def getNoise(xMin, xMax, yMin, yMax, numSamples, numClasses):
    samples = np.empty([int(numSamples), 2])
    # Generate values on the interval [x_min, x_max]
    samples[:, 0] = xMin + (xMax - xMin) * np.random.uniform(0, 1, int(numSamples))
    # Generate values on the interval [y_min, y_max]
    samples[:, 1] = yMin + (yMax - yMin) * np.random.uniform(0, 1, int(numSamples))

    labels = np.array([])
    labels = np.append(labels, np.random.randint(numClasses, size=int(numSamples)))
    labels = labels.astype(int)
    return samples, labels


def getBorderSet(x, y, radius, numCircle1, numCircle2, numCircle3, portionNoise):
    allSamples, allLabels = getCircles(x, y, radius, numCircle1, numCircle2, numCircle3, 0)

    numNoise = portionNoise * len(allSamples)

    xMin = -5
    xMax = 20
    yMin = -15
    yMax = 5

    samples, labels = getNoise(xMin, xMax, yMin, yMax, numNoise, 3)

    allSamples = np.vstack([allSamples, samples])
    allLabels = np.append(allLabels, labels)
    allLabels = allLabels.astype(int)
    return allSamples, allLabels


def getGLVQTest(x, y, x2, y2, x3, y3, var):
    allLabels = np.array([])
    allSamples = np.empty(shape=(0, 2))
    numSamples = 30
    gauss1 = createNormDistData(numSamples, mean=(x, y), var=var)

    gauss2 = createNormDistData(numSamples, mean=(x2, y2), var=var)
    gauss3 = createNormDistData(numSamples, mean=(x3, y3), var=var)

    allSamples = np.vstack([allSamples, gauss1])
    allSamples = np.vstack([allSamples, gauss2])
    allSamples = np.vstack([allSamples, gauss3])

    allLabels = np.append(allLabels, 0 * np.ones(numSamples))
    allLabels = np.append(allLabels, 1 * np.ones(numSamples))
    allLabels = np.append(allLabels, 2 * np.ones(numSamples))

    allLabels = allLabels.astype(int)
    return allSamples, allLabels


def getRelSimSet(x, y, x2, y2, var):
    allLabels = np.array([])
    allSamples = np.empty(shape=(0, 2))
    numSamples = 5000
    gauss1 = createNormDistData(numSamples, mean=(x, y), var=var)

    gauss2 = createNormDistData(numSamples, mean=(x2, y2), var=var)

    allSamples = np.vstack([allSamples, gauss1])
    allSamples = np.vstack([allSamples, gauss2])

    allLabels = np.append(allLabels, 0 * np.ones(numSamples))
    allLabels = np.append(allLabels, np.ones(numSamples))

    allLabels = allLabels.astype(int)
    return allSamples, allLabels


def getToySet2(x, y, numLeavesSamples, numLeavesParts, numBlossom, varLeaves, varBlossom, dist, startClass):
    allLabels = np.array([])
    allSamples = np.empty(shape=(0, 2))

    leavesSamples = getLeavesSamples(x, y, numLeavesParts, numLeavesSamples, varLeaves, dist)

    blossomSamples = createNormDistData(numBlossom, mean=(x, y), var=varBlossom)

    allSamples = np.vstack([allSamples, blossomSamples])
    allSamples = np.vstack([allSamples, leavesSamples])

    allLabels = np.append(allLabels, startClass * np.ones(numBlossom))

    for i in range(numLeavesParts):
        label = (i % 2) + (startClass + 1)
        allLabels = np.append(allLabels, label * np.ones(numLeavesSamples))

    allLabels = allLabels.astype(int)
    return allSamples, allLabels


def getNoiseSet(xStart, yStart, rectSize, numRect1, numRect2, startClass):
    allLabels = np.array([])
    allSamples = np.empty(shape=(0, 2))
    rect1 = getRectSamples(xStart, yStart, rectSize, rectSize, numRect1)
    rect2 = getRectSamples(xStart, yStart, rectSize, rectSize, numRect2)
    allSamples = np.vstack([allSamples, rect1])
    allSamples = np.vstack([allSamples, rect2])
    allLabels = np.append(allLabels, startClass * np.ones(numRect1))
    allLabels = np.append(allLabels, (startClass + 1) * np.ones(numRect2))
    allLabels = allLabels.astype(int)
    return allSamples, allLabels


def getOverlapSet(xStart, yStart, rectSize, offsetX, offsetY, numRect, numRect2, startClass):
    allLabels = np.array([])
    allSamples = np.empty(shape=(0, 2))

    x = xStart
    y = yStart
    rect1 = getRectSamples(x, y, rectSize, rectSize, numRect)
    x += rectSize
    rect2 = getRectSamples(x, y, rectSize, rectSize, numRect)

    x += rectSize + offsetX
    rect3 = getRectSamples(x, y, rectSize, rectSize, numRect)
    x += rectSize / 2
    rect4 = getRectSamples(x, y, rectSize, rectSize, numRect)

    x += rectSize + offsetX
    rect5 = getRectSamples(x, y, rectSize, rectSize, numRect)

    rect6 = getRectSamples(x, y, rectSize, rectSize, numRect)
    allSamples = np.vstack([allSamples, rect1])
    allSamples = np.vstack([allSamples, rect2])
    allSamples = np.vstack([allSamples, rect3])
    allSamples = np.vstack([allSamples, rect4])
    allSamples = np.vstack([allSamples, rect5])
    allSamples = np.vstack([allSamples, rect6])
    allLabels = np.append(allLabels, startClass * np.ones(numRect))
    allLabels = np.append(allLabels, (startClass + 1) * np.ones(numRect))
    allLabels = np.append(allLabels, startClass * np.ones(numRect))
    allLabels = np.append(allLabels, (startClass + 1) * np.ones(numRect))
    allLabels = np.append(allLabels, startClass * np.ones(numRect))
    allLabels = np.append(allLabels, (startClass + 1) * np.ones(numRect))

    x = xStart
    y = yStart - rectSize - offsetY
    rect7 = getRectSamples(x, y, rectSize, rectSize, numRect2)
    x += rectSize
    rect8 = getRectSamples(x, y, rectSize, rectSize, numRect)

    x += rectSize + offsetX
    rect9 = getRectSamples(x, y, rectSize, rectSize, numRect2)
    x += rectSize / 2
    rect10 = getRectSamples(x, y, rectSize, rectSize, numRect)

    x += rectSize + offsetX
    rect11 = getRectSamples(x, y, rectSize, rectSize, numRect2)
    rect12 = getRectSamples(x, y, rectSize, rectSize, numRect)

    allSamples = np.vstack([allSamples, rect7])
    allSamples = np.vstack([allSamples, rect8])
    allSamples = np.vstack([allSamples, rect9])
    allSamples = np.vstack([allSamples, rect10])
    allSamples = np.vstack([allSamples, rect11])
    allSamples = np.vstack([allSamples, rect12])


    allLabels = np.append(allLabels, (startClass + 2) * np.ones(numRect2))
    allLabels = np.append(allLabels, (startClass + 3) * np.ones(numRect))
    allLabels = np.append(allLabels, (startClass + 2) * np.ones(numRect2))
    allLabels = np.append(allLabels, (startClass + 3) * np.ones(numRect))
    allLabels = np.append(allLabels, (startClass + 2) * np.ones(numRect2))
    allLabels = np.append(allLabels, (startClass + 3) * np.ones(numRect))
    allLabels = allLabels.astype(int)

    return allSamples, allLabels


def getCompleteToySet(numLeaves, numLeavesParts, numBlossom, numRect, numRect2, numCircle1, numCircle2, numCircle3,
                      portionNoise):
    allLabels = np.array([])
    allSamples = np.empty(shape=(0, 2))

    [samples, labels] = getBorderSet(-10, 9, 3, numCircle1, numCircle2, numCircle3, 0.0)  # gMax 0.1 14 proto 2adds
    allSamples = np.vstack([allSamples, samples])
    allLabels = np.append(allLabels, labels)
    startClass = np.max(allLabels) + 1
    samples, labels = getToySet2(9, 9, numLeaves, numLeavesParts, numBlossom, (0.15, 1.5), (1.5, 1.5), 6, startClass)
    allSamples = np.vstack([allSamples, samples])
    allLabels = np.append(allLabels, labels)
    startClass = np.max(allLabels) + 1
    samples, labels = getOverlapSet(-15, -10, 6, 1, 2, numRect, numRect2, startClass)
    allSamples = np.vstack([allSamples, samples])
    allLabels = np.append(allLabels, labels)

    numClasses = np.max(allLabels)
    xMin = np.min(allSamples[:, 0])
    xMax = np.max(allSamples[:, 0])
    yMin = np.min(allSamples[:, 1])
    yMax = np.max(allSamples[:, 1])
    numNoise = portionNoise * len(allSamples)
    samples, labels = getNoise(xMin, xMax, yMin, yMax, numNoise, numClasses)
    allSamples = np.vstack([allSamples, samples])
    allLabels = np.append(allLabels, labels)
    allLabels = allLabels.astype(int)
    return allSamples, allLabels


def getCircles(x, y, radius, numCircle1, numCircle2, numCircle3, startClass):
    allLabels = np.array([])
    allSamples = np.empty(shape=(0, 2))
    allLabels = allLabels.astype(int)

    rad = 0
    # circle1 = getCircleSamples(x, y, rad, rad+radius-0.01, numCircle1)
    circle1 = getCircleUniform(x, y, rad, rad + radius - 0.01, numCircle1)
    rad += radius
    circle2 = getCircleUniform(x, y, rad, rad + radius - 0.01, numCircle2)
    # circle2 = getCircleSamples(x, y, rad, rad+radius-0.01, numCircle2)
    rad += radius
    circle3 = getCircleUniform(x, y, rad, rad + radius - 0.01, numCircle3)
    #circle3 = getCircleSamples(x, y, rad, rad+radius-0.01, numCircle3)

    allSamples = np.vstack([allSamples, circle1])
    allSamples = np.vstack([allSamples, circle2])
    allSamples = np.vstack([allSamples, circle3])

    allLabels = np.append(allLabels, startClass * np.ones(len(circle1)))
    allLabels = np.append(allLabels, (startClass + 1) * np.ones(len(circle2)))
    allLabels = np.append(allLabels, (startClass + 2) * np.ones(len(circle3)))
    return allSamples, allLabels


def getRectSamples(x, y, width, height, numSamples):
    data = np.empty([numSamples, 2])

    data[:, 0] = x + width * np.random.uniform(0, 1, numSamples)
    data[:, 1] = y + height * np.random.uniform(0, 1, numSamples)
    return data


def getCircleUniform(x, y, radMin, radMax, numSamples):
    data = np.empty([numSamples * 5, 2])
    data[:, 0] = x - radMax + 2 * radMax * np.random.uniform(0, 1, numSamples * 5)
    data[:, 1] = y - radMax + 2 * radMax * np.random.uniform(0, 1, numSamples * 5)

    distances = np.linalg.norm(data - np.array([x, y]), axis=1)

    delIndices = np.where((distances > radMax) | (distances < radMin))[0]
    data = np.delete(data, delIndices, 0)

    numToDelete = len(data) - numSamples
    data = np.delete(data, np.arange(numToDelete), 0)

    return data


def getCircleSamples(x, y, radMin, radMax, numSamples):
    # class 1
    phiMin = 0
    phiMax = 2. * math.pi

    # polar coordinates
    rad = radMin + (radMax - radMin) * np.random.uniform(0, 1, numSamples)
    phi = phiMin + (phiMax - phiMin) * np.random.uniform(0, 1, numSamples)

    data = np.empty([numSamples, 2])

    data[:, 0] = np.transpose(x + np.multiply(rad, np.sin(phi)))
    data[:, 1] = np.transpose(y + np.multiply(rad, np.cos(phi)))
    return data


def getSineSamples(xMin, xMax, yMin, yMax, numSamples):
    # Generate values on the interval [x_min, x_max]
    samples = np.empty([numSamples, 2])
    samples[:, 0] = xMin + (xMax - xMin) * np.random.uniform(0, 1, numSamples)

    # Generate values on the interval [y_min, y_max]
    b = yMin + (yMax - yMin) * np.random.uniform(0, 1, numSamples)
    samples[:, 1] = np.sin(samples[:, 0]) + b

    return samples

