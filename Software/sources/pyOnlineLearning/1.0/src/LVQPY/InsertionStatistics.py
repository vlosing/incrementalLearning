__author__ = 'viktor'

class InsertionStatistics(object):
    def __init__(self):
        self._insertionCount = 0
        self._triedInsertionCount = 0

        self._totalInsertionWindowDeltaCost = 0
        self._lastInsertionWindowDeltaCost = 0

        self._totalTriedInsertionWindowDeltaCost = 0
        self._lastTriedInsertionWindowDeltaCost = 0

        self._lastInsertionWindowPrototypeCount = 0
        self._totalInsertionWindowPrototypeCount = 0

        self._lastTriedInsertionWindowPrototypeCount = 0
        self._totalTriedInsertionWindowPrototypeCount = 0

        self._totalInsertionWindowPrototypeDensity = 0
        self._lastInsertionWindowPrototypeDensity = 0

        self._totalTriedInsertionWindowPrototypeDensity = 0
        self._lastTriedInsertionWindowPrototypeDensity = 0
