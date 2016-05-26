
import numpy as np

class DefClassifierPlotter(object):
    def __init__(self):
        pass

    def plot(self, classifier, samples, labels, fig, subplot, title, colors, XRange, YRange, emphasizeLastSample=False, pltBoundary=False):
        fig.hold(True)
        # subplot = fig.add_subplot(111, aspect='equal')
        if len(labels) > 0:
            self.plotSamples(subplot, samples, labels, colors)
            if emphasizeLastSample:
                self.plotSamples(subplot, np.array([samples[::-1][0]]), np.array([labels[::-1][0]]), colors, size=60)
            if pltBoundary:
                self.plotDeciscionBoundary(classifier, np.unique(labels), subplot, XRange, YRange)

        subplot.set_title(title, fontsize=20)
        subplot.get_axes().xaxis.set_ticks([])
        subplot.set_xlim([XRange[0], XRange[1]])
        subplot.set_ylim([YRange[0], YRange[1]])
        subplot.get_axes().xaxis.set_ticks([])
        subplot.get_axes().yaxis.set_ticks([])

    def plotSamples(self, subplot, samples, samplesLabels, colors, size=5):
        subplot.scatter(samples[:, 0], samples[:, 1], s=size, c=colors[samplesLabels.astype(int)],
                        edgecolor=colors[samplesLabels.astype(int)])

    def plotDeciscionBoundary(self, classifier, classes, subplot, XRange, YRange):
        preciscion = 200
        u = np.linspace(XRange[0], XRange[1], preciscion)
        v = np.linspace(YRange[0], YRange[1], preciscion)

        z = np.zeros((len(u), len(v)), dtype=np.uint8)
        for i in range(len(u)):
            for j in range(len(v)):
                tdata = np.array([u[i], v[j]])
                result = classifier.predict(tdata)
                z[i, j] = result

        z = z.transpose()
        #assumes classes are numeric and begin from 0
        for i in classes:
            idx = np.where(classes == i)[0][0]
            subplot.contour(u, v, z, [idx, idx + 1], linewidths=[1], colors='black', antialiased=True, alpha=0.5)







