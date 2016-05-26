// SH: made some corrections to networklist routines so that python example works / but some functions are still wrong, e.g. networklist_remove

#include "Python.h"
//#include "/usr/lib64/python2.7/site-packages/numpy/core/include/numpy/arrayobject.h"
#include "arrayobject.h"
#include <string.h>
//#include "classifier.h"
#include "hyperparameters.h"
//#include "experimenter.h"
#include <Eigen/Core>
#include <Eigen/Array>
#include "online_rf.h"

OnlineRF* orf = NULL;
Hyperparameters* hp = NULL;
VectorXd* minFeatRange = NULL;
VectorXd* maxFeatRange = NULL;


void fitORFintern(OnlineRF* model, vector<Sample>& samples, int numClasses, vector<int>& predictedTrainLabels){
	for (unsigned i = 0; i < samples.size(); i++) {
		Result result(numClasses);
		model->eval(samples[i], result);
		predictedTrainLabels.push_back(result.prediction);
		model->update(samples[i]);
	}
}

void fitORF(OnlineRF* model, double* data, int* label, int* predictedLabels, const int& numSamples, const int& numFeatures, const int& numClasses){
	vector<Sample> samples;
	for (int i=0; i< numSamples; i++){
		const double* sampledata = &data[numFeatures * i];
        Sample sample;
        sample.x = VectorXd(numFeatures);
        sample.w = 1.0;
        sample.y = label[i];
        for (int nFeat = 0; nFeat < numFeatures; nFeat++) {
        	sample.x(nFeat) = sampledata[nFeat];
		}
        samples.push_back(sample);
	}
	vector<int> predictedTrainLabels;
	fitORFintern(model, samples, numClasses, predictedTrainLabels);
	for (int i=0; i< predictedTrainLabels.size(); i++){
		predictedLabels[i] = predictedTrainLabels[i];
	}
}

std::vector<int> predictORF_intern(OnlineRF * model, vector<Sample>& samples, const int& numClasses) {
	vector<int> results;
	for (int nSamp = 0; nSamp < samples.size(); nSamp++) {
		Result result(numClasses);
		model->eval(samples[nSamp], result);
		results.push_back(result.prediction);
	}
	return results;
}

void predictORF(OnlineRF * model, const double* data, int* labels,  const int& numSamples, const int& numFeatures, const int& numClasses) {
	vector<Sample> samplesVec;
	for (int i=0; i< numSamples; i++){
		const double* sampledata = &data[numFeatures * i];
        Sample sample;
        sample.x = VectorXd(numFeatures);
        sample.w = 1.0;
        sample.y = labels[i];
        for (int nFeat = 0; nFeat < numFeatures; nFeat++) {
        	sample.x(nFeat) = sampledata[nFeat];
		}
        samplesVec.push_back(sample);
	}
	vector<int> labelsVec = predictORF_intern(model, samplesVec, numClasses);
	for (int i=0; i< labelsVec.size(); i++){
		labels[i] = labelsVec[i];
	}
}
OnlineRF* initORFintern(const int& numTrees, const int& numRandomTests, const int& maxDepth, const int& counterThreshold, const int& numClasses, const int& numFeatures)
{
	if (orf != NULL){
		delete orf;
		delete hp;
		delete minFeatRange;
		delete maxFeatRange;
		orf = NULL;
		hp = NULL;
		minFeatRange = NULL;
		maxFeatRange = NULL;
	}
	//XXVL free before if set
	hp = new Hyperparameters(numTrees, numRandomTests, maxDepth, counterThreshold);
	minFeatRange = new VectorXd(numFeatures);
	maxFeatRange = new VectorXd(numFeatures);
    for (int nFeat = 0; nFeat < numFeatures; nFeat++) {
    	(*minFeatRange)(nFeat) = std::numeric_limits<double>::max();
    	(*maxFeatRange)(nFeat) = std::numeric_limits<double>::min();
    }
	orf = new OnlineRF(*hp, numClasses, numFeatures, *minFeatRange, *maxFeatRange);

	return orf;
}

static PyObject *py_initORF(PyObject *self, PyObject *args) {
	int numTrees, numRandomTests, maxDepth, counterThreshold, numClasses, numFeatures;

	if (!PyArg_ParseTuple(args, "iiiiii", &numTrees, &numRandomTests,
			&maxDepth, &counterThreshold, &numClasses,
			&numFeatures))
		return NULL;
	initORFintern(numTrees, numRandomTests, maxDepth, counterThreshold, numClasses, numFeatures);
	return PyInt_FromLong(true);
}
PyDoc_STRVAR(py_initORF__doc__,
		"name, dimensionality, type, learnrate_per_node, # of classes, prototypes per class, trainsteps, (opt) random init, (opt) thread per dimensionality, just select the x first dimensions for euclidean distance - inital");


static PyObject *py_getComplexity(PyObject *self, PyObject *args) {
	int complexity = orf->getComplexity();
	return PyInt_FromLong(complexity);
}

PyDoc_STRVAR(py_getComplexity__doc__, "name, protonr");


static PyObject *py_getComplexityNumParametric(PyObject *self, PyObject *args) {
	int complexity = orf->getComplexityNumParamMetric();
	return PyInt_FromLong(complexity);
}
PyDoc_STRVAR(py_getComplexityNumParametric__doc__, "name, protonr");


static PyObject *py_fitORF(PyObject *self, PyObject *args) {
		PyObject *data, *label;
	int numOfClasses;

	if (!PyArg_ParseTuple(args, "OOi", &data, &label, &numOfClasses))
		return NULL;

	//create c-arrays:
	PyArrayObject *mat_data;
	mat_data = (PyArrayObject *) PyArray_ContiguousFromObject(data,
			PyArray_DOUBLE, 1, 2);
	if ((NULL == mat_data)
			|| (((PyArrayObject *) data)->descr->type_num != NPY_DOUBLE)) {
		printf(
				"Error: data must be a 2-D double mat, each row is one datasample\n");
		return NULL;
	}

	double *carr_data = (double*) (mat_data->data);

	npy_intp carr_data_rows = mat_data->dimensions[0];
	npy_intp carr_data_cols = mat_data->dimensions[1];

	if (mat_data->nd == 1) {
		carr_data_cols = carr_data_rows;
		carr_data_rows = 1;
	}

	PyArrayObject *mat_labels;
	mat_labels = (PyArrayObject *) PyArray_ContiguousFromObject(label,
			PyArray_INT, 1, 1);
	if ( NULL == mat_labels) {
		printf("Error: label must be a 1-D double mat\n");
		return NULL;
	}
	int *carr_label = (int*) (mat_labels->data);
	npy_intp carr_label_rows = mat_labels->dimensions[0];

	if (carr_data_rows != carr_label_rows) {
		printf(
				"Error: data's number of rows must equal the number of labels!\n");
		return NULL;
	}


	PyArrayObject *predLabel = (PyArrayObject *) PyArray_SimpleNew(1,
			&carr_label_rows, PyArray_INT);
	PyArrayObject *mat_predLabel;
	mat_predLabel = (PyArrayObject *) PyArray_ContiguousFromObject(
			(PyObject* )predLabel, PyArray_INT, 1, 1);
	if (NULL == mat_labels) {
		printf("Error: label must be a 1-D double mat\n");
		return NULL;
	}

	int *carr_predLabel = (int*) (mat_predLabel->data);

	vector<Sample> samples;
	for (int i=0; i< carr_label_rows; i++){
		const double* sampledata = &carr_data[carr_data_cols * i];
        Sample sample;
        sample.x = VectorXd(carr_data_cols);
        sample.w = 1.0;
        for (int nFeat = 0; nFeat < carr_data_cols; nFeat++) {
        	sample.x(nFeat) = sampledata[nFeat];
		}
        samples.push_back(sample);
	}

    for (int nFeat = 0; nFeat < carr_data_cols; nFeat++) {
        for (int nSamp = 1; nSamp < carr_label_rows; nSamp++) {
            if (samples[nSamp].x(nFeat) < (*minFeatRange)(nFeat)) {
            	(*minFeatRange)(nFeat) = samples[nSamp].x(nFeat);
            }
            if (samples[nSamp].x(nFeat) > (*maxFeatRange)(nFeat)) {
            	(*maxFeatRange)(nFeat) = samples[nSamp].x(nFeat);
            }
        }
    }

	fitORF(orf, carr_data, carr_label, carr_predLabel, carr_label_rows, carr_data_cols, numOfClasses);
	Py_DECREF(mat_data);
	Py_DECREF(mat_labels);
	PyObject *result = Py_BuildValue("O", mat_predLabel);
	Py_DECREF(mat_predLabel);
	Py_DECREF(predLabel);
	return result;

}

PyDoc_STRVAR(py_fitORF__doc__,
		"name, data 2D-arr, label 1D-arr, adapt_metric_weights. Returns medium weight shift");

static PyObject *py_predictORF(PyObject *self, PyObject *args) {
	PyObject *data;
	int numClasses;
	if (!PyArg_ParseTuple(args, "Oi", &data, &numClasses))
		return NULL;


	//create c-arrays:
	PyArrayObject *mat_data;
	mat_data = (PyArrayObject *) PyArray_ContiguousFromObject(data,
			PyArray_DOUBLE, 1, 2);
	if ((NULL == mat_data)
			|| (((PyArrayObject *) data)->descr->type_num != NPY_DOUBLE)) {
		printf(
				"Error: data must be a 2-D double mat, each row is one datasample\n");
		return NULL;
	}
	double *carr_data = (double*) (mat_data->data);
	npy_intp carr_data_rows = mat_data->dimensions[0];
	npy_intp carr_data_cols = mat_data->dimensions[1];

	if (mat_data->nd == 1) {
		carr_data_cols = carr_data_rows;
		carr_data_rows = 1;
	}

	PyArrayObject *label = (PyArrayObject *) PyArray_SimpleNew(1,
			&carr_data_rows, PyArray_INT);
	PyArrayObject *mat_labels;
	mat_labels = (PyArrayObject *) PyArray_ContiguousFromObject(
			(PyObject* )label, PyArray_INT, 1, 1);
	if (NULL == mat_labels) {
		printf("Error: indices must be a 1-D double mat\n");
		return NULL;
	}
	int *carr_label = (int*) (mat_labels->data);
	int carr_label_rows = mat_labels->dimensions[0];

	if (carr_data_rows != carr_label_rows) {
		printf(
				"Error: data's number of rows must equal the number of labels!\n");
		return NULL;
	}
	predictORF(orf, carr_data, carr_label, carr_data_rows, carr_data_cols, numClasses);

	Py_DECREF(mat_data);
	//Py_DECREF(label);
	PyObject *result = Py_BuildValue("O", mat_labels);
	Py_DECREF(mat_labels);
	Py_DECREF(label);
	return result;

}
PyDoc_STRVAR(py_predictOF__doc__,
		"name, data 2D-arr. Returns label result matrix");



/* The module doc string */
PyDoc_STRVAR(ORF__doc__, "Mandelbrot point evalutation kernel");



/* A list of all the methods defined by this module. */
/* "iterate_point" is the name seen inside of Python */
/* "py_iterate_point" is the name of the C function handling the Python call */
/* "METH_VARGS" tells Python how to call the handler */
/* The {NULL, NULL} entry indicates the end of the method definitions */
static PyMethodDef ORF_methods[] = { {"initORF", py_initORF, METH_VARARGS, py_initORF__doc__}, { "getComplexity", py_getComplexity,
		METH_VARARGS, py_getComplexity__doc__ }, { "getComplexityNumParametric", py_getComplexityNumParametric,
				METH_VARARGS, py_getComplexityNumParametric__doc__ },
				{ "fitORF", py_fitORF, METH_VARARGS, py_fitORF__doc__ },
				{ "predictORF",
				py_predictORF, METH_VARARGS, py_predictOF__doc__},
				{ NULL, NULL } /* sentinel */
};

/* When Python imports a C module named 'X' it loads the module */
/* then looks for a method named "init"+X and calls it.  Hence */
/* for the module "mandelbrot" the initialization function is */
/* "initmandelbrot".  The PyMODINIT_FUNC helps with portability */
/* across operating systems and between C and C++ compilers */
PyMODINIT_FUNC initliborfPythonIntf(void) {
	import_array()
		/* There have been several InitModule functions over time */
	PyObject* myclass = Py_InitModule3("liborfPythonIntf", ORF_methods,
			ORF__doc__);

}
