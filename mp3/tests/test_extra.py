import unittest, h5py, extra, os
from PIL import Image
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

# TestSequence
class TestStep(unittest.TestCase):
    @weight(1)
    def test_extra_accuracy_above_50(self):
        with h5py.File('solutions_extra.hdf5','r') as f:
            hyps = extra.classify(f['Xtrain'][:],f['Ytrain'][:],f['Xdev'][:],f['Ydev'][:],f['Xtest'][:])
            accuracy = np.count_nonzero(hyps==f['Ytest'][:])/len(f['Ytest'][:])
            self.assertGreater(accuracy, 0.5, msg='Accuracy not greater than 0.5')

    @weight(1)
    def test_extra_accuracy_above_52(self):
        with h5py.File('solutions_extra.hdf5','r') as f:
            hyps = extra.classify(f['Xtrain'][:],f['Ytrain'][:],f['Xdev'][:],f['Ydev'][:],f['Xtest'][:])
            accuracy = np.count_nonzero(hyps==f['Ytest'][:])/len(f['Ytest'][:])
            self.assertGreater(accuracy, 0.52, msg='Accuracy not greater than 0.52')

    @weight(1)
    def test_extra_accuracy_above_54(self):
        with h5py.File('solutions_extra.hdf5','r') as f:
            hyps = extra.classify(f['Xtrain'][:],f['Ytrain'][:],f['Xdev'][:],f['Ydev'][:],f['Xtest'][:])
            accuracy = np.count_nonzero(hyps==f['Ytest'][:])/len(f['Ytest'][:])
            self.assertGreater(accuracy, 0.54, msg='Accuracy not greater than 0.54')

    @weight(1)
    def test_extra_accuracy_above_56(self):
        with h5py.File('solutions_extra.hdf5','r') as f:
            hyps = extra.classify(f['Xtrain'][:],f['Ytrain'][:],f['Xdev'][:],f['Ydev'][:],f['Xtest'][:])
            accuracy = np.count_nonzero(hyps==f['Ytest'][:])/len(f['Ytest'][:])
            self.assertGreater(accuracy, 0.56, msg='Accuracy not greater than 0.56')

    @weight(1)
    def test_extra_accuracy_above_58(self):
        with h5py.File('solutions_extra.hdf5','r') as f:
            hyps = extra.classify(f['Xtrain'][:],f['Ytrain'][:],f['Xdev'][:],f['Ydev'][:],f['Xtest'][:])
            accuracy = np.count_nonzero(hyps==f['Ytest'][:])/len(f['Ytest'][:])
            self.assertGreater(accuracy, 0.58, msg='Accuracy not greater than 0.58')

