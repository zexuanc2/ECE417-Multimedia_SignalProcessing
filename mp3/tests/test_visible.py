import unittest, h5py, submitted, os
from PIL import Image
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

# TestSequence
class TestStep(unittest.TestCase):
    @weight(6.25)
    def test_dataset_mean(self):
        with h5py.File('solutions.hdf5','r') as h5:
            mu = submitted.todo_dataset_mean(h5['Xtrain'][:])
            self.assertAlmostEqual(
                np.average(np.abs(mu-h5['mu'][:])), 0, places=3,
                msg='\*** sum(abs(dataset_mean)) is off by an average of more than 0.001'
            )

    @weight(6.25)
    def test_center_datasets(self):
        with h5py.File('solutions.hdf5','r') as h5:
            ctrain,cdev,ctest=submitted.todo_center_datasets(h5['Xtrain'][:],h5['Xdev'][:],
                                                             h5['Xtest'][:], h5['mu'][:])
            self.assertAlmostEqual(
                np.average(np.abs(ctrain-h5['ctrain'][:])), 0, places=3,
                msg='\*** todo_center_datasets ctrain is off by an average of more than 0.001'
            )

    @weight(6.25)
    def test_find_transform(self):
        with h5py.File('solutions.hdf5','r') as h5:
            transform, Lambda = submitted.todo_find_transform(h5['ctrain'][:])
            self.assertAlmostEqual(
                np.average(np.abs(Lambda-h5['Lambda'][:])), 0, places=3,
                msg='\*** todo_find_transform Lambda is off by an average of more than 0.001'
            )

    @weight(6.25)
    def test_transform_datasets(self):
        with h5py.File('solutions.hdf5','r') as h5:
            ttrain, tdev, ttest = submitted.todo_transform_datasets(h5['ctrain'][:], h5['cdev'][:],
                                                                    h5['ctest'][:], h5['V'][:])
            self.assertAlmostEqual(
                np.average(np.abs(ttrain-h5['ttrain'][:])), 0, places=3,
                msg='\*** todo_transform_datasets ttrain is off by an average of more than 0.001'
            )

    @weight(6.25)
    def test_distances(self):
        with h5py.File('solutions.hdf5','r') as h5:
            ttrain = h5['ttrain'][:]
            Dtraindev = submitted.todo_distances(ttrain, h5['tdev'][:], ttrain.shape[1])
            self.assertAlmostEqual(
                np.average(np.abs(Dtraindev-h5['Dtraindev'][:])), 0, places=3,
                msg='\*** todo_distances Dtraindev is off by an average of more than 0.001'
            )

    @weight(6.25)
    def test_nearest_neighbor(self):
        with h5py.File('solutions.hdf5','r') as h5:
            hypsfull = submitted.todo_nearest_neighbor(h5['Ytrain'][:], h5['Dtraindev'][:])
            self.assertAlmostEqual(
                np.average(np.abs(hypsfull-h5['hypsfull'][:])), 0, places=3,
                msg='\*** todo_nearest_neighbor(Ytrain,Dtraindev) is off by an average of more than 0.001'
            )

    @weight(6.25)
    def test_compute_accuracy(self):
        with h5py.File('solutions.hdf5','r') as h5:
            accuracyfull, confusionfull = submitted.todo_compute_accuracy(h5['Ydev'][:],h5['hypsfull'][:])
            self.assertAlmostEqual(
                np.average(np.abs(confusionfull-h5['confusionfull'][:])), 0, places=3,
                msg='\*** todo_compute_accuracy(Ydev,hypsfull) is off by an average of more than 0.001'
            )

    @weight(6.25)
    def test_find_bestsize(self):
        with h5py.File('solutions.hdf5','r') as h5:
            bestsize, accuracies = submitted.todo_find_bestsize(h5['ttrain'][:], h5['tdev'][:],
                                                                h5['Ytrain'][:], h5['Ydev'][:],
                                                                h5['Lambda'][:])
            self.assertAlmostEqual(
                np.average(np.abs(accuracies-h5['accuracies'][:])), 0, places=3,
                msg='\*** todo_find_bestsize accuracies off by an average of more than 0.001'
            )
