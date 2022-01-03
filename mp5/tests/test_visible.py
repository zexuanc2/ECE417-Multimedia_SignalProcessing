import unittest, h5py, submitted
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

class TestStep(unittest.TestCase):
    def setUp(self):
        with h5py.File('data.hdf5','r') as f:
            self.features = f['features'][:]
            self.rects = f['rects'][:]
            self.targets = f['targets'][:]
            self.anchors = f['anchors'][:]
        with h5py.File('weights_initial.hdf5','r') as f:
            self.W1=f['W1'][:]
            self.W2=f['W2'][:]
        self.h5 = h5py.File('solutions.hdf5','r')
   
    @weight(8.34)
    def test_forwardprop(self):
        H, Yhat = submitted.forwardprop(self.features[0,:,:,:], self.W1, self.W2)
        e=np.sum(np.abs(H-self.h5['H']))/np.sum(np.abs(self.h5['H']))
        self.assertTrue(e < 0.04, 'forwardprop H wrong by more than 4% (visible case)')
        e=np.sum(np.abs(Yhat-self.h5['Yhat']))/np.sum(np.abs(self.h5['Yhat']))
        self.assertTrue(e < 0.04, 'forwardprop Yhat wrong by more than 4% (visible case)')

    @weight(8.34)
    def test_detect(self):
        best_rects = submitted.detect(self.h5['Yhat'][:], 10, self.anchors)
        d = 0
        for i in range(10):
            d+=np.amin([np.sum(np.abs(best_rects[i,:]-self.h5['best_rects'][j,:])) for j in range(10)])
        e=d/np.sum(np.abs(self.h5['best_rects']))
        self.assertTrue(e < 0.04, 'detect best_rects wrong by more than 4% (visible case)')

    @weight(8.33)
    def test_loss(self):
        bce_loss, mse_loss = submitted.loss(self.h5['Yhat'][:], self.targets[0,:,:,:,:])        
        e=np.sum(np.abs(bce_loss-self.h5['bce_loss'][0]))/np.sum(np.abs(self.h5['bce_loss'][0]))
        self.assertTrue(e < 0.04, 'bce_loss wrong by more than 4% (visible case)')
        e=np.sum(np.abs(mse_loss-self.h5['mse_loss'][0]))/np.sum(np.abs(self.h5['mse_loss'][0]))
        self.assertTrue(e < 0.04, 'mse_loss wrong by more than 4% (visible case)')

    @weight(8.33)
    def test_backprop(self):
        GradXi1,GradXi2=submitted.backprop(self.targets[0,:,:,:,:],self.h5['Yhat'][:],
                                           self.h5['H'][:],self.W2)
        e=np.sum(np.abs(GradXi1-self.h5['GradXi1']))/np.sum(np.abs(self.h5['GradXi1']))
        self.assertTrue(e < 0.04, 'backprop GradXi1 wrong by more than 4% (visible case)')
        e=np.sum(np.abs(GradXi2-self.h5['GradXi2']))/np.sum(np.abs(self.h5['GradXi2']))
        self.assertTrue(e < 0.04, 'backprop GradXi2 wrong by more than 4% (visible case)')

    @weight(8.33)
    def test_weight_gradient(self):
        dW1,dW2 = submitted.weight_gradient(self.features[0,:,:,:],self.h5['H'][:],
                                            self.h5['GradXi1'][:],self.h5['GradXi2'][:],3,3)
        e=np.sum(np.abs(dW1-self.h5['dW1']))/np.sum(np.abs(self.h5['dW1']))
        self.assertTrue(e < 0.04, 'weight_gradient dW1 wrong by more than 4% (visible case)')
        e=np.sum(np.abs(dW2-self.h5['dW2']))/np.sum(np.abs(self.h5['dW2']))
        self.assertTrue(e < 0.04, 'weight_gradient dW2 wrong by more than 4% (visible case)')

    @weight(8.33)
    def test_weight_update(self):
        new_W1, new_W2 = submitted.weight_update(self.W1,self.W2,self.h5['dW1'][:],self.h5['dW2'][:],
                                                 0.0001)
        e=np.sum(np.abs(new_W1-self.h5['new_W1']))/np.sum(np.abs(self.h5['new_W1']))
        self.assertTrue(e < 0.04, 'weight_update new_W1 wrong by more than 4% (visible case)')
        e=np.sum(np.abs(new_W2-self.h5['new_W2']))/np.sum(np.abs(self.h5['new_W2']))
        self.assertTrue(e < 0.04, 'weight_update new_W2 wrong by more than 4% (visible case)')

