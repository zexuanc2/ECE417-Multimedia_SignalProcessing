import unittest, h5py, submitted
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

# TestSequence
class TestStep(unittest.TestCase):
    @weight(7.2)
    def test_smooth_video(self):
        with h5py.File('solutions.hdf5','r') as f:
            smoothed = submitted.smooth_video(f['lowres'][:], sigma=1.5, L=7)
            self.assertAlmostEqual(
                np.average(np.abs(smoothed-f['smoothed'][:])), 0, places=3,
                msg='\*** smooth_video(lowres,{},{}) is off by an average of more than 0.001'.format(
                    1.5, 7
                )
            )
            
    @weight(7.2)
    def test_gradients(self):
        with h5py.File('solutions.hdf5','r') as f:
            gt, gr, gc =  submitted.gradients(f['smoothed'][:])
            self.assertAlmostEqual(
                np.average(np.abs(gt-f['gt'][:])), 0, places=3,
                msg='\*** gt is off by an average of more than 0.001')
            self.assertAlmostEqual(
                np.average(np.abs(gr-f['gr'][:])), 0, places=3,
                msg='\*** gr is off by an average of more than 0.001')
            self.assertAlmostEqual(
                np.average(np.abs(gc-f['gc'][:])), 0, places=3,
                msg='\*** gc is off by an average of more than 0.001')

    @weight(7.2)
    def test_lucas_kanade(self):
        p = 10
        with h5py.File('solutions.hdf5','r') as f:
            vr, vc = submitted.lucas_kanade(f['gt'][:],f['gr'][:],f['gc'][:],6,6)
            self.assertAlmostEqual(
                np.average(np.abs(vr-f['vr'][:])), 0, places=3,
                msg='\*** vr is off by an average of more than 0.001')
            self.assertAlmostEqual(
                np.average(np.abs(vc-f['vc'][:])), 0, places=3,
                msg='\*** vc is off by an average of more than 0.001')

    @weight(7.1)
    def test_medianfilt(self):
        with h5py.File('solutions.hdf5','r') as f:
            smooth_vr = submitted.medianfilt(f['vr'][:],3,3)
            self.assertAlmostEqual(
                np.average(np.abs(smooth_vr-f['smooth_vr'][:])), 0, places=3,
                msg='\*** medianfilt(vr,3,3) is off by an average of more than 0.001')
            
    @weight(7.1)
    def test_interpolate(self):
        with h5py.File('solutions.hdf5','r') as f:
            highres_vc = submitted.interpolate(f['smooth_vc'][:], 12)
            self.assertAlmostEqual(
                np.average(np.abs(highres_vc-f['highres_vc'][:])), 0, places=3,
                msg='\*** interpolate(smooth_vc,12) is off by an average of more than 0.001')
                
    @weight(7.1)
    def test_scale_velocities(self):
        with h5py.File('solutions.hdf5','r') as f:
            vr_ref = np.array(f['highres_vr'][:])
            scaled_vr = submitted.scale_velocities(vr_ref,2)
            self.assertAlmostEqual(
                np.average(np.abs(scaled_vr-f['scaled_vr'][:])), 0, places=3,
                msg='\*** scale_velocities(vr_ref,2) is off by an average of more than 0.001')
                        
    @weight(7.1)
    def test_velocity_fill(self):
        with h5py.File('solutions.hdf5','r') as f:
            hr_ref = np.array(f['highres'][:])
            svr_ref = np.array(f['scaled_vr'][:])
            svc_ref = np.array(f['scaled_vc'][:])
            highres_filled=submitted.velocity_fill(hr_ref, svr_ref, svc_ref, f['keep'][:])
            self.assertAlmostEqual(
                np.average(np.abs(highres_filled-f['highres_filled'][:])), 0, places=3,
                msg='\*** highres_filled is off by an average of more than 0.001')
