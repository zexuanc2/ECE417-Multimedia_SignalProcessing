import unittest, h5py, submitted
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

# TestSequence
class TestStep(unittest.TestCase):
    @weight(5.6)
    def test_make_frames(self):
        hop_length = 120
        win_length = 240
        with h5py.File('solutions.hdf5','r') as f:
            frames = submitted.make_frames(f['speech'], hop_length, win_length)
            (M,N) = frames.shape
            for m in np.random.randint(0,M,10):
                for n in np.random.randint(0,N,10):
                    self.assertAlmostEqual(
                        f['frames'][m,n], frames[m,n],
                        msg= "\n*** frames[{},{}] should be {}".format(
                            m, n, f['frames'][m,n]
                        )
                    )

    @weight(5.6)
    def test_correlate(self):
        with h5py.File('solutions.hdf5','r') as f:
            autocor = submitted.correlate(f['frames'])
            (M,N) = autocor.shape
            for m in np.random.randint(0,M,10):
                for n in np.random.randint(0,N,10):
                    self.assertAlmostEqual(
                        f['autocor'][m,n], autocor[m,n],
                        msg= "\n*** autocor[{},{}] should be {}".format(
                            m, n, f['autocor'][m,n]
                    )
                )

    @weight(5.6)
    def test_make_matrices(self):
        p = 10
        with h5py.File('solutions.hdf5','r') as f:
            R, gamma = submitted.make_matrices(f['autocor'], p)
            (M,N) = gamma.shape
            for m in np.random.randint(0,M,10):
                for n in np.random.randint(0,N,10):
                    self.assertAlmostEqual(
                        f['gamma'][m,n], gamma[m,n],
                        msg= "\n*** gamma[{},{}] should be {}".format(
                            m, n, f['gamma'][m,n]
                        )
                    )
                    for k in np.random.randint(0,N,10):
                        self.assertAlmostEqual(
                            f['R'][m,n,k], R[m,n,k],
                            msg= "\n*** R[{},{},{}] should be {}".format(
                                m, n, k, f['R'][m,n,k]
                            )
                        )

    @weight(5.6)
    def test_lpc(self):
        with h5py.File('solutions.hdf5','r') as f:
            a = submitted.lpc(f['R'], f['gamma'])
            (M,N) = a.shape
            for m in np.random.randint(0,M,10):
                for n in np.random.randint(0,N,10):
                    self.assertAlmostEqual(
                        f['a'][m,n], a[m,n],
                        msg="\n*** a[{},{}] should be {}".format(
                            m, n, f['a'][m,n]
                        )
                    )

    @weight(5.6)
    def test_framepitch(self):
        with h5py.File('solutions.hdf5','r') as f:
            samplerate = 8000
            framepitch = submitted.framepitch(f['autocor'], samplerate)
            M = len(framepitch)
            for m in np.random.randint(0,M,10):
                self.assertAlmostEqual(
                    f['framepitch'][m], framepitch[m],
                    msg="\n*** framepitch[{}] should be {}".format(
                        m, f['framepitch'][m]
                    )
                )
                
    @weight(5.5)
    def test_framelevel(self):
        with h5py.File('solutions.hdf5','r') as f:
            samplerate = 8000
            framelevel = submitted.framelevel(f['frames'])
            M = len(framelevel)
            for m in np.random.randint(0,M,10):
                self.assertAlmostEqual(
                    f['framelevel'][m], framelevel[m],
                    msg="\n*** framelevel[{}] should be {}".format(
                        m, f['framelevel'][m]
                    )
                )
                
    @weight(5.5)
    def test_interpolate(self):
        with h5py.File('solutions.hdf5','r') as f:
            hop_length = 120
            # Test only the first 100 frames
            samplelevel, samplepitch = submitted.interpolate(
                f['framelevel'][:100],
                f['framepitch'][:100],
                hop_length
            )
            M = len(samplelevel)
            for m in np.random.randint(0,M,10):
                self.assertAlmostEqual(
                    f['samplelevel'][m], samplelevel[m],
                    msg="\n*** samplelevel[{}] should be {}".format(
                        m, f['samplelevel'][m]
                    )
                )
            for m in np.random.randint(0,M,10):
                self.assertAlmostEqual(
                    f['samplepitch'][m], samplepitch[m],
                    msg="\n*** samplepitch[{}] should be {}".format(
                        m, f['samplepitch'][m]
                    )
                )
                
    @weight(5.5)
    def test_excitation(self):
        with h5py.File('solutions.hdf5','r') as f:
            # Test only the first 16000 samples
            phase, excitation = submitted.excitation(f['samplelevel'][:16000],f['samplepitch'][:16000])
            M = len(phase)
            for m in np.random.randint(0,M,10):
                self.assertAlmostEqual(
                    f['phase'][m], phase[m],
                    msg="\n*** phase[{}] should be {}".format(
                        m, f['phase'][m]
                    )
                )
            for m in np.random.randint(0,M,10):
                self.assertAlmostEqual(
                    f['excitation'][m], excitation[m],
                    msg="\n*** excitation[{}] should be {}".format(
                        m, f['excitation'][m]
                    )
                )
                
    @weight(5.5)
    def test_synthesize(self):
        with h5py.File('solutions.hdf5','r') as f:
            # Test only the first 100 frames
            hop_length = 120
            y = submitted.synthesize(
                np.array(f['excitation'][:99*hop_length+1]),
                np.array(f['a'][:100,:])
            )
            M = len(y)
            for m in np.random.randint(0,M,10):
                self.assertAlmostEqual(
                    f['y'][m], y[m],
                    msg="\n*** y[{}] should be {}".format(
                        m, f['y'][m]
                    )
                )

                


