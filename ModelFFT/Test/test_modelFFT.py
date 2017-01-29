from unittest import TestCase

from numpy import array, allclose

from ModelFFT.Test.get_data_from_files import get_data_files, get_data_from_file
from ModelFFT.model_fft import ModelFFT


class TestModelFFT(TestCase):

    def test_run(self):
        model_fft = ModelFFT()
        files = get_data_files("FFTModelDat")
        for file in files:
            data = get_data_from_file(file)
            model_fft.permutations = array(data.p, dtype=int)
            model_fft.w = array(data.w, dtype=complex)
            res = model_fft.run(array(data.input, dtype=complex))
            self.assertTrue(allclose(res, data.ans), "Error in file "+file)

    def test_generate_permutation(self):
        model_fft = ModelFFT()
        model_fft.generate_permutation(2)


