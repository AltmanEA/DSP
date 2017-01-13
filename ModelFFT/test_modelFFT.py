from unittest import TestCase

from numpy import array, allclose

from ModelFFT.model_fft import ModelFFT


class TestModelFFT(TestCase):

    w2 = array([[1, 1], [1, 1]])
    p2 = array([[0, 1]])
    input_2 = array([1, 2])
    right_ans_2 = array([[1, 2], [3, -1], [3, -1]])

    w4 = array([[0, 1, 2, 3]])
    p4 = array([[0, 1, 2, 3]])

    def test_run(self):
        model_fft = ModelFFT()

        model_fft.permutations = self.p2
        model_fft.w = self.w2
        res = model_fft.run(self.input_2)
        self.assertTrue(allclose(res, self.right_ans_2))



    def test_generate_permutation(self):
        model_fft = ModelFFT()
        model_fft.generate_permutation(2)
