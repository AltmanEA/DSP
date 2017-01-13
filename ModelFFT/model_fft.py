from numpy import zeros, log2, array
from numpy.fft import fft

from tools import bit_reverse_index, wf


class ModelFFT:
    w = zeros((0, 0))
    permutations = zeros((0, 0))

    def run(self, data):
        stages, points = self.permutations.shape
        assert (stages+1, points) == self.w.shape
        assert points == data.size

        result = zeros((stages+2, points), dtype=complex)
        result[0, :] = data

        for stage in range(stages):
            muls = result[stage, :] * self.w[stage, :]
            butterflys = zeros(muls.shape, dtype=complex)

            for i in range(points // 2):
                j = i + points // 2
                butterflys[i] = muls[i] + muls[j]
                butterflys[j] = muls[i] - muls[j]

            for i in range(points):
                result[stage+1, i] = butterflys[self.permutations[stage, i]]

        result[stages+1, :] = result[stages, :] * self.w[stages, :]

        return result

    def generate_permutation(self, stages):
        points = 2 ** stages
        result = zeros((stages+1, points))
        for stage in range(stages):
            pass




