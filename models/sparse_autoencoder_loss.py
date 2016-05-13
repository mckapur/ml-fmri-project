import caffe
import numpy as np


class SparseAutoencoderLoss(caffe.Layer):
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need latent data, reconstruction, and latent state to compute loss.")

        # Hyperparams
        self.beta = 3. # weighting of KL divergence term
        self.rho = 0.01 # desired average activation

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count or bottom[0].count != bottom[2].count:
            raise Exception("Inputs must have the same dimension.")

        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)

        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data

        s = np.sum(self.bottom[2]) / bottom[2].num
        sparsity_term = s * np.log(s / this.rho) + (1-s) * np.log((1-s) / (1-this.rho))

        top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2. + this.beta * sparsity_term

    def backward(self, top, propagate_down, bottom):
        for i in range(3):
            if not propagate_down[i]:
                continue

            s = np.sum(self.bottom[2]) / bottom[2].num
            bottom[i].diff[...] = self.diff / bottom[i].num + self.beta * (-(self.rho / s) + ((1 - self.rho) / (1 - s)))
