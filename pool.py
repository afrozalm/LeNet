import numpy as np

class MaxPool_2(object):
    def __init__(self):
        self.params = []
        self.ker_size = 2
        self.stride = 2
        self.name = 'maxpool'
        self.gradParams = []

    def forward(self, fmap):
        n_channels, width, height = fmap.shape
        # max_idxs = np.ndarray(n_channels,
        #                       width/2,
        #                       height/2)
        output = np.ndarray((n_channels,
                             width/2,
                             height/2))
        mask = np.zeros(fmap.shape)

        for ch in xrange(n_channels):
            for row in xrange(width/2):
                for col in xrange(height/2):
                    i_beg = 2*row
                    j_beg = 2*col
                    i_end = 2*(row+1)
                    j_end = 2*(col+1)
                    max_idx = np.argmax(fmap[ch][i_beg:i_end, j_beg:j_end])
                    output[ch][row][col] = np.max(fmap[ch][i_beg:i_end, j_beg:j_end])
                    # max_idxs[ch][row][col] = max_idx
                    mask[ch][2*row + max_idx/2 ][2*col + max_idx%2]

        self.mask = mask
        return output

    def backward(self, deltas):
        # expand deltas
        n_channels, width, height = self.mask.shape
        dE_dX = np.ndarray(self.mask.shape)
        for ch in xrange(n_channels):
            for row in xrange(width/2):
                for col in xrange(height/2):
                    i_beg = 2*row
                    j_beg = 2*col
                    i_end = 2*(row+1)
                    j_end = 2*(col+1)
                    dE_dX[ch][i_beg:i_end, j_beg:j_end] = self.mask[ch][i_beg:i_end, j_beg:j_end]*deltas[ch][row][col]

        return dE_dX

if __name__ == '__main__':
    shape = ( 4, 32, 32 )
    a = np.ones(shape)
    d = np.ones((shape[0], shape[1]/2, shape[2]/2))
    m = MaxPool_2()
    print m.forward(a).shape
    print m.backward(d).shape
