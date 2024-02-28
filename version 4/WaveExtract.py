import numpy as np
import tifffile
from scipy.signal import convolve2d, argrelextrema
from scipy.ndimage import gaussian_filter1d
from scipy.stats import iqr
from tqdm.notebook import tqdm
from multiprocessing import Pool, cpu_count
from itertools import repeat
from functools import partial

# Class to extract wave
class WaveExtract:
    
    def __init__(self, tiff_file, image_range = None):
        self.path = tiff_file.split("/")
        self.path.pop(-1)
        self.path = '/'.join(self.path)
        # import and process image
        print(1)
        if image_range:
            self.mov =  tifffile.imread(tiff_file, key = range(image_range[0], image_range[1]))
        else:
            self.mov =  tifffile.imread(tiff_file)

        def normalize(array):
            mini = array.min()
            maxi = array.max()
            return ((array - mini) / (maxi - mini)  * 255).astype('uint8')

        self.inter_q = iqr(self.mov, rng=(25, 75))

        self.median = np.median(self.mov)
        print('self.inter_q:', self.inter_q, 'self.median: ', self.median,  'upper limit:', self.median + 50 * self.inter_q)
        self.mov = normalize(np.clip(self.mov, self.median - 50 * self.inter_q, self.median + 50 * self.inter_q))
        self.dim = self.mov[0].shape

    def gen_convolve(self, pool_size=56, output_path = 'wave_convolution.npy'):
        # generate convolved matrix over the tiff 
        output_path = self.path + '/' + output_path
        def create_circle_kernel(radius):
            # circle shape kernel 
            diameter = 2 * radius + 1
            y, x = np.ogrid[-radius:diameter - radius, -radius:diameter - radius]
            circle_mask = x**2 + y**2 <= radius**2
            return circle_mask.astype(np.float64) / circle_mask.sum()

        radius = int(pool_size / 2)
        kernel = create_circle_kernel(radius)

        convolution = []
        # apply convolution to each frame
        p = Pool(cpu_count())
        convolution = list(
            tqdm(
                p.imap(
                        partial(convolve2d, in2=kernel, mode = 'same', boundary = 'fill'), self.mov
                    )
                , total=len(self.mov))
            )
        p.close()
        p.join()

        self.convolve = np.array(convolution)
        # save the convolved matrix into target file
        with open(output_path, 'wb') as f:
            np.save(f, self.convolve)
        
    def gen_mask(self, image_range = None, input_path = None, output_path = 'wave_mask.tif', sigma=6):
        # create labeled tiff result

        # you can use your own input convolved matrix from input path or use the one saved in WaveExtract class after gen_convolve
        if input_path:
            with open(self.path + '/' + input_path, 'rb') as f:
                convolution = np.load(f)
        else:
            convolution = self.convolve
            
        if image_range:
            start = image_range[0]
            end = image_range[1]
        else:
            start = 0
            end = len(convolution)
        
        # smooth the signal
        filtered_distribution = gaussian_filter1d(convolution[start:end],
                                    sigma=sigma, 
                                    axis=0, 
                                    order=0, 
                                    mode='reflect', 
                                    cval=0.0, 
                                    truncate=4.0, radius=None)

        # reconstruct the tiff
        filtered_distribution = np.insert(filtered_distribution, [0, len(filtered_distribution)], np.zeros(self.dim), axis = 0)

        filtered_shape = filtered_distribution.shape

        # find all the minimum of the signal 
        reshaped = filtered_distribution.reshape(filtered_shape[0], filtered_shape[1] * filtered_shape[2])
        wave_idx = argrelextrema(reshaped,
                      np.less, 
                      axis = 0,
                      mode = 'clip'
                     )
        # label the minimum with white 255
        self.mask = np.zeros(filtered_shape).astype('uint8')
        self.mask[wave_idx[0], wave_idx[1] // filtered_shape[2], wave_idx[1] % filtered_shape[2]] = 255
        tifffile.imwrite(self.path + '/' + output_path, self.mask)

