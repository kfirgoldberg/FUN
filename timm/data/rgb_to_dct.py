import scipy.ndimage
import torch
import numpy as np
from jpeg2dct.numpy import loads
import io
from torchvision.transforms import transforms


class rgb2ycbcr(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return sample.convert('YCbCr')


class DCT(object):
    def __init__(self, normalize=True):
        self.normalize = normalize

    def __call__(self, sample):
        bytearr = io.BytesIO()
        pil_sample = transforms.ToPILImage()(sample)
        pil_sample.save(fp=bytearr, format='jpeg', quality=100, subsampling=0)
        y, cb, cr = loads(bytearr.getvalue(), normalized=self.normalize)
        img_dct = self._upsample_and_concat(y, cb, cr)
        return img_dct

    @staticmethod
    def _upsample_and_concat(y, cb, cr):
        if cb.shape[1] < y.shape[1]:
            cb_ups = scipy.ndimage.zoom(cb, 2, order=0)[:, :, ::2]
            cr_ups = scipy.ndimage.zoom(cr, 2, order=0)[:, :, ::2]
        else:
            cb_ups = cb
            cr_ups = cr
        tensor_dct = torch.Tensor(np.concatenate((y, cb_ups, cr_ups), axis=2)).permute(2, 0, 1)
        return tensor_dct
