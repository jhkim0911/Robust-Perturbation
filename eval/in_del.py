from torch import nn
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
from util.ops import *

def gkern(klen, nsig):
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class InsertionDeletion():
    def __init__(self, input_img, t_class, attr_map, model, device, img_size, mode, step, substrate_fn):
        self.input_img = input_img
        self.t_class = t_class
        self.attr_map = np.expand_dims(attr_map, axis=0)
        self.model = model
        self.mode = mode
        self.device = device
        self.img_size = img_size
        self.step = step
        self.substrate_fn = substrate_fn

    def run(self):
        input_img = self.input_img.to(self.device)
        m = torch.nn.Sigmoid()
        eu = self.model(Variable(input_img).to(self.device))

        pred = m(eu)
        top, c = torch.max(pred, 1)
        c = c.cpu().numpy()[0]
        n_steps = (self.img_size * self.img_size + self.step - 1) // self.step

        if self.mode == 'del':
            start = input_img.clone()
            finish = self.substrate_fn(input_img)
        elif self.mode == 'ins':
            start = self.substrate_fn(input_img)
            finish = input_img.clone()

        scores = np.zeros(n_steps + 1)
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(self.attr_map.reshape(-1, self.img_size * self.img_size), axis=1), axis=-1)
        for i in range(n_steps+1):
            eu = self.model(start.to(self.device))
            pred = m(eu)
            #pr, cl = torch.topk(pred, 2)
            scores[i] = pred[0, c]

            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]

                # 여기서 부터 coords index들 finish랑 똑같게 start에 넣으면 댐

                start = start.cpu()
                finish = finish.cpu()

                start.cpu().detach().numpy().reshape(1, 3, self.img_size * self.img_size)[0, :, coords] \
                    = finish.cpu().detach().numpy().reshape(1, 3, self.img_size * self.img_size)[0, :, coords]

                start = start.to(self.device)
                finish = finish.to(self.device)

        return scores
