import numpy as np
import torch

class energy_pointing_game:
    def __init__(self, data, attr_map, device, num_classes):
        self.data = data
        self.device = device
        self.attr_map = attr_map
        self.num_classes = num_classes
        self.hits = np.zeros(num_classes)
        self.misses = np.zeros(num_classes)

    def run(self):
        annot = self.data

        return_v = []

        for i in range(annot.shape[0]):
            total_sum = self.attr_map.sum()
            mask = np.zeros(np.shape(self.attr_map), dtype=int)
            mat = annot[i]
            bnd_box = (mat[0], mat[1], mat[0] + mat[2], mat[1] + mat[3])
            mask[int(bnd_box[1]):int(bnd_box[3]), int(bnd_box[0]):int(bnd_box[2])] = 1

            masked_attr_map = mask * self.attr_map
            energy_sum = masked_attr_map.sum()
            energy = energy_sum/total_sum

            return_v.append(energy)

        return sum(return_v) / len(return_v)
