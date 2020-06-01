from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage.morphology import distance_transform_edt
import cv2

class AbstractMapProcessor:

    def process_maps(self, predictions, sample_name, sample_num, save_dir):
        raise NotImplementedError


class VarianceMap(AbstractMapProcessor):

    def process_maps(self, predictions, sample_name, sample_num, save_dir):
        scaler = MinMaxScaler()
        map = np.var(predictions, axis=0)
        map = scaler.fit_transform(map)

        fig_path = os.path.join(save_dir, 'map_{}.png'.format(sample_num))
        fig = plt.figure()
        plt.imshow(map, cmap='viridis')
        plt.colorbar()
        plt.title('{} uncertainty map'.format(sample_name))
        plt.savefig(fig_path)
        plt.close(fig)


class EdtVarMap(AbstractMapProcessor):

    def process_maps(self, predictions, sample_name, sample_num, save_dir):
        std = np.std(predictions, axis=0)
        transform = distance_transform_edt((1 - np.mean(predictions, axis=0)))
        map = np.multiply(std, transform)
        map = MinMaxScaler().fit_transform(map)

        fig_path = os.path.join(save_dir, 'map_{}.png'.format(sample_num))
        fig = plt.figure()
        plt.imshow(map, cmap='viridis')
        plt.colorbar()
        plt.title('{} edt uncertainty map'.format(sample_name))
        plt.savefig(fig_path)
        plt.close(fig)