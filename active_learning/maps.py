from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage.morphology import distance_transform_edt
import cv2
from scipy.special import softmax
import seaborn as sns
from utils.utils import get_paths, get_files
from os.path import join


class AbstractMapProcessor:

    def process_maps(self, predictions, sample_name, sample_num, save_dir, input_img=None, gt_img=None, scores=None):
        raise NotImplementedError


class VarianceMap(AbstractMapProcessor):

    def process_maps(self, predictions, sample_name, sample_num, save_dir, input_img=None, gt_img=None, scores=None):
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

    def process_maps(self, predictions, sample_name, sample_num, save_dir, input_img=None, gt_img=None, scores=None):

        pred_probs = softmax(predictions, 0)
        pred_classes = np.argmax(pred_probs, 0)
        std = np.std(pred_classes, axis=-1)
        transform = distance_transform_edt((1 - np.mean(pred_classes, axis=-1)))

        map = np.multiply(std, transform)
        map = MinMaxScaler().fit_transform(map)

        fig_path = os.path.join(save_dir, 'map_{}.png'.format(sample_num))
        fig = plt.figure()
        plt.imshow(map, cmap='viridis')
        plt.colorbar()
        plt.title('{} edt uncertainty map'.format(sample_name))
        plt.savefig(fig_path)
        plt.close(fig)


class ComparativeVarianceMap(AbstractMapProcessor):

    def process_maps(self, predictions, sample_name, sample_num, save_dir, input_img=None, gt_img=None, scores=None):

        pred_probs = softmax(predictions, 0)
        pred_classes = np.argmax(pred_probs, 0)

        # Thresh gt
        gt_image = gt_img.squeeze(1)
        gt_image = (gt_image != 0) * 1

        # Threshold mean prediction
        mean_preds = np.mean(pred_classes, axis=-1)
        thresh_mean_preds = (mean_preds > 0.5) * 1.0

        # variance map
        # std = np.std(pred_classes, axis=-1)
        var = np.var(pred_classes, axis=-1)
        # transform = distance_transform_edt(thresh_mean_preds)
        # transform = transform / transform.max()
        # var = np.multiply(var, transform)
        map = MinMaxScaler().fit_transform(var)

        fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, figsize=(15, 10))

        # Plot input image
        # [3, h, w] => [h, w, 3]
        ax1.imshow(input_img.squeeze(0).transpose((1, 2, 0)))
        ax1.set_title('Input image')

        # Plot uncertainty map
        ax2.set_title('MC variance map')
        # pcm = ax2.pcolormesh(map, cmap='viridis')
        # fig.colorbar(pcm, ax=ax2)
        ax2.imshow(map, cmap='viridis')


        # Plot gt image
        ax3.imshow(gt_image.squeeze(0))
        ax3.set_title('Ground truth segmentation')

        # Plot avg segmentation results
        ax4.imshow(thresh_mean_preds)
        ax4.set_title('Segmentation prediction')

        # Plot distribution
        img_score = scores[sample_num]
        sns.distplot(scores, hist=False, rug=True, ax=ax5, axlabel='acq scores')
        ax5.set_title('Acquisition scores distribution')
        ax5.vlines(img_score, 0, ax5.get_ylim()[1] * 0.25, colors='r')

        # Plot WSL cam pred
        wsl_model = 'deepmil_multi'
        wsl_cam = get_wsl_cam(sample_name, model=wsl_model)
        ax6.imshow(wsl_cam)
        ax6.set_title('WSL {} CAM pred'.format(wsl_model))

        # Save fig
        fig_path = os.path.join(save_dir, 'map_{}.png'.format(sample_num))
        plt.savefig(fig_path, dpi=100)
        plt.close(fig)


def get_wsl_cam(img_name, model='deepmil_multi'):
    wsl_dir = '/home/victor/PycharmProjects/active-learning-segmentation-baselines/wsl_cams'
    cams_dir = join(wsl_dir, '{}/npy/'.format(model))
    paths = get_paths(cams_dir, 'npy')
    file_names = get_files(cams_dir, 'npy')

    file_names = [f.replace('.npy', '') for f in file_names]
    wsl_cam_path = paths[file_names.index(img_name[0].replace('.bmp', ''))]

    return np.load(wsl_cam_path)


if __name__ == '__main__':
    get_wsl_cam('testA_21')