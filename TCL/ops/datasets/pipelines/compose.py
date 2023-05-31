# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence
import random

import numpy as np
from matplotlib import animation, pyplot as plt
from mmcv.utils import build_from_cfg

from ..builder import PIPELINES


@PIPELINES.register_module()
class Compose:
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms, noise=False):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        self.noise = noise
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f'transform must be callable or a dict, '
                                f'but got {type(transform)}')

    def add_random_noise(self, kps, h, w):
        alpha = 0.0001

        mean = 0
        std_h = alpha * h
        std_w = alpha * w
        kps_flat = kps.reshape(-1, 2)

        new_kps_flat = []
        for point in kps_flat:
            x, y = point

            delta_x = np.random.normal(mean, std_h, size=1)[0]
            delta_y = np.random.normal(mean, std_w, size=1)[0]

            new_x = max(0, min(h, x + delta_x))
            new_y = max(0, min(w, y + delta_y))

            new_kps_flat.append([new_x, new_y])

        new_kps_flat = np.array(new_kps_flat)
        new_kps = new_kps_flat.reshape(kps.shape)
        return new_kps

    def plot_skeleton(self, keypoints, pairs, h, w, title):
        fig, ax = plt.subplots()

        scatter = ax.scatter([], [], color='red', label='Keypoints')
        lines = []
        def update(frame):
            ax.cla()

            ax.set_xlim(np.min(keypoints[:, :, 0]), np.max(keypoints[:, :, 0]))
            ax.set_ylim(np.max(keypoints[:, :, 1]), np.min(keypoints[:, :, 1]))

            # ax.set_xlim([0, h])
            # ax.set_ylim([w, 0])

            scatter.set_offsets(keypoints[frame])

            for connection in pairs:
                start_point = keypoints[frame][connection[0]]
                end_point = keypoints[frame][connection[1]]
                line = ax.plot(*zip(start_point, end_point), color='blue')
                lines.append(line)

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=len(keypoints), interval=200)
        # return ani

        # Set up the video writer
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)

        # Save the animation as a video file
        ani.save(f'videos_compose/{title}.mp4', writer=writer)
        # ani.save(f'videos_compose/animation_{title}.gif', writer=writer)

        print('! videos_compose')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
            dict: Transformed data.
        """

        skeletons = ((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7),
                     (7, 9), (0, 6), (6, 8), (8, 10), (5, 11), (11, 13),
                     (13, 15), (6, 12), (12, 14), (14, 16), (11, 12))

        first_iter = True
        random_number = random.randint(0, 10000)

        for t in self.transforms:

            if first_iter and self.noise:
                first_iter = False
                h, w = data['img_shape']
                init_kps = data['keypoint'].copy()
                data['keypoint'] = self.add_random_noise(data['keypoint'], h, w)
                label = data['label']

                # self.plot_skeleton(init_kps[0], skeletons, h, w, f'init_kps_{label}_{random_number}')
                # self.plot_skeleton(data['keypoint'][0], skeletons, h, w, f'noise_kps_{label}_{random_number}')

            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
