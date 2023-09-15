import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np

import mmcv
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.utils import ProgressBar, path

# from mmyolo.registry import VISUALIZERS
from mmyolo.utils import switch_to_deploy
from mmyolo.utils.labelme_utils import LabelmeFormat
from mmyolo.utils.misc import get_file_list, show_data_classes


from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.utils.typing import ConfigDict

from mmyolo.utils import register_all_modules


# from mmdet.apis import inference_detector, init_detector
# from mmengine.registry import DefaultScope


# from mmyolo.utils import register_all_modules
# from mmyolo.registry import VISUALIZERS
# from mmyolo.utils import switch_to_deploy
# from mmyolo.utils.labelme_utils import LabelmeFormat
# from mmyolo.utils.misc import get_file_list, show_data_classes



has_mmdet = True


class RTMPose:
    def __init__(self, args):
        self.det_config = args.det_config
        self.det_checkpoint = args.det_checkpoint
        self.pose_config = args.pose_config
        self.pose_checkpoint = args.pose_checkpoint
        self.device = args.cuda_device
        self.draw_heatmap = False
        self.radius = 3
        self.alpha = 0.8
        self.thickness = 1
        self.skeleton_style = 'mmpose'
        self.det_cat_id = 0
        self.bbox_thr = 0.2
        self.nms_thr = 0.7
        self.draw_bbox = True
        self.show_kpt_idx = True
        self.show = False
        self.kpt_thr = 0.3

        # build the detection model from a config file and a checkpoint file
        # with DefaultScope.overwrite_default_scope('mmdet'):
        self.detector = init_detector(self.det_config, self.det_checkpoint, device=self.device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        # build the pose model from a config file and a checkpoint file
        self.pose_estimator = init_pose_estimator(self.pose_config,
                                              self.pose_checkpoint,
                                              device=self.device,
                                              cfg_options=dict(
                                                  model=dict(test_cfg=dict(output_heatmaps=self.draw_heatmap))))
        self.pose_estimator.cfg.visualizer.radius = self.radius
        self.pose_estimator.cfg.visualizer.alpha = self.alpha
        self.pose_estimator.cfg.visualizer.line_width = self.thickness
        # if args.visualizer:
        self.visualizer = VISUALIZERS.build(self.pose_estimator.cfg.visualizer)
        self.visualizer.set_dataset_meta(self.pose_estimator.dataset_meta, skeleton_style=self.skeleton_style)
        # else:
        #     self.visualizer = None




    def visualize(self,
                  img,
                  result,
                  radius=4,
                  thickness=1,
                  kpt_score_thr=0.3,
                  bbox_color='green',
                  dataset='TopDownAIHubDataset',
                  show=False,
                  out_file=None):
        """Visualize the detection results on the image.

        Args:
            model (nn.Module): The loaded detector.
            img (str | np.ndarray): Image filename or loaded image.
            result (list[dict]): The results to draw over `img`
                    (bbox_result, pose_result).
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            kpt_score_thr (float): The threshold to visualize the keypoints.
            skeleton (list[tuple()]): Default None.
            show (bool):  Whether to show the image. Default True.
            out_file (str|None): The filename of the output visualization image.
        """
        if hasattr(self.pose_model, 'module'):
            model = self.pose_model.module

        palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                            [255, 255, 255]])
        # print(dataset)
        if dataset in ('TopDownCocoDataset', 'BottomUpCocoDataset',
                       'TopDownOCHumanDataset', 'AnimalMacaqueDataset'):
            # show the results
            # skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            # [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            # [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

            skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11],
                        [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2],
                        [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]

            pose_limb_color = palette[[
                0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
            ]]
            pose_kpt_color = palette[[
                16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
            ]]
        elif dataset in ('TopDownAIHubDataset'):
            # show the results
            skeleton = [[15, 13], [13, 11], [14, 12], [12, 10], [11, 10],
                        [4, 11], [3, 10], [4, 3], [4, 6], [3, 5],
                        [6, 8], [5, 7], [4, 2], [3, 2], [11, 2],
                        [10, 2], [4, 2], [3, 2], [2, 1], [1, 0]
                        ]

            pose_limb_color = palette[[
                0, 0, 0, 0, 7,
                7, 7, 9, 9, 9,
                9, 9, 16, 16, 16,
                16, 16, 16, 16, 0
            ]]
            pose_kpt_color = palette[[
                0, 0, 0,
                7, 7,
                9, 9,
                9, 9,
                16, 16, 16,
                18, 18,
                18, 18,
                0
            ]]
        else:
            raise NotImplementedError()

        # img = self.pose_model.show_result(
        #     img,
        #     result,
        #     skeleton,
        #     radius=radius,
        #     thickness=thickness,
        #     pose_kpt_color=pose_kpt_color,
        #     pose_limb_color=pose_limb_color,
        #     kpt_score_thr=kpt_score_thr,
        #     bbox_color=bbox_color,
        #     show=show,
        #     out_file=out_file)

        return img

    def pose_estimate(self, img):
        # image size

        # with DefaultScope.overwrite_default_scope('mmdet'):
        register_all_modules()

        det_result = inference_detector(self.detector, img)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes_club = bboxes[np.logical_and(pred_instance.labels == 1, pred_instance.scores > 0.3)]
        try:
            bboxes_club = [list(sorted(bboxes_club, key=lambda x: x[3])[0])]
        except:
            bboxes_club = []
        bboxes = bboxes[np.logical_and(pred_instance.labels == self.det_cat_id,
                                       pred_instance.scores > self.bbox_thr)]
        bboxes = bboxes[nms(bboxes, self.nms_thr), :4]

        
        if len(bboxes) == 0:
            print('박스가 없음')
            return None

        bboxes_size = []
        for box in bboxes:
            bboxes_size.append((box[2] - box[0]) * (box[3] - box[1]))

        biggest_box_size = max(bboxes_size)
        max_index = bboxes_size.index(biggest_box_size)
        bboxes = [bboxes[max_index]]

        img_area = img.shape[0] * img.shape[1]

        if biggest_box_size < img_area * 0.05:
            print('박스가 너무 작음')
            return None

        # predict keypoints
        pose_results = inference_topdown(self.pose_estimator, img, bboxes)
        data_samples = merge_data_samples(pose_results)

        if self.visualizer is not None:
            self.visualizer.add_datasample(
                'result',
                img,
                data_sample=data_samples,
                draw_gt=False,
                draw_heatmap=self.draw_heatmap,
                draw_bbox=self.draw_bbox,
                show_kpt_idx=self.show_kpt_idx,
                skeleton_style=self.skeleton_style,
                show=self.show,
                wait_time=0,
                kpt_thr=self.kpt_thr,
                out_file='./tmp.jpg'
            )
            frame_vis = self.visualizer.get_image()
        else:
            frame_vis = None

        return data_samples.get('pred_instances', None), frame_vis, bboxes_club