import os
import sys
import cv2
import argparse
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.utils.data

from datasets.coco import COCO_MEAN, COCO_STD, COCO_NAMES, COCO_IDS
from datasets.pascal import VOC_MEAN, VOC_STD, VOC_NAMES

from nets.hourglass import get_hourglass
from nets.resdcn import get_pose_net

from utils.utils import load_model
from utils.image import transform_preds, get_affine_transform, preprocess_image, draw_bbox_with_id
from utils.post_process import ctdet_decode

from nms.nms import soft_nms

# from nms import soft_nms

COCO_COLORS = sns.color_palette('hls', len(COCO_NAMES))
VOC_COLORS = sns.color_palette('hls', len(VOC_NAMES))

# Training settings
parser = argparse.ArgumentParser(description='centernet')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--img_dir', type=str, default='./data/demo.png')
parser.add_argument('--ckpt_dir', type=str, default='./ckpt/pascal_resdcn18_512/checkpoint.t7')

parser.add_argument('--arch', type=str, default='resdcn_18')

parser.add_argument('--dataset', type=str, default='pascal')
parser.add_argument('--img_size', type=int, default=512)

parser.add_argument('--test_flip', action='store_true')
parser.add_argument('--test_scales', type=str, default='1')  # 0.5,0.75,1,1.25,1.5

parser.add_argument('--test_topk', type=int, default=100)

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

cfg.test_scales = [float(s) for s in cfg.test_scales.split(',')]
CLASSES = len(COCO_IDS) + 1

def main():
  cfg.device = torch.device('cuda')
  torch.backends.cudnn.benchmark = False

  max_per_image = 100

  print('Creating model...')
  if 'hourglass' in cfg.arch:
    model = get_hourglass[cfg.arch]
  elif 'resdcn' in cfg.arch:
    model = get_pose_net(num_layers=int(cfg.arch.split('_')[-1]),
                        num_classes=2 if cfg.dataset == 'coco' else 20)
  else:
    raise NotImplementedError

  model = load_model(model, cfg.ckpt_dir)
  model = model.to(cfg.device)
  model.eval()

  for i in range(2):
    # filez = 'carousel_2_frame_' + str(i+4300).zfill(10) + '.jpg'
    filez = str(i+1000).zfill(4) + '.jpg'
    # print(filename)
    print(cfg.img_dir + filez)
    image = cv2.imread(cfg.img_dir + filez)

    imgs = preprocess_image(cfg.img_dir + filez, cfg.test_scales, cfg.img_size, cfg.arch, cfg.dataset, COCO_MEAN, COCO_STD)

    # orig_image = image

    with torch.no_grad():
      detections = []
      for scale in imgs:
        imgs[scale]['image'] = imgs[scale]['image'].to(cfg.device)

        output = model(imgs[scale]['image'])[-1]
        dets = ctdet_decode(*output, K=cfg.test_topk)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]
        print(dets.shape)

        top_preds = {}
        dets[:, :2] = transform_preds(dets[:, 0:2],
                                      imgs[scale]['center'],
                                      imgs[scale]['scale'],
                                      (imgs[scale]['fmap_w'], imgs[scale]['fmap_h']))
        dets[:, 2:4] = transform_preds(dets[:, 2:4],
                                      imgs[scale]['center'],
                                      imgs[scale]['scale'],
                                      (imgs[scale]['fmap_w'], imgs[scale]['fmap_h']))
        cls = dets[:, -1]
        for j in range(CLASSES):
          inds = (cls == j)
          top_preds[j + 1] = dets[inds, :5].astype(np.float32)
          top_preds[j + 1][:, :4] /= scale

        detections.append(top_preds)

      bbox_and_scores = {}

      for j in range(1, (CLASSES+1) if cfg.dataset == 'coco' else 21):
        bbox_and_scores[j] = np.concatenate([d[j] for d in detections], axis=0)
        if len(cfg.test_scales) > 1:
          soft_nms(bbox_and_scores[j], Nt=0.5, method=2)
      scores = np.hstack([bbox_and_scores[j][:, 4] for j in range(1, (CLASSES+1) if cfg.dataset == 'coco' else 21)])

      if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, (CLASSES+1) if cfg.dataset == 'coco' else 21):
          keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
          bbox_and_scores[j] = bbox_and_scores[j][keep_inds]

      # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      # plt.show()
      # fig = plt.figure(0)

      colors = COCO_COLORS if cfg.dataset == 'coco' else VOC_COLORS
      # names = COCO_NAMES if cfg.dataset == 'coco' else VOC_NAMES
      # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


      for lab in bbox_and_scores:
        for boxes in bbox_and_scores[lab]:
          x1, y1, x2, y2, score = boxes
          if score > 0.3:
            draw_bbox_with_id(image, [x1, y1, x2, y2], 'part%.2f' % score)
            # plt.gca().add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
            #                               linewidth=2, edgecolor=colors[1], facecolor='none'))
            # plt.text(x1 + 3, y1 + 3, names[1] + '%.2f' % score,
            #         bbox=dict(facecolor=colors[1], alpha=0.5), fontsize=7, color='k')

      # fig.patch.set_visible(False)
      # plt.axis('off')
      # plt.savefig('data/frame_{}.png'.format(str(i).zfill(5)), dpi=300, transparent=True)
      # plt.show()
      cv2.imshow('Test', image)
      cv2.waitKey(1)
      cv2.imwrite('frame_{}.jpg'.format(str(i).zfill(5)), image)


if __name__ == '__main__':
  main()
