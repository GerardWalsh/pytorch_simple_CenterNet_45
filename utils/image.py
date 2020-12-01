import numpy as np
import cv2
import random
import torch

TEXT_COLOR = (52, 206, 255)
ID_OFFSET = 10


def flip(img):
  return img[:, :, ::-1].copy()

# todo what the hell is this?
def get_border(border, size):
  i = 1
  while size - border // i <= border // i:
    i *= 2
  return border // i

def transform_preds(coords, center, scale, output_size):
  target_coords = np.zeros(coords.shape)
  trans = get_affine_transform(center, scale, 0, output_size, inv=1)
  for p in range(coords.shape[0]):
    target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
  return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
  if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
    scale = np.array([scale, scale], dtype=np.float32)

  scale_tmp = scale
  src_w = scale_tmp[0]
  dst_w = output_size[0]
  dst_h = output_size[1]

  rot_rad = np.pi * rot / 180
  src_dir = get_dir([0, src_w * -0.5], rot_rad)
  dst_dir = np.array([0, dst_w * -0.5], np.float32)

  src = np.zeros((3, 2), dtype=np.float32)
  dst = np.zeros((3, 2), dtype=np.float32)
  src[0, :] = center + scale_tmp * shift
  src[1, :] = center + src_dir + scale_tmp * shift
  dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
  dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

  src[2:, :] = get_3rd_point(src[0, :], src[1, :])
  dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

  if inv:
    trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
  else:
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

  return trans


def affine_transform(pt, t):
  new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
  new_pt = np.dot(t, new_pt)
  return new_pt[:2]


def get_3rd_point(a, b):
  direct = a - b
  return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
  _sin, _cos = np.sin(rot_rad), np.cos(rot_rad)

  src_result = [0, 0]
  src_result[0] = src_point[0] * _cos - src_point[1] * _sin
  src_result[1] = src_point[0] * _sin + src_point[1] * _cos

  return src_result


def crop(img, center, scale, output_size, rot=0):
  trans = get_affine_transform(center, scale, rot, output_size)

  dst_img = cv2.warpAffine(img,
                           trans,
                           (int(output_size[0]), int(output_size[1])),
                           flags=cv2.INTER_LINEAR)

  return dst_img


def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1 = 1
  b1 = (height + width)
  c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  # r1 = (b1 + sq1) / 2 #
  r1 = (b1 - sq1) / (2 * a1)

  a2 = 4
  b2 = 2 * (height + width)
  c2 = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  # r2 = (b2 + sq2) / 2
  r2 = (b2 - sq2) / (2 * a2)

  a3 = 4 * min_overlap
  b3 = -2 * min_overlap * (height + width)
  c3 = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3 = (b3 + sq3) / 2
  # r3 = (b3 + sq3) / (2 * a3)
  return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
  m, n = [(ss - 1.) / 2. for ss in shape]
  y, x = np.ogrid[-m:m + 1, -n:n + 1]

  h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
  h[h < np.finfo(h.dtype).eps * h.max()] = 0
  return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]

  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap


def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
  dim = value.shape[0]
  reg = np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) * value
  if is_offset and dim == 2:
    delta = np.arange(diameter * 2 + 1) - radius
    reg[0] = reg[0] - delta.reshape(1, -1)
    reg[1] = reg[1] - delta.reshape(-1, 1)

  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]

  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
  masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom,
                    radius - left:radius + right]
  masked_reg = reg[:, radius - top:radius + bottom,
               radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
    idx = (masked_gaussian >= masked_heatmap).reshape(
      1, masked_gaussian.shape[0], masked_gaussian.shape[1])
    masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg
  regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
  return regmap


def draw_msra_gaussian(heatmap, center, sigma):
  tmp_size = sigma * 3
  mu_x = int(center[0] + 0.5)
  mu_y = int(center[1] + 0.5)
  w, h = heatmap.shape[0], heatmap.shape[1]
  ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
  br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
  if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
    return heatmap
  size = 2 * tmp_size + 1
  x = np.arange(0, size, 1, np.float32)
  y = x[:, np.newaxis]
  x0 = y0 = size // 2
  g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
  g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
  g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
  img_x = max(0, ul[0]), min(br[0], h)
  img_y = max(0, ul[1]), min(br[1], w)
  heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
  return heatmap


def grayscale(image):
  return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lighting_(data_rng, image, alphastd, eigval, eigvec):
  alpha = data_rng.normal(scale=alphastd, size=(3,))
  image += np.dot(eigvec, eigval * alpha)


def blend_(alpha, image1, image2):
  image1 *= alpha
  image2 *= (1 - alpha)
  image1 += image2


def saturation_(data_rng, image, gs, gs_mean, var):
  alpha = 1. + data_rng.uniform(low=-var, high=var)
  blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
  alpha = 1. + data_rng.uniform(low=-var, high=var)
  image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
  alpha = 1. + data_rng.uniform(low=-var, high=var)
  blend_(alpha, image, gs_mean)


def color_aug(data_rng, image, eig_val, eig_vec):
  functions = [brightness_, contrast_, saturation_]
  random.shuffle(functions)

  gs = grayscale(image)
  gs_mean = gs.mean()
  for f in functions:
    f(data_rng, image, gs, gs_mean, 0.4)
  lighting_(data_rng, image, 0.1, eig_val, eig_vec)


def preprocess_image(image_path, test_scales=1, img_size=None, arch='resdcn', dataset='coco', dataset_mean=None, dataset_std=None, test_flip=False):
    image = cv2.imread(image_path)
    # orig_image = image

    height, width = image.shape[0:2]
    padding = 127 if 'hourglass' in arch else 31

    imgs = {}

    for scale in test_scales:
      new_height = int(height * scale)
      new_width = int(width * scale)

      if img_size > 0:
        img_height, img_width = img_size, img_size
        center = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
        scaled_size = max(height, width) * 1.0
        scaled_size = np.array([scaled_size, scaled_size], dtype=np.float32)
      else:
        img_height = (new_height | padding) + 1
        img_width = (new_width | padding) + 1
        center = np.array([new_width // 2, new_height // 2], dtype=np.float32)
        scaled_size = np.array([img_width, img_height], dtype=np.float32)

      img = cv2.resize(image, (new_width, new_height))
      trans_img = get_affine_transform(center, scaled_size, 0, [img_width, img_height])
      img = cv2.warpAffine(img, trans_img, (img_width, img_height))

      img = img.astype(np.float32) / 255.
      img -= np.array(dataset_mean, dtype=np.float32)[None, None, :]
      img /= np.array(dataset_std, dtype=np.float32)[None, None, :]
      img = img.transpose(2, 0, 1)[None, :, :, :]  # from [H, W, C] to [1, C, H, W]

      if test_flip:
        img = np.concatenate((img, img[:, :, :, ::-1].copy()), axis=0)

      imgs[scale] = {'image': torch.from_numpy(img).float(),
              'center': np.array(center),
              'scale': np.array(scaled_size),
              'fmap_h': np.array(img_height // 4),
              'fmap_w': np.array(img_width // 4)}

    return imgs


def draw_bbox_with_id(image, bbox, obj_class="", idx=None):
    """ Draws bbox on image from bounding box data 

    Arguments:
        image {unint8} -- Numpy array
        bbox {float} -- List of floating point values: top left, bottom right x & y. 
        obj_class {str} -- Name of object class
        id {int} -- Object id - zero indexed
    """
    text = str(obj_class)
    if idx is not None:
        text += str(idx)

    top_left = (int(bbox[0]), int(bbox[1]))
    bottom_right = (int(bbox[2]), int(bbox[3]))

    image = cv2.rectangle(image, top_left, bottom_right, (255, 153, 51), 2, 1)
    text_pos = (int(bbox[0]) - ID_OFFSET, int(bbox[1]) - ID_OFFSET)

    cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_PLAIN, 2, TEXT_COLOR, 2)