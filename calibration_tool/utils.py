from matplotlib import pyplot as plt
import cv2
import numpy as np

"""
A set of utility functions handling opencv image formats and color-spaces
"""


def patch(img, kwargs):
    """
  A patching function that:
    - Defaults cmap to grayscale if detects images with only 1 channel.
    - Defaults cmap to rgb if detects images with 3 channels and no cmap defined.
    - Converts opencv default BGR format to RGB if detects images with 3 channels.
    - Converts opencv HSV to RGB
  """

    grayscale = {'cmap': 'gray', 'vmin': 0, 'vmax': 255}

    cmap_patched = kwargs.copy()
    if len(img.shape) == 2:
        # num channels == 1
        # Defaulting cmap to grayscale
        if 'cmap' not in kwargs:
            cmap_patched.update(grayscale)

    img_patched = img
    if len(img.shape) == 3:
        if 'cmap' not in kwargs:
            # Changing BGR opencv format to RGB
            if img.shape[2] == 4:
                img_patched = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            else:
                img_patched = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # matplotlib expects hsv in [0, 1] range, simply convert opencv HSV to RGB format
        if 'cmap' in kwargs and kwargs['cmap'] == 'hsv':
            img_patched = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    return img_patched, cmap_patched


def imshow(img, **kwargs):
    """
  imshow wrapper for matplotlib.pyplot.imshow
  """
    patched_img, patched_cmap = patch(img, kwargs)
    plt.imshow(patched_img, **patched_cmap)
    plt.axis('off')


def show_images(images, titles=None, ret=None, **kwargs):
    num_images = len(images)
    fig, axs = plt.subplots(1, num_images, figsize=(12, 6))
    if titles is None:
        titles = [None for _ in images]
    for ax, img, title in zip(axs, images, titles):
        patched_img, patched_cmap = patch(img, kwargs)
        ax.imshow(patched_img, **patched_cmap)
        ax.axis('off')
        ax.set_title(title)

    if ret is not None:
        ret['plot'] = fig, axs


def plot_transform(r, s, label=None, title=None, fig=None):
    if fig is None:
        plt.figure(figsize=(5, 5))
    if not isinstance(s, list):
        ss = [s]
    else:
        ss = s

    legend = True
    if label is None:
        legend = False
        ls = [None] * len(ss)
    else:
        if not isinstance(label, list):
            ls = [label] * len(ss)
        else:
            ls = label

    for s, lbl in zip(ss, ls):
        plt.plot(r, s, label=lbl)

    plt.grid()
    plt.xlabel("r")
    plt.ylabel("s")
    plt.title(title)
    if legend:
        plt.legend()
    plt.ylim(0, 256)
    plt.xlim(0, 256)



def pix(np_array):
    return tuple(np.round(np_array).astype(int))


def draw_text(img, text,
              pos=(0, 0),
              font=cv2.FONT_HERSHEY_PLAIN,
              font_scale=3,
              font_thickness=2,
              text_color=(0, 255, 0),
              text_color_bg=(0, 0, 0)
              ):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    if text_color_bg is not None:
        cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size
