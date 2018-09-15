import cv2
import os
from copy import deepcopy
import math
import numpy as np


# flags
draw = False
load_new = True
mode = None  # long or short
LABEL_COLOR = (255, 0, 0)
LINE_COLOR = (0, 255, 255)
POINT_COLOR = (0, 255, 0)
POINT_COLOR_R = (255, 255, 0)
sensity = 5

center = None
angle = None
axes = None


img_dir = '/Users/yuqicui/Desktop/img_dataset/oocyst_img'
save_dir = 'label_cell'
keyValue = {
    '2': 50,
    '3': 51,
    '4': 52,
    '5': 53,
    'a': 97,
    'c': 99,
    'd': 100,
    'i': 105,
    'q': 113,
    't': 116,
    'e': 101,
    'y': 121,
    'z': 122,
    '[': 91,
    ']': 93,
    'l': 108,
    'esc': 27,
    's': 115,

}

# listener
def draw_ellipse(event, x, y, flags, param):
    global mode, draw, center, axes, angle, img_raw, img_show, long_pt, short_pt
    if draw == True and event == cv2.EVENT_LBUTTONDOWN:
        if check_is_pt(x, y, long_pt):
            # long pt, change length and rotation
            mode = 'long'
        elif check_is_pt(x, y, short_pt):
            # short pt, change length only
            mode = 'short'
        elif check_is_pt(x, y, center):
            # center, move center
            mode = 'center'
        else:
            mode = None
        print(mode)
    if draw == True and event == cv2.EVENT_MOUSEMOVE:
        if mode == 'center':
            center = (x, y)
        if mode == 'short':
            angle = compute_angle((x, y)) - 90
            set_axes(short=compute_distance((x, y), center))
        if mode == 'long':
            angle = compute_angle((x, y))
            set_axes(long=compute_distance((x, y), center))
        fresh_img()
    if draw == True and event == cv2.EVENT_LBUTTONUP:
        mode = None

    pass


def set_axes(long=None, short=None):
    global axes
    axes = list(axes)
    if short is not None:
        axes[1] = int(short)
    if long is not None:
        axes[0] = int(long)
    axes = tuple(axes)


def compute_distance(p1, p2):
    return abs(math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def compute_angle(p):
    global center
    return int(math.degrees(math.atan2(p[1] - center[1], p[0] - center[0])))


def fresh_img():
    global img_show, img_raw, center, long_pt, short_pt, axes, angle
    img_show = deepcopy(img_raw)
    long_pt, short_pt = draw_auxiliary_ellipse(img_show, center, axes, angle)
    cv2.imshow(window_name, img_show)


def draw_auxiliary_ellipse(img, c, axes, angle):
    cv2.ellipse(img, c, axes, angle, 0, 360, color=LABEL_COLOR, thickness=1)
    pt1, pt2 = compute_pt_coordinate(c, axes[0], angle)
    draw_line_with_endpoint(pt1, pt2, img, pt_color=POINT_COLOR_R)  # long axes

    pt3, pt4 = compute_pt_coordinate(c, axes[1], angle + 90)
    draw_line_with_endpoint(pt3, pt4, img)  # short axes

    cv2.circle(img, c, radius=5, color=POINT_COLOR, thickness=3)

    return pt1, pt3


def draw_line_with_endpoint(pt1, pt2, img, pt_color=POINT_COLOR):
    cv2.line(img, pt1, pt2, color=LINE_COLOR)
    cv2.circle(img, pt1, radius=5, color=pt_color, thickness=3)



def compute_pt_coordinate(c, nL, angle):
    pt1, pt2 = [0, 0], [0, 0]
    pt1[1] = c[1] + int(math.sin(math.radians(angle)) * nL)
    pt2[1] = c[1] - int(math.sin(math.radians(angle)) * nL)

    pt1[0] = c[0] + int(math.cos(math.radians(angle)) * nL)
    pt2[0] = c[0] - int(math.cos(math.radians(angle)) * nL)
    return tuple(pt1), tuple(pt2)


def check_is_pt(x, y, pt):
    if abs(x - pt[0]) <= sensity and abs(y-pt[1]) <= sensity:
        return True
    else:
        return False


imgs_from_dir = os.listdir(img_dir)
imgs_from_dir = [item for item in imgs_from_dir if item[-3:] == 'jpg']
nImgs = len(imgs_from_dir)
print('{} imgs detected from {}'.format(nImgs, img_dir))
img_idx = 0
window_name = 'ellipse_label_software'

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, draw_ellipse)

img_raw = None
img_show = None
long_pt, short_pt = None, None

while True:
    img_idx = min(img_idx, nImgs - 1)
    img_idx = max(img_idx, 0)
    img_show_path = imgs_from_dir[img_idx]
    if load_new:
        img_raw = cv2.imread(os.path.join(img_dir, img_show_path))
        print('[LOAD IMG] {}/{}'.format(img_idx, nImgs))
        img_show = deepcopy(img_raw)
    elif draw:
        nRow, nCol, nChan = img_show.shape
        center = (int(nRow / 2), int(nCol / 2))
        angle = 60
        axes = (100, 60)
        long_pt, short_pt = draw_auxiliary_ellipse(img_show, center, axes, angle)
    else:
        pass

    cv2.imshow(window_name, img_show)

    c = cv2.waitKey()
    if c == keyValue['esc']:
        cv2.destroyAllWindows()
        exit()

    elif c == keyValue['a']:
        img_idx -= 1
        load_new = True
    elif c == keyValue['d']:
        img_idx += 1
        load_new = True

    elif c == keyValue['i']:
        draw = True
        load_new = False
        print('Draw mode')

    elif c == keyValue['z']:
        img_show = deepcopy(img_raw)

    elif c == keyValue['s']:
        # compute 0-1 mask according to ellipse
        mask = np.zeros(img_show.shape)
        cv2.ellipse(mask, center, axes, angle, 0, 360, color=(1), thickness=-1)

        np.savez(os.path.join(save_dir, img_show_path[:-4]),
                 center=center,
                 axes=axes,
                 angle=angle,
                 mask=mask)
        load_new = False
        draw = False

