import os
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

PATH = os.path.join('.', 'images')

def main(args):
    FPATH = os.path.join(PATH, args.task)
    fnames = os.listdir(FPATH)
    imgs = [np.load(os.path.join(FPATH, fname)) for fname in fnames]
    
    if args.type == 'ego':
        type = 1
    elif args.type == 'allo':
        type = 0
    
    img_h = imgs[type][0]
    for i, img in enumerate(imgs[type][1:]):
        if i == 5:
            break
        img_h = cv2.hconcat([img_h, img])

    cv2.imshow('{}: {}'.format(fnames[type], 10), img_h)
    cv2.waitKey(0)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='OddOneOutContent')
    parser.add_argument('--type', type=str, choices=['ego', 'allo'], default='ego')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
