import sys
import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from opticalflow import LIP


PATH = os.path.join('..', 'data', 'images')


def main(args):
    FPATH = os.path.join(PATH, args.task)
    fnames = os.listdir(FPATH)
    imgs = [np.load(os.path.join(FPATH, fname)) for fname in fnames]

    if args.type == 'ego':
        type = 1
    elif args.type == 'allo':
        type = 0

    imgs = imgs[type][0:args.span*10:args.span]
    
    lip = LIP()
    img_saliency = lip.call(imgs[0])['to_fef'][0]
    opt0 = lip.call(imgs[0])['to_fef'][1][:,:,0]
    opt1 = lip.call(imgs[0])['to_fef'][1][:,:,1]
    img_normal = imgs[0]
    for img in imgs[1:]:
        img_normal = cv2.hconcat([img_normal, img])
        img_ = lip.call(img)
        img_saliency = cv2.hconcat([img_saliency, img_['to_fef'][0]])
        opt0 = cv2.hconcat([opt0, img_['to_fef'][1][:,:,0]])
        opt1 = cv2.hconcat([opt1, img_['to_fef'][1][:,:,1]])
       
    plt.figure(figsize=(12, 10))
    
    imgl = [img_normal, img_saliency, opt0, opt1]
    for i, img in enumerate(imgl): 
        plt.subplot(4, 1, i + 1)
        plt.imshow(img)

    plt.savefig(os.path.join(PATH, 'opticalflow_{}.png'.format(args.span)))
    # plt.show()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='MultipleObjectTrackingContent')
    parser.add_argument('--type', type=str, choices=['ego', 'allo'], default='ego')
    parser.add_argument('--span', type=int, default='5')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
