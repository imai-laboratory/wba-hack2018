import sys
import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from opticalflow import LIP


PATH = os.path.join('..', 'data', 'images')


def get_optical_flow_hsv(optical_flow):
    h, w = optical_flow.shape[:2]
    fx, fy = optical_flow[:, :, 0], optical_flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return image


def angle(x, y):
    u = np.array([1, 0])
    v = np.array([x, y])
    i = np.inner(u, v)
    n = np.linalg.norm(u) * np.linalg.norm(v)
    
    c = i / n
    
    return np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))


def ang_to_vec(angle):
    res = angle[0]
    for i in angle[1:]:
        res = np.max([res, i]) - (np.abs(res - i) / 2)
        
        print(res)

        return np.cos(res * np.pi / 180), np.sin(res * np.pi / 180)
    

def show_optical_flow(optical_flow):
    image = get_optical_flow_hsv(optical_flow)
    
    step = 16
    
    h, w = optical_flow.shape[:2]
    y, x = np.mgrid[step // 2:h:step, step // 2:w:step].reshape(2, -1).astype(int)
    fx, fy = optical_flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    
    cv2.polylines(image, lines, 0, (0, 255, 0))
    
    ang = []
    for i, ((x1, y1), (x2, y2)) in enumerate(lines):
        print(i, (x1, y1), (x2, y2))
        
        ang.append(angle(x1, y1))
        cv2.circle(image, (x1, y1), 1, (0, 255, 0), -1)
        
        x, y = ang_to_vec(ang)
        x, y = int(np.round(100*x, 2)), int(np.round(100*y, 2))
        line = np.vstack([0, 0, x, y]).T.reshape(-1, 2, 2)
        cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
        cv2.polylines(image, line, 0, (255, 0, 0))
        
        plt.imshow(image)
        plt.show()
        
        
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

    # imgl = [img_normal, img_saliency, opt0, opt1]
    # for i, img in enumerate(imgl):
    #     plt.subplot(4, 1, i + 1)
    #     plt.imshow(img)

    plt.savefig(os.path.join(PATH, 'opticalflow_{}.png'.format(args.span)))
    show_optical_flow(lip.last_optical_flow)
    
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--task', type=str, default='MultipleObjectTrackingContent')
    parser.add_argument('--type', type=str, choices=['ego', 'allo'], default='ego')
    parser.add_argument('--span', type=int, default='5')
    
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
