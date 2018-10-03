import tensorflow as tf
import numpy as np
import argparse
import cv2
import time

from util import tile_images, restore_constants
from train import build
from read_dataset import read_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--data', type=str)
    parser.add_argument('--split', type=int, default=4)
    args = parser.parse_args()

    # restore configuration
    constants = restore_constants(args.config)

    # get image data
    image_size = tuple(constants.IMAGE_SIZE[:-1])
    get_next, _ = read_dataset(args.data, image_size, int(1e4),
                               constants.BATCH_SIZE, constants.EPOCH)

    # make network
    reconstruct, generate_from_latent, train, _ = build(constants)

    sess = tf.Session()
    sess.__enter__()

    # restore parameters
    saver = tf.train.Saver()
    saver.restore(sess, args.model)

    train_iterator = get_next()
    batch_images = np.array([next(train_iterator)[0]], dtype=np.float32) / 255.0

    # reconstruction
    reconst, latent = reconstruct(batch_images)
    latent_range = np.linspace(-3.0, 3.0, num=20)
    latent_in_page = int(constants.LATENT_SIZE / args.split)

    for page in range(args.split):
        image_rows = []
        for i in range(latent_in_page):
            index = page * latent_in_page + i
            # change specific element of latent variable
            tiled_latent = np.tile(latent[0].copy(), (20, 1))
            tiled_latent[:,index] = latent_range

            # reconstruct from latent variable
            reconst = generate_from_latent(tiled_latent)

            # tiling reconstructed images
            reconst_images = np.array(reconst * 255, dtype=np.uint8)
            reconst_tiled_images = tile_images(reconst_images, row=1)
            image_rows.append(reconst_tiled_images)

        # show reconstructed images
        image_rows = tile_images(np.array(image_rows), row=latent_in_page)
        cv2.imshow('test{}'.format(page), image_rows)

    cv2.imshow('reconstructed', reconst[0])
    cv2.imshow('original', batch_images[0])

    while cv2.waitKey(10) < 10:
        time.sleep(0.1)

if __name__ == '__main__':
    main()
