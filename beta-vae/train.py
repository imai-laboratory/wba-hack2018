import numpy as np
import tensorflow as tf
import cv2
import constants
import argparse
import os

from util import tile_images, dump_constants
from datetime import datetime
from build_graph import build_graph
from network import make_encoder, make_decoder, make_latent
from read_dataset import read_dataset


def build(constants):
    if constants.ACTIVATION == 'leaky_relu':
        activation = tf.nn.leaky_relu
    elif constants.ACTIVATION == 'relu':
        activation = tf.nn.relu
    elif constants.ACTIVATION == 'tanh':
        activation = tf.nn.tanh
    else:
        activation = tf.nn.relu

    # make networks
    encoder = make_encoder(constants.CONVS, constants.FCS, activation)
    decoder = make_decoder(constants.CONVS, constants.FCS, activation)
    sample_latent = make_latent(constants.LATENT_SIZE)

    # build graphs
    reconstruct,\
    generate,\
    train = build_graph(
        encoder=encoder,
        decoder=decoder,
        sample_latent=sample_latent,
        image_size=constants.IMAGE_SIZE,
        latent_size=constants.LATENT_SIZE,
        lr=constants.LR
    )
    return reconstruct, generate, train

def main():
    date = datetime.now().strftime('%Y%m%d%H%M%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', type=str, default=date)
    parser.add_argument('--data', type=str)
    args = parser.parse_args()

    # get image data
    image_size = tuple(constants.IMAGE_SIZE[:-1])
    get_next, get_test = read_dataset(args.data, image_size, int(1e5),
                                      constants.BATCH_SIZE, constants.EPOCH)

    # make network
    reconstruct, generate, train = build(constants)

    sess = tf.Session()
    sess.__enter__()
    sess.run(tf.global_variables_initializer())

    train_iterator = get_next()
    test_iterator = get_test()

    # start training
    count = 0
    for batch_images in train_iterator:
        batch_images = np.array(batch_images, dtype=np.float32) / 255.0
        loss = train(batch_images, keep_prob=constants.KEEP_PROB,
                     beta=constants.BETA)
        print('loss {}:'.format(count), loss)
        count += 1

        # visualize
        if count % 100 == 0:
            test_images = next(test_iterator)
            test_images = np.array(test_images, dtype=np.float32) / 255.0
            # reconstruction
            reconst, latent = reconstruct(test_images)

            # show reconstructed images
            reconst_images = np.array(reconst * 255, dtype=np.uint8)
            reconst_tiled_images = tile_images(reconst_images)
            cv2.imshow('test', reconst_tiled_images)

            # show original images
            original_images = np.array(test_images * 255, dtype=np.uint8)
            original_tiled_images = tile_images(original_images)
            cv2.imshow('original', original_tiled_images)

            if cv2.waitKey(10) > 0:
                pass

    # save model
    print('saving model...')
    modeldir = 'saved_models/' + args.modeldir
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    saver = tf.train.Saver()
    saver.save(sess, modeldir + '/model.ckpt')
    # save configuration as json
    dump_constants(constants, modeldir + '/constants.json')

if __name__ == '__main__':
    main()
