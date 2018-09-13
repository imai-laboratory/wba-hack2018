from utils import DataGenerator
import argparse

parser = argparse.ArgumentParser(
    prog = 'oddoutoutgen.py'
)
parser.add_argument('--episode', type=int, default=10)
parser.add_argument('--length', type=int, default=100)
parser.add_argument('--scene', type=int, default=100)
args = parser.parse_args()

def main(args):
    dg = DataGenerator(content_name='OddOneOutContent')
    print('egocentric images: {} episode, {} length'.format(args.episode, args.length))
    print('allocentric images: {} scene'.format(args.scene))
    print('image shape: {} height, {} width, {} channel'.format(128, 128, 3))
    dg.generate_egocentric_images(episode=args.episode, length=args.length, inplace=True)
    dg.generate_allocentric_images(scene=args.scene, inplace=True)
    e_path = dg.save_egocentric_images(dirname='images', prefix='egocentric_images')
    a_path = dg.save_allocentric_images(dirname='images', prefix='allocentric_images')
    print('save {}'.format(str(e_path)))
    print('save {}'.format(str(a_path)))

if __name__ == '__main__':
    main(args)