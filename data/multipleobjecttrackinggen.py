from utils import DataGenerator
import argparse

parser = argparse.ArgumentParser(
    prog='multipleobjecttrackinggen.py'
)

parser.add_argument('--episode', type=int, default=10)
parser.add_argument('--length', type=int, default=100)
parser.add_argument('--scene', type=int, default=1000)
args = parser.parse_args()

def main(args):
    dg = DataGenerator(content_name='MultipleObjectTrackingContent')
    print('egocentric images: {} episode, {} length'.format(
        args.episode, args.length))
    print('allocentric images: {} scene'.format(args.scene))
    print('image shape: {} height, {} width, {} channel'.format(128, 128, 3))
    print('collecting egocentric images...')
    dg.generate_egocentric_images(
        episode=args.episode, length=args.length, inplace=True)
    e_path = dg.save_egocentric_images(
        dirname='images', prefix='egocentric_images')
    dg.reset_egocentric_images()
    print('save {}'.format(str(e_path)))
    print('collecting allocentric images...')
    dg.generate_allocentric_images(scene=args.scene, inplace=True)
    a_path = dg.save_allocentric_images(
        dirname='images', prefix='allocentric_images')
    dg.reset_allocentric_images()
    print('save {}'.format(str(a_path)))


if __name__ == '__main__':
    main(args)
