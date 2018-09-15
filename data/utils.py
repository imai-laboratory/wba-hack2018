import os
import numpy as np
from oculoenv import OddOneOutContent, Environment
from oculoenv.utils import deg2rad
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import matplotlib.animation as animation


class DataGenerator():
    def __init__(self, content_name):
        self.CAMERA_INITIAL_ANGLE_V = deg2rad(10.0)
        # TODO: 他の環境も指定できるようにする
        self.content_name = content_name
        if self.content_name == 'OddOneOutContent':
            self.content = OddOneOutContent()
        else:
            raise ValueError('{} is invalid content.'.format(content_name))
        self.env = Environment(self.content)
        self.egocentric_images = None
        self.allocentric_images = None

    # TODO: 他の環境でも同様のmethodで動くか確認する
    def generate_egocentric_images(self, episode=5, length=400, inplace=True):
        self.env.reset()

        # egocentric_images.shape: (episode, length, height, width)
        egocentric_images = []
        for _ in range(episode):
            self.env.reset()
            # 初期状態から遷移するのに必要な行動
            action = np.array([0, -self.CAMERA_INITIAL_ANGLE_V])
            obs, reward, done, _ = self.env.step(action)
            images = []
            for _ in range(length):
                dh = np.random.uniform(low=-0.02, high=0.02)
                dv = np.random.uniform(low=-0.02, high=0.02)
                action = np.array([dh, dv])
                obs, reward, done, _ = self.env.step(action)
                if reward != 0:
                    # タスクがたまたま成功して終了した場合は強制的に再開する
                    self.env.reset()
                    action = np.array([0, -self.CAMERA_INITIAL_ANGLE_V])
                    obs, reward, done, _ = self.env.step(action)
                image = obs['screen'].copy()
                images.append(image)
            egocentric_images.append(images)
        egocentric_images = np.array(egocentric_images).reshape((-1, 128, 128, 3))
        
        if inplace:
            self.egocentric_images = egocentric_images
        
        return egocentric_images
    
    def save_egocentric_images(self, dirname='images', prefix='egocentric_images'):
        dirname = str(Path(dirname).joinpath(self.content_name))
        os.makedirs(dirname, exist_ok=True)
        now = datetime.datetime.now()
        filename = prefix + '{:%Y%m%d}'.format(now) + '.npy'
        path = Path(dirname).joinpath(filename)

        if self.egocentric_images is not None:
            np.save(path, self.egocentric_images)
        
        return path

    # TODO: 他の環境でも同様のmethodで動くか確認する
    def generate_allocentric_images(self, scene=5, inplace=True):
        self.env.reset()

        # allocentric_images.shape: (scene, height, width)
        allocentric_images = []
        for _ in range(scene):
            self.env.reset()
            # 初期状態から遷移するのに必要な行動
            action = np.array([0, -self.CAMERA_INITIAL_ANGLE_V])
            obs, reward, done, _ = self.env.step(action)
            images = obs['screen'].copy()
            allocentric_images.append(images)
        allocentric_images = np.array(allocentric_images)
        
        if inplace:
            self.allocentric_images = allocentric_images
        
        return allocentric_images
    
    def save_allocentric_images(self, dirname='images', prefix='allocentric_images'):
        dirname = str(Path(dirname).joinpath(self.content_name))
        os.makedirs(dirname, exist_ok=True)
        now = datetime.datetime.now()
        filename = prefix + '{:%Y%m%d}'.format(now) + '.npy'
        path = Path(dirname).joinpath(filename)

        if self.allocentric_images is not None:
            np.save(path, self.allocentric_images)
        
        return path
