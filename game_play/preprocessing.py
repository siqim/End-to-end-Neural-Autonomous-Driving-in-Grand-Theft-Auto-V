# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 00:40:39 2019

@author: msq96
"""


import gzip
import pickle
from deepgtav.messages import frame2numpy
import cv2
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
import os


def get_num_files(dataset):
    num = 0
    cnt = 0
    while 1:
        try:
            data_dct = pickle.load(dataset)
            if data_dct['steering'] >= 0.1 or data_dct['steering'] <= -0.1:
                cnt += 1
            num += 1
        except EOFError:
            return num, cnt

def get_mean_std(dataset, total_num):
    totensor = transforms.ToTensor()
    means = []
    stds = []

    for i in tqdm(range(total_num)):

        data_dct = pickle.load(dataset)

        frame = frame2numpy(data_dct['frame'], (350,205+20))[20:]
        frame = totensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).data.numpy()

        mean = np.mean(frame, axis=(1, 2))
        std = np.std(frame, axis=(1, 2))

        means.append(mean)
        stds.append(std)

    mean = np.mean(np.stack(means), axis=0)
    std = np.mean(np.stack(stds), axis=0)

    return mean, std

def get_init_num(data_dir, phase='train'):
    path = data_dir + phase + '/'
    data_names = os.listdir(path)
    init_num = np.max([int(each.split('_')[1][:-3]) for each in data_names]) + 1
    return init_num

def split_dataset(dataset, total_num, keys, data_dir, transform, prob_for_saving_bad_ones):

    train_num = get_init_num(data_dir, 'train')
    val_num = get_init_num(data_dir, 'val')
    for i in tqdm(range(total_num)):

        data_point = {}
        data_dct = pickle.load(dataset)

        if -0.1 <= data_dct['steering'] <= 0.1:
            if np.random.binomial(1, prob_for_saving_bad_ones):

                for key in keys:
                    if key != 'frame':
                        data_point[key] = torch.FloatTensor([data_dct[key]])

                    elif key == 'frame':
                        image = frame2numpy(data_dct[key], (350,205+20))[20:]
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        data_point[key] = transform(image)

                    else:
                        print('Encounter wrong key!')
                        raise KeyboardInterrupt

                if np.random.binomial(1, 0.9):
                    data_name = data_dir + 'train/data_%d.gz' % train_num
                    with gzip.open(data_name, 'wb', compresslevel=4) as fout:
                        pickle.dump(data_point, fout)
                        train_num += 1

                else:
                    data_name = data_dir + 'val/data_%d.gz' % val_num
                    with gzip.open(data_name, 'wb', compresslevel=4) as fout:
                        pickle.dump(data_point, fout)
                        val_num += 1
        else:

            for key in keys:
                if key != 'frame':
                    data_point[key] = torch.FloatTensor([data_dct[key]])

                elif key == 'frame':
                    image = frame2numpy(data_dct[key], (350,205+20))[20:]
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    data_point[key] = transform(image)

                else:
                    print('Encounter wrong key!')
                    raise KeyboardInterrupt

            if np.random.binomial(1, 0.9):
                data_name = data_dir + 'train/data_%d.gz' % train_num
                with gzip.open(data_name, 'wb', compresslevel=4) as fout:
                    pickle.dump(data_point, fout)
                    train_num += 1

            else:
                data_name = data_dir + 'val/data_%d.gz' % val_num
                with gzip.open(data_name, 'wb', compresslevel=4) as fout:
                    pickle.dump(data_point, fout)
                    val_num += 1

def get_raw_y(data_dir, keys):

    path = data_dir + 'train/'
    data_names = os.listdir(path)

    raw_y = {key:[] for key in keys if key!='frame'}

    for data_name in tqdm(data_names):
        with gzip.open(path+data_name, 'rb') as fin:
            data_dct = pickle.load(fin)

        for key in keys:
            if key != 'frame':
                raw_y[key].append(data_dct[key].item())

    with open(data_dir+'y_raw.pickle', 'wb') as fout:
        pickle.dump(raw_y, fout)
    return raw_y

def split_bins(data_dir, num_class, keys, bin_type='even'):
    with open(data_dir + 'y_raw.pickle', 'rb') as fin:
        raw_y = pickle.load(fin)

    all_bins = {key: [] for key in keys if key!='frame'}
    all_bins_info = {key: [] for key in keys if key!='frame'}

    if bin_type == 'even':
        for key, value in raw_y.items():
            value = sorted(value)
            num_samples_per_bin = len(value)//(num_class[key])

            for i in range(0, len(value), num_samples_per_bin):
                one_bin = value[i: i+num_samples_per_bin]
                one_bin_info = {'mean': (np.min(one_bin)+np.max(one_bin))/2, 'max': np.max(one_bin),
                                'min': np.min(one_bin), 'num_samples': len(one_bin)}

                all_bins[key].append(one_bin)
                all_bins_info[key].append(one_bin_info)

            all_bins[key] = all_bins[key][:-1]
            all_bins_info[key] = all_bins_info[key][:-1]

            assert len(all_bins[key]) == num_class[key]
            assert len(all_bins_info[key]) == num_class[key]

    elif bin_type == 'linear':

        for key, value in raw_y.items():
            max_v = np.max(value)
            min_v = np.min(value)

            value_range_one_class = (max_v - min_v) / num_class[key]

            for i in range(num_class[key]):
                class_range = [min_v+(value_range_one_class*i), min(min_v+(value_range_one_class*(i+1)), max_v)]
                value_in_this_class = []
                # Not good, but ok.
                for v in value:
                    if class_range[0] <= v <= class_range[1]:
                        value_in_this_class.append(v)

                all_bins[key].append(value_in_this_class)
                all_bins_info[key].append({'mean': np.mean(class_range), 'max': class_range[1],
                                           'min': class_range[0], 'num_samples': len(value_in_this_class)})

            assert len(all_bins[key]) == num_class[key]
            assert len(all_bins_info[key]) == num_class[key]

    else:
        print('No such type of bin!')
        raise KeyboardInterrupt

    with open(data_dir + 'y_bin.pickle', 'wb') as fout:
        pickle.dump(all_bins, fout)
    with open(data_dir + 'y_bin_info.pickle', 'wb') as fout:
        pickle.dump(all_bins_info, fout)


def calc_weights():
    data_path = data_dir + 'train/'
    data_names = os.listdir(data_path)

    data_counts = {key: [] for key in ['steering', 'throttle']}

    for data_name in tqdm(data_names):
        with gzip.open(data_path+data_name, 'rb') as fin:
            data = pickle.load(fin)
            for action in ['steering', 'throttle']:
                data_counts[action].append(data[action].item())


def filter_bad_ones():
    path = data_dir + 'val/'
    data_names = os.listdir(path)

    for data_name in tqdm(data_names):
        with gzip.open(path+data_name, 'rb') as fin:
            data_dct = pickle.load(fin)

        steering = data_dct['steering']


        if  -0.2 <= steering <= -0.1:
            if np.random.binomial(1, 0.6):
                os.remove(path + data_name)
        elif -0.1 <= steering <= 0.1:
            if np.random.binomial(1, 0.9):
                os.remove(path + data_name)
        elif 0.1 <= steering <= 0.2:
            if np.random.binomial(1, 0.6):
                os.remove(path + data_name)


    idx = 0
    data_names = os.listdir(path)
    for data_name in tqdm(data_names):
        os.rename(path + data_name, path + 'temp_data_%d.gz'%idx)
        idx += 1

    idx = 0
    data_names = os.listdir(path)
    for data_name in tqdm(data_names):
        os.rename(path + data_name, path + 'data_%d.gz'%idx)
        idx += 1


if __name__ == '__main__':

    data_dir = '../data/'

    dataset = gzip.open('dataset.pz')
    keys = ['throttle', 'brake', 'steering', 'speed', 'frame']


#    total_num, good_num = get_num_files(dataset)
    total_num = 222036


#    mean, std = get_mean_std(dataset, total_num)
    mean = [0.35739973, 0.35751262, 0.36058474]
    std = [0.19338858, 0.19159749, 0.2047393 ]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


    split_dataset(dataset, total_num, keys, data_dir, transform, prob_for_saving_bad_ones=0.2)


    get_raw_y(data_dir, keys)


    num_class = {'throttle': 20, 'brake': 20, 'steering': 20, 'speed': 20}
#
#    split_bins(data_dir, num_class, keys, bin_type='even')
