# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 00:40:39 2019

@author: msq96
"""


import gzip
import pickle
from deepgtav.messages import frame2numpy
from matplotlib.pyplot import imshow
import cv2
import time
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
import os
import numpy as np
from copy import deepcopy


def get_mean_std():
    means = []
    stds = []

    for i in tqdm(range(493504)):

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

def get_raw_y():
    raw_y = {key:[] for key in keys if key!='frame'}

    for i in tqdm(range(493504)):
        data_dct = pickle.load(dataset)
        for key in keys:
            if key != 'frame':
                raw_y[key].append(data_dct[key])

    with open(data_dir+'y_raw.pickle', 'wb') as fout:
        pickle.dump(raw_y, fout)
    return raw_y


def find_even_bins(num_class):
    with open(data_dir + 'y_raw.pickle', 'rb') as fin:
        raw_y = pickle.load(fin)

    all_bins = {'throttle': [], 'brake': [], 'steering': [], 'speed': []}
    for key, value in raw_y.items():
        value = sorted(value)
        bins = []
        for i in range(0, len(value), len(value)//(num_class[key])):
            one_bin = value[i:i+len(value)//(num_class[key])]
            bins.append(one_bin)

        all_bins[key] = bins[:-1]
        assert len(all_bins[key]) == num_class[key]

    with open(data_dir + 'y_bin.pickle', 'wb') as fout:
        pickle.dump(all_bins, fout)
    return all_bins


def find_linear_bins(num_class):
    with open(data_dir + 'y_raw.pickle', 'rb') as fin:
        raw_y = pickle.load(fin)

    all_bins = {'throttle': [], 'brake': [], 'steering': [], 'speed': []}
    all_bins_info = {'throttle': [], 'brake': [], 'steering': [], 'speed': []}
    for key, value in raw_y.items():
        max_v = np.max(value) + 0.0001
        min_v = np.min(value) - 0.0001

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

    with open(data_dir + 'y_bin.pickle', 'wb') as fout:
        pickle.dump(all_bins, fout)
    with open(data_dir + 'y_bin_info.pickle', 'wb') as fout:
        pickle.dump(all_bins_info, fout)
    return all_bins


def split_dataset():

    for i in tqdm(range(total_num//seq_len)):
        data_point = {key:[] for key in keys}
        data_name = data_dir + 'data_%d'%i + '.gz'

        for j in range(seq_len):
            data_dct = pickle.load(dataset)
            for key in keys:
                if key != 'frame':
                    data_point[key].append(data_dct[key])

                elif key == 'frame':
                    frame = frame2numpy(data_dct[key], (350,205+20))[20:]
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    data_point[key].append(transform(frame))

                else:
                    print('Encounter wrong key!')
                    raise KeyboardInterrupt

        for key in keys:
            if key == 'frame':
                data_point[key] = torch.stack(data_point[key])
            elif key != 'frame':
                data_point[key] = torch.FloatTensor(data_point[key])


        with gzip.open(data_name, 'wb', compresslevel=4) as f:
            pickle.dump(data_point, f)

        s = time.time()
        with gzip.open(data_name, 'rb') as f:
            pickle.load(f)
        e = time.time()
        print(e-s)

def split_train_val():
    datanames = [each for each in os.listdir(data_dir) if 'data' in each and '.gz' in each]
    np.random.shuffle(datanames)
    for each_data in datanames:
        if np.random.binomial(1, 0.8):
            os.rename(data_dir + each_data, data_dir + 'train/' + each_data)
        else:
            os.rename(data_dir + each_data, data_dir + 'val/' + each_data)

def save_bins_info():

    y_bin_fname = data_dir + 'y_bin.pickle'
    with open(y_bin_fname, 'rb') as fin:
        y_bin = pickle.load(fin)
        for k, v in y_bin.items():
            y_bin[k] = [{'min': np.min(each_bin), 'mean': np.mean(each_bin), 'max': np.max(each_bin)} for each_bin in v]
            temp = []
            for each in y_bin[k]:
                if each not in temp:
                    temp.append(each)
            y_bin[k] = temp

    y_bin_info_fname = data_dir + 'y_bin_info.pickle'
    with open(y_bin_info_fname, 'wb') as fout:
        pickle.dump(y_bin, fout)

def build_non_seq_data(phase='val_old'):
    data_path = data_dir + phase + '/'
    data_names = os.listdir(data_dir + phase + '/')
    np.random.shuffle(data_names)

    save_idx = [i for i in range(64) if (i+1)%8==0]

    global_counter = 0

    for data_name in tqdm(data_names):

        with gzip.open(data_path+data_name, 'rb') as fin:
            seq_data = pickle.load(fin)

            for k, v in seq_data.items():
                seq_data[k] = v[save_idx]

        for idx in range(len(save_idx)):
            save_dict = {}
            for k, v, in seq_data.items():
                save_dict[k] = torch.FloatTensor(v[idx].numpy())


            filename = data_dir +  phase.split('_')[0] + '/data_%d.gz'%global_counter
            with gzip.open(filename, 'wb', compresslevel=4) as fout:
                pickle.dump(save_dict, fout)
                global_counter += 1


if __name__ == '__main__':

    data_dir = '../data/'

#    dataset = gzip.open('dataset.pz')
    keys = ['throttle', 'brake', 'steering', 'speed', 'frame']
    total_num = 493531
    seq_len = 64
    num_class = {'throttle': 20, 'brake': 20, 'steering': 20, 'speed': 20}


    totensor = transforms.ToTensor()
#    mean, std = get_mean_std()
#    y_raw = get_raw_y()
#    all_bins = find_even_bins(num_class)


    mean = [0.35739973, 0.35751262, 0.36058474]
    std = [0.19338858, 0.19159749, 0.2047393 ]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

#    split_dataset()
#    split_train_val()
#    save_bins_info()

