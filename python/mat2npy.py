import scipy.io as io
import numpy as np
import os


load_path = '../processed_csi/'
save_path = '../dataset/'
files = os.listdir(load_path)
m = {'empty': 0, 'walk': 1, 'sit': 2, 'stand': 3, 'stooll': 4, 'stoolr': 5, 'stoolf': 6}

cnt = [0]*len(m)

for key in m.keys():
    if not os.path.exists(save_path + key):
        os.makedirs(save_path + key)

for it in files:
    tmp = it.split('_')
    motion = tmp[0]
    matr = io.loadmat(load_path + it)
    save_name = motion + '_' + str(cnt[m[motion]]) + '.npy'
    cnt[m[motion]] += 1
    np.save(save_path + motion + '/' + save_name, matr['csi_segment'])

print("Totally converts %d .mat files in to numpy array" % (len(files)))


