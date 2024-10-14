import numpy as np
import glob
import os
import random
split_list = ['mvi_selected99', 'mvi_selected98']
#['mvi_selected0', 'mvi_selected1', 'mvi_selected2', 'mvi_selected3', 'mvi_selected4']


for split_name in split_list:
    
    scene_list = sorted(glob.glob(os.path.join(split_name, '*')))
    random.shuffle(scene_list)

    with open(os.path.join(split_name, 'shuffled_list.txt'), 'w') as fp:
        for scene_name in scene_list:
            print(scene_name)
            fp.write('{}\n'.format(scene_name.split('/')[-1]))
