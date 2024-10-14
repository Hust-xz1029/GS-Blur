import os
import numpy as np
import imageio
import glob

def gen_image_ids(datadir):
    all_images = [x for x in glob.glob(os.path.join(datadir, "images/*")) if os.path.isfile(x)]
    all_images.sort()

    file = open(os.path.join(datadir, 'train.txt'), 'w')

    for i, name in enumerate(all_images):
        name = name.split('/')[-1]
        if i < len(all_images) - 1:
            file.write(name+'\n')
        else:
            file.write(name)
    file.close()

    file = open(os.path.join(datadir, 'test.txt'), 'w')
    file.close()


basedir = 'datasets/MVImgNet'
selectedscenedir = 'datasets/MVImgNet/mvi_selected'
split_list =['mvi_29', 'mvi_39'] #['mvi_00', 'mvi_01', 'mvi_02', 'mvi_03', 'mvi_05']
classes = [6, 8, 14, 17, 26, 28, 29, 36, 37, 38, 39, 41, 44, 45, 46, 52, 93, 99,
            112, 137, 152, 155, 156, 157, 158, 159] #144, 48, 22, 27, 49,

for split in split_list:
    splitdir = os.path.join(basedir, split)
    class_list = sorted(glob.glob(os.path.join(splitdir, '*')))
    splitidx = int(split[-2:])
    selectedscenedir_i = os.path.join(basedir, f'mvi_selected{splitidx}')
    os.makedirs(selectedscenedir_i)
    
    for collector in class_list:
        scene_list = sorted(glob.glob(os.path.join(collector, '*')))

        for scene in scene_list:
            
            img_list = sorted(glob.glob(os.path.join(scene, 'images', '*')))
            img_thumb = img_list[0]
            
            assert os.path.exists(img_thumb)
            id_split = split.split('/')[-1].replace('mvi_', '')
            id_class = collector.split('/')[-1]
            id_scene = scene.split('/')[-1]

            out_scene_dir = os.path.join(selectedscenedir_i, '{}_{}_{}'.format(id_class, id_split, id_scene))
            check_cam_exists = os.path.exists(os.path.join(scene, 'sparse/0/cameras.bin'))
            if int(id_class) in classes and check_cam_exists:
                print(scene)
                try:
                    os.system('cp -r {} {}'.format(scene, out_scene_dir))
                    os.system('rm -rf {}/images/*_bg_removed.png'.format(out_scene_dir))
                    gen_image_ids(out_scene_dir)
                except:
                    os.system('rm -rf {}'.format(out_scene_dir))
