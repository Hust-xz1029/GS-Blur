#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from utils.image_utils import psnr
from utils.blur_utils import *

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, scale, d_theta, repeats, ksize, render_path, masks, save_kernel, resolution):
    
    scene_name = model_path.split('/')[-1]

    # Set output path.
    blurred_path = os.path.join(render_path, "input")
    blurred_noise_path = os.path.join(render_path, "input_noise")
    sharp_path = os.path.join(render_path, "target")
    kernel_path = os.path.join(render_path, "kernel")
    kernel_vis_path = os.path.join(render_path, "kernel_vis")

    blurred_failed_path = os.path.join(render_path, "failed", "input")
    blurred_noise_failed_path = os.path.join(render_path, "failed", "input_noise")
    sharp_failed_path = os.path.join(render_path, "failed", "sharp")
    kernel_failed_path = os.path.join(render_path, "failed", "kernel")
    kernel_failed_vis_path = os.path.join(render_path, "failed", "kernel_vis")

    # Make folders for outputs.
    makedirs(blurred_path, exist_ok=True)
    makedirs(blurred_noise_path, exist_ok=True)
    makedirs(sharp_path, exist_ok=True)
    makedirs(blurred_failed_path, exist_ok=True)
    makedirs(blurred_noise_failed_path, exist_ok=True)
    makedirs(sharp_failed_path, exist_ok=True)
    
    if save_kernel:
        makedirs(kernel_path, exist_ok=True)
        makedirs(kernel_vis_path, exist_ok=True)
        makedirs(kernel_failed_path, exist_ok=True)
        makedirs(kernel_failed_vis_path, exist_ok=True)
    
    # Compute scale of camera trajectory based on averaged distance between training cams.
    d_xyz = compute_avg_distance_cams(views, scale=scale)
    
    render_list = []
    gt_list = []

    # Set file name.
    if masks is None:
        img_name_base = '{0}_{1:02d}_r{2}_{3}_{4}'
    else:
        img_name_base = '{0}_{1:02d}_r{2}_{3}_dy_{4}'

    # Iterate over random camera motion trajectory.
    for i_repeat in range(repeats):
        # Iterate over training views.
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            # When foreground mask is given, generate both fg /bg trajectory.
            if masks is None:
                views_traj_i = gen_blur_trajectory(view, d_xyz, d_theta, ksize=ksize, forward_motion=False)
            else:
                scale_xyz_bg = np.random.uniform(0. ,0.1)
                scale_rot_bg = np.random.uniform(0. ,0.1)
                views_traj_bg_i = gen_blur_trajectory(view, scale_xyz_bg*d_xyz, scale_rot_bg*d_theta, ksize=ksize, forward_motion=False)
                views_traj_i = gen_blur_trajectory(view, d_xyz, d_theta, ksize=ksize, forward_motion=True, dynamic=True)

            rendering_traj_i = []
            rendering_traj_bg_i = []
            mask_traj_i = []
            kernel_traj_i = []
            kernel_traj_bg_i = []

            # Render camera views sampled from random trajectory.
            for idx_view, view_i in enumerate(views_traj_i):

                render_output_i = render(view_i, gaussians, pipeline, background)
                rendering_i = render_output_i["render"]
                
                if idx_view == ksize//2:
                    render_center = rendering_i.clone()
                
                rendering_i = rgb2lin(rendering_i)
                rendering_traj_i.append(rendering_i)
                
                if masks is not None:
                    render_output_bg_i = render(views_traj_bg_i[idx_view], gaussians, pipeline, background)
                    rendering_bg_i = render_output_bg_i["render"]

                    rendering_bg_i = rgb2lin(rendering_bg_i)
                    rendering_traj_bg_i.append(rendering_bg_i)

                    xy_center_i, xy_kernel_i = compute_warp3D(render_output_i["depth"], view_i, views_traj_i[ksize//2])

                    mask_i = compute_blur_mask(masks[idx], xy_center_i)
                    mask_traj_i.append(mask_i)
                    
                if save_kernel and idx_view % 2==0:
                    xy_center_i, xy_kernel_i = compute_warp3D(render_output_i["depth"], view_i, views_traj_i[ksize//2])
                    kernel_traj_i.append(xy_kernel_i)

                    if masks is not None:
                        xy_center_bg_i, xy_kernel_bg_i = compute_warp3D(render_output_bg_i["depth"], views_traj_bg_i[idx_view], views_traj_bg_i[ksize//2])
                        kernel_traj_bg_i.append(xy_kernel_bg_i)

            gt = view.original_image[0:3, :, :]

            # Synthesize blurry image by aggregating sharp renderings along the trajectory.
            rendering_blurred_lin = torch.mean(torch.stack(rendering_traj_i), dim=0)            
            #rendering_blurred_clean = lin2rgb(rendering_blurred_lin.clone())
            rendering_blurred_noise = realistic_blur_synthesis(rendering_blurred_lin,)

            if i_repeat == 0:
                render_list.append(render_center)
                gt_list.append(gt)

            # Use center image as a ground truth sharp pair.
            render_center = lin2rgb(rgb2lin(render_center.clone()))

            # Synthesize blurry image of background.
            if masks is not None:
                rendering_blurred_bg_lin = torch.mean(torch.stack(rendering_traj_bg_i), dim=0)
                #rendering_blurred_bg_clean = lin2rgb(rendering_blurred_bg_lin.clone())
                rendering_blurred_bg_noise = realistic_blur_synthesis(rendering_blurred_bg_lin,)

                mask_blurred = torch.mean(torch.stack(mask_traj_i).float(), dim=0) /255.
                mask_blurred = torch.sqrt(mask_blurred)
                
                rendering_blurred_noise = compute_blurred_dynamic(rendering_blurred_noise, rendering_blurred_bg_noise, mask_blurred)

            # Save rendered output and pixel-wise blur kernels (default stride: 16 pixels).
            flag_save = True
            if save_kernel:
                kernel = torch.stack(kernel_traj_i)
                #kernel_vis = vis_kernel_for_debug(rendering_blurred_clean, kernel)
                
                if masks is not None:
                    kernel_bg = torch.stack(kernel_traj_bg_i)
                    kernel = compute_blurred_dynamic_kernel(kernel, kernel_bg, mask_blurred)
                    #kernel_vis = vis_kernel_for_debug(rendering_blurred_clean, kernel)
                
                kernel = kernel.to(torch.float16).cpu()
                if torch.sum(torch.isinf(kernel)) >0 or torch.sum(torch.isnan(kernel)) >0:
                    flag_save = False
                else:
                    torch.save(kernel, os.path.join(kernel_path, img_name_base.format(scene_name, idx, resolution, scale, i_repeat)+'.pt'))
                    #torchvision.utils.save_image(kernel_vis, os.path.join(kernel_vis_path, img_name_base.format(scene_name, idx, resolution, scale, i_repeat)+".png"))
            if flag_save:
                #torchvision.utils.save_image(rendering_blurred_clean, os.path.join(blurred_path, img_name_base.format(scene_name, idx, resolution, scale, i_repeat) + ".png"))
                torchvision.utils.save_image(rendering_blurred_noise, os.path.join(blurred_noise_path, img_name_base.format(scene_name, idx, resolution, scale, i_repeat) + ".png"))
                torchvision.utils.save_image(render_center, os.path.join(sharp_path, img_name_base.format(scene_name, idx, resolution, scale, i_repeat) + ".png"))

        # Filter out mechanism for the scenes that are failed to reconsturct 3D scene.
        if i_repeat==0:
            render_all_view = torch.stack(render_list)
            gt_all_view = torch.stack(gt_list)

            psnr_list = psnr(render_all_view, gt_all_view)
            psnr_mean = psnr_list.mean()
            psnr_min = psnr_list.min()
            psnr_diff = psnr_mean - psnr_min
            print(f'psnr mean - min: {psnr_mean-psnr_min:.4f}')

            if psnr_diff > 3. and masks is None and resolution == 2:
                with open(os.path.join(render_path, f'psnr_diff.txt'), 'a') as fp:
                    fp.write(f'{scene_name}  PSNR: {psnr_diff}\n')
                    fp.close()

                with open(os.path.join(model_path, f'Failed_scene'), 'w') as fp:
                        fp.write(' ')
                        fp.close()
                
                blur_mv_path = os.path.join(blurred_path, f'{scene_name}_*')
                blur_noise_mv_path = os.path.join(blurred_noise_path, f'{scene_name}_*')
                sharp_mv_path = os.path.join(sharp_path, f'{scene_name}_*')
                
                os.system(f'mv {blur_mv_path} {blurred_failed_path}/')
                os.system(f'mv {blur_noise_mv_path} {blurred_noise_failed_path}/')
                os.system(f'mv {sharp_mv_path} {sharp_failed_path}/')

                if save_kernel:
                    kernel_mv_path = os.path.join(kernel_path, f'{scene_name}_*') 
                    kernel_vis_path = os.path.join(sharp_path, f'{kernel_vis_path}_*')
                    
                    os.system(f'mv {kernel_mv_path} {kernel_failed_path}/')
                    os.system(f'mv {kernel_vis_path} {kernel_failed_vis_path}/')
                return


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, scale : float, d_theta : float, repeats : int, ksize : int, output_path : str, mask_path : str, expname : str, save_kernel : bool, resolution : int):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        img_h, img_w = scene.getTrainCameras()[0].original_image[0:3, :, :].shape[1:]
        if img_h < 256 or img_w < 256:
            return

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if mask_path != "":
            print('load mask')
            masks = load_mask(scene, mask_path, img_h, img_w)
        else:
            masks = None

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, scale, d_theta, repeats, ksize, output_path, masks, save_kernel, resolution)

             with open(os.path.join(dataset.model_path, f'Done_render_{expname}'), 'w') as fp:
                fp.write(' ')
                fp.close()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--blur", action="store_true")
    parser.add_argument("--repeats", default=5, type=int)
    parser.add_argument("--ksize", default=121, type=int)
    parser.add_argument("--scale", default=0.5, type=float)
    parser.add_argument("--d_theta", default=0.003, type=float)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--expname", type=str)
    parser.add_argument("--save_kernel", action='store_true')
    parser.add_argument("--mask_path", type=str, default="")
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Do not render again if rendering is already done.
    if os.path.exists(os.path.join(args.model_path, f'Done_render_{args.expname}')):
        print(f"{args.model_path}, rendering already processed")
    elif os.path.exists(os.path.join(args.model_path, 'Failed_scene')):
        print(f"{args.model_path}, 3D GS Failed")
    else:
        # Initialize system state (RNG)
        safe_state(args.quiet)
        print(len([model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.scale, args.d_theta, args.repeats, args.ksize, args.output_path, args.mask_path, args.expname, args.save_kernel, args.resolution]))
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.scale, args.d_theta, args.repeats, args.ksize, args.output_path, args.mask_path, args.expname, args.save_kernel, args.resolution)