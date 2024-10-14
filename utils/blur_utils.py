import os
import math
import torch
import numpy as np
import scipy.special

from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

from PIL import Image

from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
from utils.Demosaicing_malvar2004_pytorch import Demosaic

demosaic_fn = Demosaic()


def compute_avg_distance_cams(views, scale=0.01):
    deltas = []
    for idx, view in enumerate(views):
        xyz_cur = views[idx].camera_center
        xyz_next = views[idx+1].camera_center

        dist_i = torch.norm(xyz_cur-xyz_next)
        deltas.append(dist_i)
        
        if idx+1 >= len(views)-1:
            break
    d_xyz = torch.mean(torch.stack(deltas)) * scale
    
    return d_xyz.cpu().numpy()


def compute_bezier_coeff(curve_order, binom_coeff, t):
    bezier_coeff = []
    for i in range(curve_order+1):
        coeff_i = binom_coeff[i] * np.power(1-t, curve_order-i) * np.power(t, i)
        bezier_coeff.append(coeff_i)

    bezier_coeff = np.stack(bezier_coeff, axis=-1)
    bezier_coeff = torch.from_numpy(bezier_coeff)

    return bezier_coeff


def gen_blur_trajectory(camera, d_xyz, d_theta, ksize=31, forward_motion=False, dynamic=False):
    R = camera.R
    T = camera.T
    
    fovx = camera.FoVx
    fovy = camera.FoVy

    znear = camera.znear
    zfar = camera.zfar
    
    trans = camera.trans
    scale = camera.scale

    if np.sum(trans) !=0 or scale != 1.0:
        assert ValueError

    curve_order = np.random.randint(1,5)
    world_view_transform_traj = gen_random_bezier_curve(R, T, trans, scale, d_xyz, d_theta, ksize, curve_order=curve_order, forward_motion=forward_motion, dynamic=dynamic)
    projection_matrix_i = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0,1).cuda()

    view_traj = []
    
    for i in range(ksize):
        world_view_transform_i = world_view_transform_traj[i].cuda()
        full_proj_transform_i = (world_view_transform_i.unsqueeze(0).bmm(projection_matrix_i.unsqueeze(0))).squeeze(0)

        view_i = MiniCam(width=camera.image_width, height=camera.image_height,
                         fovy=fovy, fovx=fovx, znear=znear, zfar=zfar,
                         world_view_transform=world_view_transform_i,
                         full_proj_transform=full_proj_transform_i)
        
        view_traj.append(view_i)
    
    return view_traj


def gen_random_bezier_curve(R, T, trans, scale, d_xyz, d_theta, ksize, curve_order, forward_motion, dynamic):
    transform = torch.tensor(getWorld2View2(R, T, trans, scale))
    
    if not forward_motion:
        translation = np.concatenate([np.random.uniform(-1., 1., size=2), np.random.uniform(-1/10, 1/10, size=1)])
    else:
        translation = np.concatenate([np.random.uniform(-1., 1., size=2), np.random.uniform(-1., 1., size=1)])

    if dynamic==True:
        translation = np.concatenate([np.random.uniform(-1., 1., size=1), np.random.uniform(-1, 1, size=1), np.random.uniform(-1/5, 1/5, size=1)])

    random_scale = (np.random.uniform(0, 0.66*d_xyz) + np.random.uniform(0.33*d_xyz, d_xyz)) / 2
    translation = translation / np.linalg.norm(translation) * random_scale
    
    norm_tanslation = np.linalg.norm(translation)
    rotation = np.random.uniform(-d_theta*np.pi, d_theta*np.pi, size=3)
    norm_rotation = np.linalg.norm(rotation)

    t = np.linspace(0, 1, ksize)
    binom_coeff = [scipy.special.binom(curve_order, k) for k in range(curve_order+1)]
    
    bezier_coeff = compute_bezier_coeff(curve_order, binom_coeff, t)

    # Generate random control points.
    control_points_location = []
    control_points_rotation = []

    for i in range(curve_order+1):
        w = (i - curve_order / 2) / curve_order
    
        trans_i = translation * w
        rot_i = rotation * w
        # generate random translation and rotation
        if not dynamic:
            random_trans_i = np.concatenate([np.random.uniform(-norm_tanslation / 3, norm_tanslation / 3, size=2),
		                                 np.random.uniform(-norm_tanslation / 3, norm_tanslation / 3, size=1)])
        else:
            random_trans_i = np.concatenate([np.random.uniform(-norm_tanslation / 20, norm_tanslation / 20, size=2),
		                                 np.random.uniform(-norm_tanslation / 20, norm_tanslation / 20, size=1)])
        
        random_rotation_i = np.random.uniform(-norm_rotation / 6, norm_rotation / 6, size=3)
        
        control_points_location.append(trans_i + random_trans_i)
        control_points_rotation.append(rot_i + random_rotation_i)
    
    control_points_location = np.stack(control_points_location)
    control_points_rotation = np.stack(control_points_rotation)
    
    control_points = np.concatenate([control_points_location, control_points_rotation], axis=-1)
    control_points = torch.from_numpy(control_points)

    control_points_repeat = control_points[None].repeat(ksize, 1, 1)    
    control_points_repeat = bezier_coeff.unsqueeze(-1) * control_points_repeat

    control_points_mat = vectoMat(control_points_repeat.reshape(-1, 6)).reshape(ksize, curve_order+1, 4, 4)

    curve_mat = torch.eye(4)[None].repeat(ksize, 1, 1)
    for i_order in range(curve_order+1):
        curve_mat = torch.matmul(control_points_mat[:, i_order], curve_mat)
    
    idx_center = ksize//2
    curve_center = curve_mat[idx_center]

    # world-to-camera system.
    curve_mat = torch.matmul(curve_mat, torch.linalg.inv(curve_center[None]))
    
    world_view_transform_traj = torch.matmul(curve_mat, transform[None].repeat(ksize, 1, 1))
    world_view_transform_traj = world_view_transform_traj.transpose(1, 2)

    return world_view_transform_traj


def vectoMat(vec):
    '''
    input:  vector  (-1, 6)
    output: matrix  (-1, 4, 4)
    '''

    # vec: (x, y, z, pi, chi, psi)
    cos_pi = torch.cos(vec[:, 3])
    sin_pi = torch.sin(vec[:, 3])

    cos_chi = torch.cos(vec[:, 4])
    sin_chi = torch.sin(vec[:, 4])

    cos_psi = torch.cos(vec[:, 5])
    sin_psi = torch.sin(vec[:, 5])

    mat = torch.zeros(vec.shape[0], 4, 4).float()

    # 3D Rotation.
    mat[:, 0, 0] = cos_pi * cos_chi
    mat[:, 0, 1] = cos_pi * sin_chi * sin_psi - sin_pi * cos_psi
    mat[:, 0, 2] = cos_pi * sin_chi * cos_psi + sin_pi * sin_psi

    mat[:, 1, 0] = sin_pi * cos_chi
    mat[:, 1, 1] = sin_pi * sin_chi * sin_psi + cos_pi * cos_psi
    mat[:, 1, 2] = sin_pi * sin_chi * cos_psi - cos_pi * sin_psi

    mat[:, 2, 0] = -sin_chi
    mat[:, 2, 1] = cos_chi * sin_psi
    mat[:, 2, 2] = cos_chi * cos_psi

    # Translation.
    mat[:, 0, 3] = vec[:, 0]
    mat[:, 1, 3] = vec[:, 1]
    mat[:, 2, 3] = vec[:, 2]

    mat[:, 3, 3] = 1.

    return mat


def Mat2Vec(mat):
    '''
    input:  matrix  (-1, 3, 4)
    output: vector  (-1, 6)
    '''

    input_shape = mat.shape

    # vec: (x, y, z, pi, chi, psi)
    vec = torch.zeros(input_shape[0], 6).float() #.to(device=device)

    # (x,y,z)
    vec[:, 0] = mat[:, 0, 3]  # x
    vec[:, 1] = mat[:, 1, 3]  # y
    vec[:, 2] = mat[:, 2, 3]  # z

    # (pi, chi, psi)
    vec[:, 3] = torch.atan2(mat[:, 1, 0], mat[:, 0, 0])  # pi
    vec[:, 4] = torch.atan2( -mat[:, 2, 0], torch.sqrt( mat[:, 0, 0] ** 2 + mat[:, 1, 0] ** 2 ))  # chi
    vec[:, 5] = torch.atan2(mat[:, 2, 1], mat[:, 2, 2])  # psi
    
    # when |chi| == 90
    for i in range(input_shape[0]):
        if mat[i, 2, 0] == -1:
            vec[i, 3] = torch.atan2(mat[i, 1, 2], mat[i, 0, 2])
            vec[i, 5] = 0
        elif mat[i, 2, 0] == 1:
            vec[i, 3] = torch.atan2(-mat[i, 1, 2], -mat[i, 0, 2])
            vec[i, 5] = 0

    return vec 


def normalize_depth(depth):
    min_depth = torch.min(depth)
    max_depth = torch.max(depth)

    depth_normalized = (depth - min_depth) / (max_depth - min_depth)

    return depth_normalized


def load_mask(scene, mask_path, H, W):
    model_path = scene.model_path
    data_name = model_path.split('/')[-1]
    class_id, split_id, scene_id = data_name.split('_')
    
    mask_path = os.path.join(mask_path, str(class_id), scene_id)
    
    views = scene.getTrainCameras()

    mask_list = []
    for view in views:
        image_name = view.image_name+'.jpg.png'
        image_dir = os.path.join(mask_path, image_name)

        mask_i = np.array(Image.open(image_dir).resize((W, H)))
        mask_i = torch.from_numpy(mask_i).cuda()
        mask_list.append(mask_i)
    
    return mask_list


def compute_blur_mask(mask, xy_center):
    H, W = mask.shape
    
    y_center = torch.clip(xy_center[1], 0, H - 1).long()
    x_center = torch.clip(xy_center[0], 0, W - 1).long()
    
    mask_i = mask[y_center, x_center].reshape(H, W)

    torch.set_default_device('cpu')

    return mask_i


def compute_warp3D(depth, camera, camera_center, dscale=16):
    torch.set_default_device('cuda')
    w2c_i = camera.world_view_transform.transpose(1, 0)
    w2c_center = camera_center.world_view_transform.transpose(1, 0)
    
    fovX = camera.FoVx
    fovY = camera.FoVy
    
    H = camera.image_height
    W = camera.image_width
    f_y = (H / 2) / math.tan((fovY / 2))
    f_x = (W / 2) / math.tan((fovX / 2))

    K = torch.tensor([[f_x, 0, (W / 2)], [0, f_y, (H / 2)], [0, 0, 1.]])#.cuda()
    
    # Generate pixel coordinates.
    xy = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='xy')
    
    xy = torch.stack(xy)

    if dscale > 1:
        xy = torch.nn.functional.pad(xy, (0,dscale//2,0,dscale//2), "replicate")
        xy = xy[:, dscale//2::dscale, dscale//2::dscale]

        depth = torch.nn.functional.pad(depth, (0,dscale//2,0,dscale//2), "replicate")
        depth = depth[:, dscale//2::dscale, dscale//2::dscale]

    H_down, W_down = xy.shape[1:]
    xy_h = torch.cat([xy, torch.ones_like(xy[0:1])], dim=0)

    xyz_cam = torch.matmul(torch.inverse(K), xy_h.reshape(3, -1))
    xyz_cam = xyz_cam * depth.reshape(1, -1)

    xyzh_cam = torch.cat([xyz_cam, torch.ones_like(xyz_cam[0:1])], dim=0)

    xyzh_world = torch.matmul(w2c_center, torch.matmul(torch.inverse(w2c_i), xyzh_cam))
    xyz_center = torch.matmul(K, xyzh_world[:3])
    xy_center = xyz_center[:2] / xyz_center[2:3]

    xy_center_delta = xy_center.reshape(2, H_down, W_down) - xy

    return xy_center, xy_center_delta


def compute_blurred_dynamic(rgb_blurred, rgb_blurred_bg, mask):
    mask_rgb = mask[None].repeat(3, 1, 1)
    
    # soft
    rgb_blurred_sb = mask_rgb * rgb_blurred + (1-mask_rgb) * rgb_blurred_bg
    # hard
    #rgb_blurred_sb = torch.where(mask_rgb, rgb_blurred, rgb_blurred_bg)
    
    return rgb_blurred_sb


def compute_blurred_dynamic_kernel(kernel, kernel_bg, mask, dscale=16):
    mask = mask > 0
    K = kernel.shape[0]

    mask = torch.nn.functional.pad(mask[None].float(), (0,dscale//2,0,dscale//2), "replicate")[0].bool()
    mask_rgb = mask[None, None, dscale//2::dscale, dscale//2::dscale].repeat(K, 2, 1, 1)

    # soft
    #rgb_blurred_sb = mask_rgb * rgb_blurred + (1-mask_rgb) * rgb_sharp
    # hard
    kernel_sb = torch.where(mask_rgb, kernel, kernel_bg)
    
    return kernel_sb


def vis_kernel_for_debug(image, kernel):
    _, H, W = image.shape

    stride = 32
    window_size = 100

    output = torch.zeros((window_size*(H//stride + 1), window_size*(W//stride + 1)))
    
    image_hwc = image.clone()
    
    for ix, x_i in enumerate(range(0, W-1, stride)):
        for iy, y_i in enumerate(range(0, H-1, stride)):
            window_i = torch.zeros(100, 100).long()
            window_i[window_size//2, window_size//2] = 255
            window_i[:, window_size-1] = 255
            window_i[:, 0] = 255
            window_i[0, :] = 255
            window_i[window_size-1, :] = 255
            
            kx = torch.clip(kernel[:, 0, y_i//stride, x_i//stride]+window_size//2, 0, window_size-1).long()
            ky = torch.clip(kernel[:, 1, y_i//stride, x_i//stride]+window_size//2, 0, window_size-1).long()

            window_i[kx, ky] = 255

            output[window_size*iy:window_size*(iy+1), window_size*ix:window_size*(ix+1)] = window_i

            kx_image = torch.clip(x_i+kernel[:, 0, y_i//stride, x_i//stride], 0, W-1).long()
            ky_image = torch.clip(y_i+kernel[:, 1, y_i//stride, x_i//stride], 0, H-1).long()
            
            image_hwc[0, ky_image, kx_image] = 1.
            image_hwc[1, ky_image, kx_image] = 0.
            image_hwc[2, ky_image, kx_image] = 0.

    
    return image_hwc


def rgb2lin(x):
    gamma = 2.4
    a = 1 / 1.055
    b = 0.055 / 1.055
    c = 1 / 12.92
    d = 0.04045

    in_sign = -2 * torch.lt(x, 0) + 1
    abs_x = torch.abs(x)

    lin_range = torch.lt(abs_x ,d)
    gamma_range = torch.logical_not(lin_range)

    lin_value = (c * abs_x)
    gamma_value = torch.exp(gamma * torch.log(a * abs_x + b))

    new_x = (lin_value * lin_range) + (gamma_value * gamma_range)
    new_x = new_x * in_sign

    return new_x

def lin2rgb(x):
    gamma = 1/2.4
    a = 1.055
    b = -0.055
    c = 12.92
    d = 0.0031308

    in_sign = -2 * torch.lt(x, 0) + 1
    abs_x = torch.abs(x)

    lin_range = torch.lt(abs_x ,d)
    gamma_range = torch.logical_not(lin_range)

    lin_range = lin_range
    gamma_range = gamma_range

    lin_value = (c * abs_x)
    gamma_value = a * torch.exp(gamma * torch.log(abs_x)) + b
    new_x = (lin_value * lin_range) + (gamma_value * gamma_range)
    new_x = new_x * in_sign

    return new_x


# Color conversion code
def lin2xyz(rgb): # rgb from [0,1]
    mat = torch.tensor([[0.412453, 0.357580, 0.180423],
                        [0.212671, 0.715160, 0.072169],
                        [0.019334, 0.119193, 0.950227]], device='cuda')

    rgb_vec = rgb[:,:,None,:]
    ccms = mat[None, None]
    out = torch.sum(rgb_vec*ccms, -1)
    
    return out


def xyz2lin(xyz):    
    mat =  torch.tensor([[ 3.24048134, -1.53715152, -0.49853633],
                         [-0.96925495,  1.87599   ,  0.04155593],
                         [ 0.05564664, -0.20404134,  1.05731107]],device='cuda')

    xyz_vec = xyz[:,:,None,:]
    ccms = mat[None, None]
    out = torch.sum(xyz_vec*ccms, -1)

    return out
    

def cam2xyz(img):
    # img : (h, w, c)
    # matrix : (3, 3)

    """
    same results below code
    img_reshape = img.reshape(1, h*w, 3)
    out2 = torch.matmul(img_reshape, matrix.permute(0, 2, 1))
    out2 = out2.reshape(1, h, w, 3)
    """
    matrix = torch.tensor([[ 0.66509233,  0.25284925,  0.03252843],
                           [ 0.2612683 ,  0.96594937, -0.22721767],
                           [ 0.03421707, -0.20912425,  1.26373718]], device='cuda')
    images = img[:,:,None,:] # (h, w, 1, c)
    ccms = matrix[None, None, :, :] # (1, 1, 3, 3)
    out = torch.sum(images * ccms, -1) # (h, w, 3)

    return out


def xyz2cam(img):
    # img : (h, w, c)
    # matrix : (3, 3)

    """
    same results below code
    img_reshape = img.reshape(1, h*w, 3)
    out2 = torch.matmul(img_reshape, matrix.permute(0, 2, 1))
    out2 = out2.reshape(1, h, w, 3)
    """
    matrix = torch.tensor([[ 1.69541757, -0.47160184, -0.12843299],
                           [-0.48838234,  1.21303059,  0.23067161],
                           [-0.12672319,  0.21350242,  0.83295296]], device='cuda')
    images = img[:,:,None,:] # (h, w, 1, c)
    ccms = matrix[None, None, :, :] # (1, 1, 3, 3)
    out = torch.sum(images * ccms, -1) # (h, w, 3)

    return out


def mosaic_bayer(image, pattern):

    """Extracts RGGB Bayer planes from an RGB image."""
    shape = image.shape # (h, w, c)

    if pattern == 'RGGB':
        red = image[0::2, 0::2, 0] # (h/2, w/2)
        green_red = image[0::2, 1::2, 1]
        green_blue = image[1::2, 0::2, 1]
        blue = image[1::2, 1::2, 2]
    elif pattern == 'BGGR':
        red = image[0::2, 0::2, 2] # (h/2, w/2)
        green_red = image[0::2, 1::2, 1]
        green_blue = image[1::2, 0::2, 1]
        blue = image[1::2, 1::2, 0]
    elif pattern == 'GRBG':
        red = image[0::2, 0::2, 1] # (h/2, w/2)
        green_red = image[0::2, 1::2, 0]
        green_blue = image[1::2, 0::2, 2]
        blue = image[1::2, 1::2, 1]
    elif pattern == 'GBRG':
        red = image[0::2, 0::2, 1] # (h/2, w/2)
        green_red = image[0::2, 1::2, 2]
        green_blue = image[1::2, 0::2, 0]
        blue = image[1::2, 1::2, 1]

    image = torch.stack((red, green_red, green_blue, blue), dim=2)  # (h/2, w/2, 4)
    image = image.view(shape[0] // 2, shape[1] // 2, 4)

    return image


def WB_img(img, pattern, fr_now, fb_now):

    red_gains = fr_now
    blue_gains = fb_now
    green_gains = 1.0

    if pattern == 'RGGB':
        gains = torch.tensor([red_gains, green_gains, green_gains, blue_gains], device='cuda').float()
    elif pattern == 'BGGR':
        gains = torch.tensor([blue_gains, green_gains, green_gains, red_gains], device='cuda').float()
    elif pattern == 'GRBG':
        gains = torch.tensor([green_gains, red_gains, blue_gains, green_gains], device='cuda').float()
    elif pattern == 'GBRG':
        gains = torch.tensor([green_gains, blue_gains, red_gains, green_gains], device='cuda').float()

    gains = gains[None, None, :]
    img = img * gains

    return img


def add_Poisson_noise_random(img, beta1, beta2):

    h, w, c = img.shape

    # bsd : 2.3282e-05, my : 0.0001
    min_beta1 = beta1 * 0.5
    random_K_v = min_beta1 + torch.rand(1) * (beta1 * 1.5 - min_beta1)
    random_K_v = random_K_v.view(1, 1, 1).to(img.device)

    noisy_img = torch.poisson(img / random_K_v)
    noisy_img = noisy_img * random_K_v

    # bsd : 1.9452e-04, my : 9.1504e-04
    min_beta2 = beta2 * 0.5
    random_other = min_beta2 + torch.rand(1) * (beta2 * 1.5 - min_beta2)
    random_other = random_other.view(1, 1, 1).to(img.device)

    noisy_img = noisy_img + (torch.normal(torch.zeros_like(noisy_img), std=1)*random_other)

    return noisy_img

def realistic_blur_synthesis(img_blur):#, img_gt):
    img_blur = img_blur.permute(1,2,0)

    mask_sat = torch.any(img_blur==1.0, dim=-1, keepdim=True)
    alpha_sat = np.random.uniform(0.25, 1.75)

    img_blur = img_blur + (alpha_sat * mask_sat)
    img_blur = torch.clamp(img_blur, 0, 1.)
    
    blurred_sat = img_blur.clone()

    img_XYZ = lin2xyz(img_blur)

    img_Cam = xyz2cam(img_XYZ)
    """
    # estimated noise for the RealBlur dataset
    beta1 = 1e-04 #8.8915e-05
    beta2 = 3e-05 #2.9430e-05
    
    random_uniform = np.random.uniform(0., 1.)
    scale_noise1 = scale_noise2 = 0.1* 10**random_uniform

    beta1 = scale_noise1 * beta1 #np.random.uniform(beta1*0.05, beta1)
    beta2 = scale_noise2 * beta2 #np.random.uniform(beta2*0.05, beta2)
    #import pdb; pdb.set_trace()
    #img_debug = add_Poisson_noise_random(img_mosaic, 1e-7, 1e-7)
    #torch.max(img_debug-img_mosaic)
    img_Cam = add_Poisson_noise_random(img_Cam, beta1, beta2)
    img_IXYZ = cam2xyz(img_Cam)
    """
    bayer_pattern = np.random.choice(['RGGB', 'BGGR', 'GRBG', 'GBRG'])
    img_mosaic = mosaic_bayer(img_Cam, bayer_pattern)

    # inverse white balance
    red_gain = np.random.uniform(1.9, 2.4)
    blue_gain = np.random.uniform(1.5, 1.9)
    img_mosaic = WB_img(img_mosaic, bayer_pattern, 1 / red_gain, 1 / blue_gain)

    # -------- ADDING POISSON-GAUSSIAN NOISE ON RAW -
    # estimated noise for the RealBlur dataset
    beta1 = 8.8915e-05
    beta2 = 2.9430e-05
    
    random_uniform = np.random.uniform(0., 1.)
    scale_noise1 = scale_noise2 = 0.1* 10**random_uniform

    beta1 = scale_noise1 * beta1
    beta2 = scale_noise2 * beta2

    img_mosaic_noise = add_Poisson_noise_random(img_mosaic, beta1, beta2)
    # -------- ISP PROCESS --------------------------
    # White balance
    img_demosaic = WB_img(img_mosaic_noise, bayer_pattern, red_gain, blue_gain)
    # demosaic
    img_demosaic = torch.nn.functional.pixel_shuffle(img_demosaic.permute(2, 0, 1).unsqueeze(0), 2)
    img_demosaic = torch.from_numpy(demosaicing_CFA_Bayer_Menon2007(img_demosaic[0,0].cpu().numpy(), pattern=bayer_pattern)).cuda()
    
    # from Cam to XYZ
    img_IXYZ = cam2xyz(img_demosaic)
    
    # frome XYZ to linear RGB
    img_IL = xyz2lin(img_IXYZ)

    # tone mapping
    img_Irgb = lin2rgb(img_IL)
    img_Irgb = torch.clamp(img_Irgb, 0, 1)  # (h, w, c)

    blurred = img_Irgb
    
    # don't add noise on saturated region
    sat_region = torch.ge(blurred_sat, 1.0)
    non_sat_region = torch.logical_not(sat_region)
    blurred = (blurred_sat * sat_region) + (blurred * non_sat_region)

    # Adopt a7r3 CRF
    #gt = lin2rgb_a7r3_polynomial(rgb2lin(gt))
    #blurred = lin2rgb_a7r3_polynomial(rgb2lin(blurred))

    blurred = blurred.permute(2, 0, 1)
    return blurred

