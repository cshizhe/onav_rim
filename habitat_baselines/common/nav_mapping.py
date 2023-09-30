import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import habitat.utils.depth_utils as du


def get_local_map_boundaries(
    agent_loc, local_sizes, full_sizes, global_downscaling=2,
):
    loc_r, loc_c = agent_loc
    local_w, local_h = local_sizes
    full_w, full_h = full_sizes
    
    if global_downscaling > 1:
        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
    else:
        gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

    return [gx1, gx2, gy1, gy2]

    
def init_map_and_pose(map_cfg, device, num_scenes=1, remove_batch_dim=False):
    # Initialize map variables:
    # Full map consists of multiple channels containing the following:
    # 1. Obstacle Map
    # 2. Exploread Area
    nc = 2
    visited_locs_in_full_map = [[] for _ in range(num_scenes)]

    # Calculating full and local map sizes
    map_size = map_cfg.map_size_cm // map_cfg.map_resolution
    full_w, full_h = map_size, map_size
    local_w = int(full_w / map_cfg.global_downscaling)
    local_h = int(full_h / map_cfg.global_downscaling)

    # Initializing full and local map
    full_map = torch.zeros(num_scenes, nc, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, nc, local_w, local_h).float().to(device)

    # Initial full and local pose: (x, y, o) x,y: meters, o: degree
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)

    # Origin of local map: (column, row, 0) meters (upper left point)
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries: (x1, x2, y1, y2) map unit
    lmb = np.zeros((num_scenes, 4)).astype(int)
        
    for e in range(num_scenes):
        full_map[e].fill_(0.)
        full_pose[e].fill_(0.)
        # an agent is initialized at the center of the map
        full_pose[e, :2] = map_cfg.map_size_cm / 100.0 / 2.0

        locs = full_pose[e].cpu().numpy()
    
        r, c = locs[0], locs[1]
        loc_r, loc_c = [int(r * 100.0 / map_cfg.map_resolution),
                        int(c * 100.0 / map_cfg.map_resolution)]

        # initialize explored area and current agent location
        visited_locs_in_full_map[e].append((loc_r, loc_c))

        lmb[e] = get_local_map_boundaries(
            (loc_r, loc_c), (local_w, local_h), (full_w, full_h),
            map_cfg.global_downscaling
        )

        origins[e] = [lmb[e][0] * map_cfg.map_resolution / 100.0,
                      lmb[e][2] * map_cfg.map_resolution / 100.0, 0.]

        local_map[e] = full_map[e, :,
                                lmb[e, 0]:lmb[e, 1],
                                lmb[e, 2]:lmb[e, 3]]
        local_pose[e] = full_pose[e] - \
            torch.from_numpy(origins[e]).to(device).float()
    
    if num_scenes == 1 and remove_batch_dim:
        return full_map[0], full_pose[0], local_map[0], local_pose[0], \
               lmb[0], origins[0], visited_locs_in_full_map[0]
    return full_map, full_pose, local_map, local_pose, lmb, origins, \
           visited_locs_in_full_map
           
  
def get_grid(pose, grid_size, device):
    """
    Input:
        `pose` FloatTensor(bs, 3)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w)
        `device` torch.device (cpu or gpu)
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)

    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.size(0)
    t = t * np.pi / 180.
    cos_t = t.cos()
    sin_t = t.sin()

    theta11 = torch.stack([cos_t, -sin_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta12 = torch.stack([sin_t, cos_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta1 = torch.stack([theta11, theta12], 1)

    theta21 = torch.stack([torch.ones(x.shape).to(device),
                           -torch.zeros(x.shape).to(device), x], 1)
    theta22 = torch.stack([torch.zeros(x.shape).to(device),
                           torch.ones(x.shape).to(device), y], 1)
    theta2 = torch.stack([theta21, theta22], 1)

    rot_grid = F.affine_grid(theta1, torch.Size(grid_size))
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size))

    return rot_grid, trans_grid


  
class NavMapping(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        
        self.num_processes = 1

        self.device = device
        self.screen_h = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT
        self.screen_w = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
        self.fov = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV
        
        self.resolution = config.MAPPING.map_resolution
        self.z_resolution = config.MAPPING.map_resolution
        self.map_size_cm = config.MAPPING.map_size_cm // config.MAPPING.global_downscaling
        self.vision_range = config.MAPPING.vision_range
        self.du_scale = config.MAPPING.du_scale
        self.exp_pred_threshold = config.MAPPING.exp_pred_threshold
        self.map_pred_threshold = config.MAPPING.map_pred_threshold

        self.max_height = int(360 / self.z_resolution)
        self.min_height = int(-40 / self.z_resolution)
        self.agent_height = config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT * 100.
        self.shift_loc = [self.vision_range * self.resolution // 2, 0, 0]
        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, self.fov
        )

        vr = self.vision_range

        self.init_grid = torch.zeros(
            self.num_processes, 1, vr, vr,
            self.max_height - self.min_height
        ).float().to(self.device)

        self.feat = torch.ones(
            self.num_processes, 1,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)

    def forward(self, obs, pose_obs, maps_last, poses_last, camera_elevation_degree=0):
        depth = torch.from_numpy(obs['processed_depth']).to(self.device).unsqueeze(0)
                
        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, self.device, scale=self.du_scale
        )

        # rotate along x-axis, change elevation
        agent_view_t = du.transform_camera_view_t(
            point_cloud_t, self.agent_height, camera_elevation_degree, self.device
        )

        # do not rotate along z-axis: shift_loc[-1]=0 (rotate the map later)
        agent_view_centered_t = du.transform_pose_t(
            agent_view_t, self.shift_loc, self.device
        )

        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
                               vision_range // 2.) / vision_range * 2.
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] -
                              (max_h + min_h) // 2.) / (max_h - min_h) * 2.
        
        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])

        voxels = du.splat_feat_nd(
            self.init_grid * 0., self.feat, XYZ_cm_std
        ).transpose(2, 3) # tranpose because x-axis should denote columns instead of rows
        voxels = torch.flip(voxels, (2, ))  # facing north at the front
        # (num_scenes, channel, hx, hy, hz)

        min_z = int(25 / z_resolution - min_h)
        max_z = int((self.agent_height + 1) / z_resolution - min_h)

        agent_height_proj = voxels[..., min_z:max_z].sum(4)
        all_height_proj = voxels.sum(4)

        fp_map_pred = agent_height_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        pose_pred = poses_last

        agent_view = torch.zeros(self.num_processes, 2,
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device)

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        # y1 = self.map_size_cm // (self.resolution * 2)
        y1 = self.map_size_cm // (self.resolution * 2) - self.vision_range # facing north at the front
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred

        corrected_pose = pose_obs

        def get_new_pose_batch(pose, rel_pose_change):

            pose[:, 1] += rel_pose_change[:, 0] * \
                torch.sin(pose[:, 2] / 57.29577951308232) \
                + rel_pose_change[:, 1] * \
                torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * \
                torch.cos(pose[:, 2] / 57.29577951308232) \
                - rel_pose_change[:, 1] * \
                torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose

        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2] * 100.0 / self.resolution
                            - self.map_size_cm // (self.resolution * 2)) /\
            (self.map_size_cm // (self.resolution * 2))
        # st_pose[:, 2] = 90. - (st_pose[:, 2])
        st_pose[:, :2] = torch.flip(st_pose[:, :2], (1, )) # transpose x and y axis
        st_pose[:, 2] = st_pose[:, 2] - 180

        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                      self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)

        map_pred, _ = torch.max(maps2, 1)

        return fp_map_pred, map_pred, pose_pred, current_poses
