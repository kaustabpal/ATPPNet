# @brief:    Logging and saving point clouds and range images
# @author    Aditya Sharma    [meduri99aditya@gmail.com]
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
import open3d as o3d
import numpy as np
from pathlib import Path
import yaml
import os
from atppnet.utils.utils import load_files, range_projection
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

def prepare_data_trainval(cfg):
    #Path(cfg["DATA_CONFIG"]["SAVE_PATH"]).mkdir(parents=True, exist_ok=True)
    # if not os.path.exists(cfg["DATA_CONFIG"]["SAVE_PATH"]):
    #     os.makedirs(cfg["DATA_CONFIG"]["SAVE_PATH"])
    version_train = cfg['DATA_CONFIG']['NS_VERSION'] + "-trainval"
    data_path_train = os.path.join(cfg['DATA_CONFIG']['DATASET_PATH'], version_train)
    nusc = NuScenes(version=version_train, dataroot=data_path_train, verbose=True)
    print(len(nusc.scene))
    train_scenes = splits.train
    val_scenes = splits.val

    max_range = []

    available_scenes = []
    ratio = 5 * 32 / 1024
    for scene in nusc.scene:
        # print(scene)
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, _, _ = nusc.get_sample_data(sd_rec['token'])
            # print(lidar_path)
            if not Path(lidar_path).exists():
                scene_not_exist = True
                # print("lol",scene)
                break
            else:
                break

        if scene_not_exist:
            continue
        available_scenes.append(scene)

    available_scene_names = [s['name'] for s in available_scenes]


    #   TRAINING SET GENERATION
    scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in scenes])

    token_list = []
    scene_token_list = []
    scene_token_index = []
    scene_name = 0
    cmap = mpl.colormaps["turbo_r"]
    norm = mpl.colors.Normalize(vmin=1, vmax=70, clip=False)
    for i in range(len(nusc.sample)):
        sample = nusc.sample[i]
        scene_token = sample['scene_token']

        if scene_token not in scenes:
            continue

        if scene_token not in scene_token_list:
            scene_token_list.append(scene_token)
            scene_token_index.append(i)
            # scene_name+=1
            
    for i in range(len(scene_token_index)):
        idx = 0
        # scene_name = scene_token_list[i]
        scene_index = scene_token_index[i]

        sample = nusc.sample[scene_index]

        seqstr = "{0:03d}".format(int(scene_name))
        scene_name+=1
        dst_folder = os.path.join(cfg["DATA_CONFIG"]["SAVE_PATH"], seqstr, "processed")
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        lidar_token = sample['data']['LIDAR_TOP']  # 360 lidar
        lidar_token = nusc.get('sample_data', lidar_token)
        print("Processing train = ", scene_name, idx, dst_folder)

        while True:
            lidar_path = os.path.join(data_path_train, lidar_token['filename'])
            raw_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))
            pointcloud = raw_data[:, :4]@np.array([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])

            proj_range, proj_vertex, proj_intensity, proj_idx = range_projection(
                    pointcloud,
                    fov_up=cfg["DATA_CONFIG"]["FOV_UP"],
                    fov_down=cfg["DATA_CONFIG"]["FOV_DOWN"],
                    proj_H=cfg["DATA_CONFIG"]["HEIGHT"],
                    proj_W=cfg["DATA_CONFIG"]["WIDTH"],
                    max_range=cfg["DATA_CONFIG"]["MAX_RANGE"],
            )
            mr = np.max(proj_range)
            max_range.append(mr)
            # print("max range =", np.max(max_range))
            dst_path_range = os.path.join(dst_folder, "range")
            if not os.path.exists(dst_path_range):
                os.makedirs(dst_path_range)
            file_path = os.path.join(dst_path_range, str(idx).zfill(3))
            np.save(file_path, proj_range)

            # Save xyz
            dst_path_xyz = os.path.join(dst_folder, "xyz")
            if not os.path.exists(dst_path_xyz):
                os.makedirs(dst_path_xyz)
            file_path = os.path.join(dst_path_xyz, str(idx).zfill(3))
            np.save(file_path, proj_vertex)

            # Save intensity
            dst_path_intensity = os.path.join(dst_folder, "intensity")
            if not os.path.exists(dst_path_intensity):
                os.makedirs(dst_path_intensity)
            file_path = os.path.join(dst_path_intensity, str(idx).zfill(3))
            np.save(file_path, proj_intensity)
            # print(len(proj_range[proj_range>70]))
            dst_path_range = os.path.join(cfg["DATA_CONFIG"]["SAVE_PATH"], seqstr)
            if not os.path.exists(dst_path_range):
                os.makedirs(dst_path_range)

            # props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            # fig, axs = plt.subplots(1, 1, sharex=True, figsize=(30, 30 * ratio))
            # axs.imshow(proj_range, cmap=cmap, norm=norm)
            # plt.savefig(os.path.join(dst_path_range, str(idx).zfill(3)+".jpg"), bbox_inches="tight", transparent=True) #, dpi=2000)
            # plt.close(fig)

            idx+=1

            if not lidar_token["next"]:
                break
            prev_token = lidar_token["sample_token"]
            lidar_token = nusc.get('sample_data', lidar_token["next"])

    #   VALIDATION SET GENERATION
    scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in scenes])

    token_list = []
    scene_token_list = []
    scene_token_index = []
    # scene_name = 0
    cmap = mpl.colormaps["turbo_r"]
    norm = mpl.colors.Normalize(vmin=1, vmax=70, clip=False)
    for i in range(len(nusc.sample)):
        sample = nusc.sample[i]
        scene_token = sample['scene_token']

        if scene_token not in scenes:
            continue

        if scene_token not in scene_token_list:
            scene_token_list.append(scene_token)
            scene_token_index.append(i)
            # scene_name+=1
            
    for i in range(len(scene_token_index)):
        idx = 0
        # scene_name = scene_token_list[i]
        scene_index = scene_token_index[i]

        sample = nusc.sample[scene_index]

        seqstr = "{0:03d}".format(int(scene_name))
        scene_name+=1
        dst_folder = os.path.join(cfg["DATA_CONFIG"]["SAVE_PATH"], seqstr, "processed")
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        lidar_token = sample['data']['LIDAR_TOP']  # 360 lidar
        lidar_token = nusc.get('sample_data', lidar_token)
        print("Processing val = ", scene_name, idx, dst_folder)

        while True:
            lidar_path = os.path.join(data_path_train, lidar_token['filename'])
            raw_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))
            pointcloud = raw_data[:, :4]@np.array([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])

            proj_range, proj_vertex, proj_intensity, proj_idx = range_projection(
                    pointcloud,
                    fov_up=cfg["DATA_CONFIG"]["FOV_UP"],
                    fov_down=cfg["DATA_CONFIG"]["FOV_DOWN"],
                    proj_H=cfg["DATA_CONFIG"]["HEIGHT"],
                    proj_W=cfg["DATA_CONFIG"]["WIDTH"],
                    max_range=cfg["DATA_CONFIG"]["MAX_RANGE"],
            )
            mr = np.max(proj_range)
            max_range.append(mr)
            # print("max range =", np.max(max_range))
            dst_path_range = os.path.join(dst_folder, "range")
            if not os.path.exists(dst_path_range):
                os.makedirs(dst_path_range)
            file_path = os.path.join(dst_path_range, str(idx).zfill(3))
            np.save(file_path, proj_range)

            # Save xyz
            dst_path_xyz = os.path.join(dst_folder, "xyz")
            if not os.path.exists(dst_path_xyz):
                os.makedirs(dst_path_xyz)
            file_path = os.path.join(dst_path_xyz, str(idx).zfill(3))
            np.save(file_path, proj_vertex)

            # Save intensity
            dst_path_intensity = os.path.join(dst_folder, "intensity")
            if not os.path.exists(dst_path_intensity):
                os.makedirs(dst_path_intensity)
            file_path = os.path.join(dst_path_intensity, str(idx).zfill(3))
            np.save(file_path, proj_intensity)
            # print(len(proj_range[proj_range>70]))
            dst_path_range = os.path.join(cfg["DATA_CONFIG"]["SAVE_PATH"], seqstr)
            if not os.path.exists(dst_path_range):
                os.makedirs(dst_path_range)

            # props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            # fig, axs = plt.subplots(1, 1, sharex=True, figsize=(30, 30 * ratio))
            # axs.imshow(proj_range, cmap=cmap, norm=norm)
            # plt.savefig(os.path.join(dst_path_range, str(idx).zfill(3)+".jpg"), bbox_inches="tight", transparent=True) #, dpi=2000)
            # plt.close(fig)

            idx+=1

            if not lidar_token["next"]:
                break
            prev_token = lidar_token["sample_token"]
            lidar_token = nusc.get('sample_data', lidar_token["next"])
            
    print(len(max_range))
    return scene_name



def prepare_data_test(cfg, scene_name):
    #Path(cfg["DATA_CONFIG"]["SAVE_PATH"]).mkdir(parents=True, exist_ok=True)
    # if not os.path.exists(cfg["DATA_CONFIG"]["SAVE_PATH"]):
    #     os.makedirs(cfg["DATA_CONFIG"]["SAVE_PATH"])
    version_train = cfg['DATA_CONFIG']['NS_VERSION'] + "-trainval"
    data_path_train = os.path.join(cfg['DATA_CONFIG']['DATASET_PATH'], version_train)
    nusc = NuScenes(version=version_train, dataroot=data_path_train, verbose=True)
    print(len(nusc.scene))
    train_scenes = splits.train
    val_scenes = splits.val

    max_range = []

    available_scenes = []
    ratio = 5 * 32 / 1024
    for scene in nusc.scene:
        # print(scene)
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, _, _ = nusc.get_sample_data(sd_rec['token'])
            # print(lidar_path)
            if not Path(lidar_path).exists():
                scene_not_exist = True
                # print("lol",scene)
                break
            else:
                break

        if scene_not_exist:
            continue
        available_scenes.append(scene)

    available_scene_names = [s['name'] for s in available_scenes]


    #   TRAINING SET GENERATION
    scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in scenes])

    token_list = []
    scene_token_list = []
    scene_token_index = []
    # scene_name = 0
    cmap = mpl.colormaps["turbo_r"]
    norm = mpl.colors.Normalize(vmin=1, vmax=70, clip=False)
    for i in range(len(nusc.sample)):
        sample = nusc.sample[i]
        scene_token = sample['scene_token']

        if scene_token not in scenes:
            continue

        if scene_token not in scene_token_list:
            scene_token_list.append(scene_token)
            scene_token_index.append(i)
            # scene_name+=1
            
    for i in range(len(scene_token_index)):
        idx = 0
        # scene_name = scene_token_list[i]
        scene_index = scene_token_index[i]

        sample = nusc.sample[scene_index]

        seqstr = "{0:03d}".format(int(scene_name))
        scene_name+=1
        dst_folder = os.path.join(cfg["DATA_CONFIG"]["SAVE_PATH"], seqstr, "processed")
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        lidar_token = sample['data']['LIDAR_TOP']  # 360 lidar
        lidar_token = nusc.get('sample_data', lidar_token)
        print("Processing train = ", scene_name, idx, dst_folder)

        while True:
            lidar_path = os.path.join(data_path_train, lidar_token['filename'])
            raw_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))
            pointcloud = raw_data[:, :4]@np.array([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])

            proj_range, proj_vertex, proj_intensity, proj_idx = range_projection(
                    pointcloud,
                    fov_up=cfg["DATA_CONFIG"]["FOV_UP"],
                    fov_down=cfg["DATA_CONFIG"]["FOV_DOWN"],
                    proj_H=cfg["DATA_CONFIG"]["HEIGHT"],
                    proj_W=cfg["DATA_CONFIG"]["WIDTH"],
                    max_range=cfg["DATA_CONFIG"]["MAX_RANGE"],
            )
            mr = np.max(proj_range)
            max_range.append(mr)
            # print("max range =", np.max(max_range))
            dst_path_range = os.path.join(dst_folder, "range")
            if not os.path.exists(dst_path_range):
                os.makedirs(dst_path_range)
            file_path = os.path.join(dst_path_range, str(idx).zfill(3))
            np.save(file_path, proj_range)

            # Save xyz
            dst_path_xyz = os.path.join(dst_folder, "xyz")
            if not os.path.exists(dst_path_xyz):
                os.makedirs(dst_path_xyz)
            file_path = os.path.join(dst_path_xyz, str(idx).zfill(3))
            np.save(file_path, proj_vertex)

            # Save intensity
            dst_path_intensity = os.path.join(dst_folder, "intensity")
            if not os.path.exists(dst_path_intensity):
                os.makedirs(dst_path_intensity)
            file_path = os.path.join(dst_path_intensity, str(idx).zfill(3))
            np.save(file_path, proj_intensity)
            # print(len(proj_range[proj_range>70]))
            dst_path_range = os.path.join(cfg["DATA_CONFIG"]["SAVE_PATH"], seqstr)
            if not os.path.exists(dst_path_range):
                os.makedirs(dst_path_range)

            # props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            # fig, axs = plt.subplots(1, 1, sharex=True, figsize=(30, 30 * ratio))
            # axs.imshow(proj_range, cmap=cmap, norm=norm)
            # plt.savefig(os.path.join(dst_path_range, str(idx).zfill(3)+".jpg"), bbox_inches="tight", transparent=True) #, dpi=2000)
            # plt.close(fig)

            idx+=1

            if not lidar_token["next"]:
                break
            prev_token = lidar_token["sample_token"]
            lidar_token = nusc.get('sample_data', lidar_token["next"])

    #   VALIDATION SET GENERATION
    scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in scenes])

    token_list = []
    scene_token_list = []
    # scene_token_index = []
    # scene_name = 0
    cmap = mpl.colormaps["turbo_r"]
    norm = mpl.colors.Normalize(vmin=1, vmax=70, clip=False)
    for i in range(len(nusc.sample)):
        sample = nusc.sample[i]
        scene_token = sample['scene_token']

        if scene_token not in scenes:
            continue

        if scene_token not in scene_token_list:
            scene_token_list.append(scene_token)
            scene_token_index.append(i)
            # scene_name+=1
            
    for i in range(len(scene_token_index)):
        idx = 0
        # scene_name = scene_token_list[i]
        scene_index = scene_token_index[i]

        sample = nusc.sample[scene_index]

        seqstr = "{0:03d}".format(int(scene_name))
        scene_name+=1
        dst_folder = os.path.join(cfg["DATA_CONFIG"]["SAVE_PATH"], seqstr, "processed")
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        lidar_token = sample['data']['LIDAR_TOP']  # 360 lidar
        lidar_token = nusc.get('sample_data', lidar_token)
        print("Processing val = ", scene_name, idx, dst_folder)

        while True:
            lidar_path = os.path.join(data_path_train, lidar_token['filename'])
            raw_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))
            pointcloud = raw_data[:, :4]@np.array([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])

            proj_range, proj_vertex, proj_intensity, proj_idx = range_projection(
                    pointcloud,
                    fov_up=cfg["DATA_CONFIG"]["FOV_UP"],
                    fov_down=cfg["DATA_CONFIG"]["FOV_DOWN"],
                    proj_H=cfg["DATA_CONFIG"]["HEIGHT"],
                    proj_W=cfg["DATA_CONFIG"]["WIDTH"],
                    max_range=cfg["DATA_CONFIG"]["MAX_RANGE"],
            )
            mr = np.max(proj_range)
            max_range.append(mr)
            # print("max range =", np.max(max_range))
            dst_path_range = os.path.join(dst_folder, "range")
            if not os.path.exists(dst_path_range):
                os.makedirs(dst_path_range)
            file_path = os.path.join(dst_path_range, str(idx).zfill(3))
            np.save(file_path, proj_range)

            # Save xyz
            dst_path_xyz = os.path.join(dst_folder, "xyz")
            if not os.path.exists(dst_path_xyz):
                os.makedirs(dst_path_xyz)
            file_path = os.path.join(dst_path_xyz, str(idx).zfill(3))
            np.save(file_path, proj_vertex)

            # Save intensity
            dst_path_intensity = os.path.join(dst_folder, "intensity")
            if not os.path.exists(dst_path_intensity):
                os.makedirs(dst_path_intensity)
            file_path = os.path.join(dst_path_intensity, str(idx).zfill(3))
            np.save(file_path, proj_intensity)
            # print(len(proj_range[proj_range>70]))
            dst_path_range = os.path.join(cfg["DATA_CONFIG"]["SAVE_PATH"], seqstr)
            if not os.path.exists(dst_path_range):
                os.makedirs(dst_path_range)

            # props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            # fig, axs = plt.subplots(1, 1, sharex=True, figsize=(30, 30 * ratio))
            # axs.imshow(proj_range, cmap=cmap, norm=norm)
            # plt.savefig(os.path.join(dst_path_range, str(idx).zfill(3)+".jpg"), bbox_inches="tight", transparent=True) #, dpi=2000)
            # plt.close(fig)

            idx+=1

            if not lidar_token["next"] or scene_name==170:
                break
            prev_token = lidar_token["sample_token"]
            lidar_token = nusc.get('sample_data', lidar_token["next"])
    print(len(max_range))

    

if __name__ == "__main__":
    config_filename = "./config/nuscenes_parameters.yml"
    cfg = yaml.safe_load(open(config_filename))
    if not os.path.exists(cfg["DATA_CONFIG"]["SAVE_PATH"]):
        os.makedirs(cfg["DATA_CONFIG"]["SAVE_PATH"])
    scene_name = prepare_data_trainval(cfg)
    prepare_data_test(cfg, scene_name)

