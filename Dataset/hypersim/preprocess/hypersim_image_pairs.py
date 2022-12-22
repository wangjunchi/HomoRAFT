import cv2
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import h5py
import torch
import pandas as pd
import itertools
import random


class HypersimPairDataset(Dataset):
    def __init__(self, data_dir, scene_name):
        self.focal_length = 886.81
        self.data_dir = data_dir
        # self.scene_name = scene_name
        # self.split = split
        p_path = os.path.join(data_dir, "{}/p.csv".format(scene_name))
        q_path = os.path.join(data_dir, "{}/q.csv".format(scene_name))
        self.p_list = pd.read_csv(p_path)
        self.q_list = pd.read_csv(q_path)

        self.image_path = "{}/images/scene_{}_final_preview/frame.{}.tonemap.jpg"
        self.planes_path = "{}/images/scene_{}_geometry_hdf5/frame.{}.planes.hdf5"
        self.normal_path = "{}/images/scene_{}_geometry_hdf5/frame.{}.normal_cam.hdf5"
        self.depth_path = "{}/images/scene_{}_geometry_hdf5/frame.{}.depth_meters.hdf5"
        self.camera_position_path = "{}/_detail/{}/camera_keyframe_positions.hdf5"
        self.camera_orientation_path = "{}/_detail/{}/camera_keyframe_orientations.hdf5"

    def __len__(self):
        return len(self.p_list)

    def __getitem__(self, idx):
        # row1, row2 = self.pair_lists[idx]
        row1 = self.p_list.iloc[idx]
        row2 = self.q_list.iloc[idx]
        # read image 1
        sample1 = self.get_sample(row1)
        sample2 = self.get_sample(row2)

        gt_match = self.generate_gt_matching(sample1, sample2)

        plane_instance_parameters_1 = self.extract_camera_plane_instances(sample1)
        plane_instance_parameters_2 = self.extract_camera_plane_instances(sample2)

        sample1["image"] = torch.from_numpy(sample1["image"]/255.0).permute(2, 0, 1)
        sample2["image"] = torch.from_numpy(sample2["image"] / 255.0).permute(2, 0, 1)

        sample1['instance_parameters'] = plane_instance_parameters_1
        sample2['instance_parameters'] = plane_instance_parameters_2

        return {"sample_1": sample1, "sample_2": sample2, "gt_match": gt_match}
        # image_name = self.image_names[idx]
        # image_path = os.path.join(self.data_dir, image_name)
        # image = Image.open(image_path)
        # if self.transform:
        #     image = self.transform(image)
        # return image, image_name

    @staticmethod
    def convert_distance_to_depth_map(depth, focal_length):
        """This function transformed the distance to the center to a more commonly known depth map.

        Function from: https://github.com/apple/ml-hypersim/issues/9#issuecomment-754935697

        Parameters:
            depth (np.ndarray): Depth to be transformed
            focal_length (float): Focal length of the scene.
        """
        height, width = np.array(depth).shape
        npy_image_plane_x = np.linspace((-0.5 * width) + 0.5, (0.5 * width) - 0.5, width) \
                                .reshape(1, width).repeat(height, 0).astype(np.float32)[:, :, None]
        npy_image_plane_y = np.linspace((-0.5 * height) + 0.5, (0.5 * height) - 0.5, height) \
                                .reshape(height, 1).repeat(width, 1).astype(np.float32)[:, :, None]
        npy_image_plane_z = np.full([height, width, 1], focal_length, np.float32)
        npy_image_plane = np.concatenate([npy_image_plane_x, npy_image_plane_y, npy_image_plane_z], 2)

        return depth / np.linalg.norm(npy_image_plane, 2, 2) * focal_length

    def get_sample(self, row):
        scene_name = row["scene_name"]
        camera_name = row["camera_name"]
        frame_id = str(row["frame_id"]).zfill(4)

        image = self.load_image(scene_name, camera_name, frame_id)
        planes = self.load_plane(scene_name, camera_name, frame_id)
        normal = self.load_normal(scene_name, camera_name, frame_id)
        depth = self.load_depth(scene_name, camera_name, frame_id)

        pose = self.load_camera_pose(scene_name, camera_name, int(frame_id))
        return {"image": image, "planes": planes, "normal": normal, "depth":depth, "camera_pose": pose, "metadata": row.values.tolist()[1:]}

    def get_pairs(self, sample_list):
        all_pairs = list(itertools.combinations(sample_list.iterrows(), 2))
        # random.shuffle(all_pairs)
        return all_pairs

    def get_pair_list(self):
        image_information = pd.read_csv(os.path.join(self.data_dir, "hypersim_split.csv"))
        all_files = image_information[image_information["split_partition_name"] == 'train' | image_information["split_partition_name"] == 'test']
        # check which folders are downloaded
        directories = []
        for root, dirs, files in os.walk(self.data_dir):
            print("Found dir: ", dirs)
            directories = dirs
            break

        sample_list = all_files[all_files["scene_name"].isin(directories)].copy().reset_index()
        sample_list.sort_values(by=["scene_name", "camera_name", "frame_id"], inplace=True)
        scene_names = sample_list["scene_name"].unique()
        all_pairs = []
        for scene_name in scene_names:
            scene_sample_list = sample_list[sample_list["scene_name"] == scene_name]
            all_pairs += self.get_pairs(scene_sample_list)

        # random.shuffle(all_pairs)
        return all_pairs

    def load_image(self, scene_name, camera_id, frame_id):
        img_path = os.path.join(self.data_dir, self.image_path.format(scene_name, camera_id, frame_id))
        assert os.path.exists(img_path), "Image file does not exist: {}".format(img_path)
        image_data = cv2.imread(img_path).astype('float32')
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        return image_data

    def load_plane(self, scene_name, camera_id, frame_id):
        """
        Load the plane information from the hdf5 file
        Plane must be generated in advance
        """
        planes_path = os.path.join(self.data_dir, self.planes_path.format(scene_name, camera_id, frame_id))
        assert os.path.exists(planes_path), "Plane file does not exist: {}".format(planes_path)
        planes_data = h5py.File(planes_path, 'r')
        planes = np.array(planes_data["dataset"]).astype('float32')
        planes_data.close()
        return planes

    def load_depth(self, scene_name, camera_id, frame_id):
        depth_path = os.path.join(self.data_dir, self.depth_path.format(scene_name, camera_id, frame_id))
        assert os.path.exists(depth_path), "Depth file does not exist: {}".format(depth_path)
        depth_data = h5py.File(depth_path, 'r')
        depth = np.array(depth_data["dataset"]).astype('float32')
        depth_data.close()
        depth = self.convert_distance_to_depth_map(depth, self.focal_length)
        return depth

    def load_normal(self, scene_name, camera_id, frame_id):
        normal_path = os.path.join(self.data_dir, self.normal_path.format(scene_name, camera_id, frame_id))
        assert os.path.exists(normal_path), "Normal file does not exist: {}".format(normal_path)
        normal_data = h5py.File(normal_path, 'r')
        normal = np.array(normal_data["dataset"]).astype('float32')
        normal_data.close()
        return normal

    def load_camera_pose(self, scene_name, camera_name, frame_id):
        camera_position_path = os.path.join(self.data_dir, self.camera_position_path.format(scene_name, camera_name))
        camera_position_data = h5py.File(camera_position_path, 'r')
        camera_position = np.array(camera_position_data["dataset"]).astype('float32')
        camera_position_data.close()

        camera_orientation_path = os.path.join(self.data_dir, self.camera_orientation_path.format(scene_name, camera_name))
        camera_orientation_data = h5py.File(camera_orientation_path, 'r')
        camera_orientation = np.array(camera_orientation_data["dataset"]).astype('float32')
        camera_orientation_data.close()

        r = camera_orientation[frame_id]
        t = camera_position[frame_id]
        # t[1] =2.0
        unit = 0.0254
        t = t * unit
        h = np.concatenate([r, t.reshape(3, 1)], axis=1)
        h = np.concatenate([h, np.array([[0, 0, 0, 1.]])], axis=0)

        return h


    def extract_camera_plane_instances(self, sample):
        planes = sample["planes"]
        normal = sample["normal"]
        plane_d = planes[:, :, 0]
        gt_segmentation = planes[:, :, 1]
        camera_pose = sample["camera_pose"]
        # generate plane parameters for each plane
        h, w = gt_segmentation.shape
        plane_parameters = np.zeros((3, h, w))
        valid_region = plane_d > 0.1
        # temp = (normal[valid_region, :] / (plane_d[valid_region].reshape(-1, 1)))
        temp = normal[valid_region, :]
        plane_parameters[:, valid_region] = np.transpose(temp, (1, 0))

        num_planes = int(np.max(gt_segmentation))
        plane_instance_parameters = np.zeros((num_planes, 4))
        for i in range(num_planes):
            plane_instance_parameters[i, :3] = np.median(plane_parameters[:, gt_segmentation == i + 1], axis=1)
            plane_instance_parameters[i, 3] = np.median(plane_d[gt_segmentation == i + 1])

        # # transform the plane parameters to the world coordinate
        # plane_instance_parameters = np.matmul(plane_instance_parameters, np.linalg.inv(camera_pose))
        # normalize the plane parameters
        # plane_instance_parameters = plane_instance_parameters / (plane_instance_parameters[:, 3].reshape(-1, 1) + 1e-6)
        # plane_instance_parameters[:, :4] = plane_instance_parameters[:, :4] / (np.linalg.norm(
        #     plane_instance_parameters[:, :3], axis=1).reshape(-1, 1) + 1e-6)
        return plane_instance_parameters

    def extract_plane_instances(self, sample):
        """
        Extract plane instances from the plane and normal information
        """
        planes = sample["planes"]
        normal = sample["normal"]
        plane_d = planes[:, :, 0]
        gt_segmentation = planes[:, :, 1]
        camera_pose = sample["camera_pose"]
        # generate plane parameters for each plane
        h, w = gt_segmentation.shape
        plane_parameters = np.zeros((3, h, w))
        valid_region = plane_d > 0.1
        temp = (normal[valid_region, :] / (plane_d[valid_region].reshape(-1, 1)) + 1e-6)
        plane_parameters[:, valid_region] = np.transpose(temp, (1, 0))

        num_planes = int(np.max(gt_segmentation))
        plane_instance_parameters = np.zeros((num_planes, 4))
        for i in range(num_planes):
            plane_instance_parameters[i, :3] = np.median(plane_parameters[:, gt_segmentation == i + 1], axis=1)
            plane_instance_parameters[i, 3] = 1.0

        # # transform the plane parameters to the world coordinate
        plane_instance_parameters = np.matmul(plane_instance_parameters, np.linalg.inv(camera_pose))
        # normalize the plane parameters
        # plane_instance_parameters = plane_instance_parameters / (plane_instance_parameters[:, 3].reshape(-1, 1) + 1e-6)
        plane_instance_parameters[:, :4] = plane_instance_parameters[:, :4] / (np.linalg.norm(
                                                        plane_instance_parameters[:, :3], axis=1).reshape(-1, 1)+1e-6)
        return plane_instance_parameters



    def generate_gt_matching(self, sample_1, sample_2):
        """
        Generate the ground truth matching
        """
        plane_instance_parameters_1 = self.extract_plane_instances(sample_1)
        plane_instance_parameters_2 = self.extract_plane_instances(sample_2)

        num_planes_1 = plane_instance_parameters_1.shape[0]
        num_planes_2 = plane_instance_parameters_2.shape[0]
        plane_distance = np.zeros((num_planes_1, num_planes_2))
        for i in range(num_planes_1):
            for j in range(num_planes_2):
                # plane_matches[i, j] = np.dot(parameters_1[i, :], parameters_2[j, :]) / (norm(parameters_1[i, :])*norm(parameters_2[j, :]))
                plane_distance[i, j] = np.linalg.norm(plane_instance_parameters_1[i, :4] - plane_instance_parameters_2[j, :4])

        row_ind, col_ind = linear_sum_assignment(plane_distance)

        # store all the matching pair
        matchings = torch.ones((5, 1)) * -1
        for (i, j) in zip(row_ind, col_ind):
            if plane_distance[i, j] < 0.01 and i<5 and j<5:
                matchings[i] = j
                # matchings.append([i, j])

        return matchings

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    dataset = HypersimPairDataset(data_dir="/home/junchi/sp1/dataset/hypersim", split="train")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for i in range(100):
        for step, batch in tqdm(enumerate(loader)):
            data = batch
            gt_match = data['gt_match']
            pass
