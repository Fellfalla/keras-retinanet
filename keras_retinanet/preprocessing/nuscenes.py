"""
Copyright 2017-2018 lvaleriu (https://github.com/lvaleriu/)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Standard Libraries
import csv
import os.path

# 3rd Party Libraries
import numpy as np
from PIL import Image

# Local Libraries
from .generator import Generator
from nuscenes_utils.nuscenes import NuScenes
from nuscenes_utils.data_classes import PointCloud, Box
from nuscenes_utils.geometry_utils import box_in_image, view_points, BoxVisibility


class NuscenesGenerator(Generator):
    """ Generate data for a nuScenes dataset.

    See www.nuscenes.org for more information.
    """
    DATATYPE = np.float32

    def __init__(
        self,
        nusc,
        scene_indices=None,
        channels=[1,2,3],
        **kwargs
    ):
        """ Initialize a KITTI data generator.

        Args
            nusc: Object pointing at a nuscenes database
            scene_indices: [int] Which scenes to take samples from
        """

        self.nusc = nusc
        if scene_indices is None:
            # We are using all scenes
            scene_indices = range(len(self.nusc.scene))

        """
        1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                             'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                             'Misc' or 'DontCare'
        1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                         truncated refers to the object leaving image boundaries
        1    occluded     Integer (0,1,2,3) indicating occlusion state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
        1    alpha        Observation angle of object, ranging [-pi..pi]
        4    bbox         2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
        3    dimensions   3D object dimensions: height, width, length (in meters)
        3    location     3D object location x,y,z in camera coordinates (in meters)
        1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
        """

        # Parameters
        self.dropout_chance = 0.0
        self.radar_sensors = ['RADAR_FRONT']
        self.camera_sensors = ['CAM_FRONT']
        self.labels = {}
        self.image_data = dict()
        self.classes = self._get_class_label_mapping()
        self.channels = channels
        self.normalize = False

        # Optional imports
        if self._image_plus_enabled():
            # Installing vizdom is required
            from raw_data_fusion.fusion_projection_lines import imageplus_creation
            self.image_plus_creation = imageplus_creation

        # Fill Labels
        for name, label in self.classes.items():
            self.labels[label] = name

        # Create all sample tokens
        self.sample_tokens = {}
        image_index = 0
        for scene_index in scene_indices:
            # iterate the samples in scene_rec
            sample_token = self.nusc.scene[scene_index]['first_sample_token']
            while sample_token is not '':
                self.sample_tokens[image_index] = sample_token
                image_index += 1

                # get the next token
                sample = self.nusc.get('sample', sample_token)
                sample_token = sample['next']


        # Create all annotations and put into image_data
        for image_index, sample_token in self.sample_tokens.items():
            self.image_data[image_index] = None #self.load_labels(sample_token, self.camera_sensors)

        # Finalize
        super(NuscenesGenerator, self).__init__(**kwargs)


    def _get_class_label_mapping(self):
        """
        :param nusc: [Nuscenes] The nuscenes dataset object
        :returns: [dict of (str, int)] mapping from category name to the corresponding index-number
        """
        category_indices = {c['name']: i for i, c in enumerate(self.nusc.category)}
        category_indices['bg'] = len(category_indices)
        return category_indices

    def _image_plus_enabled(self):
        r = 1 in self.channels
        g = 2 in self.channels
        b = 3 in self.channels
        return len(self.channels) > r+g+b

    def size(self):
        """ Size of the dataset.
        """
        return len(self.sample_tokens)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return len(self.classes)

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        All images of nuscenes dataset have the same aspect ratio which is 16/9
        """
        # All images of nuscenes dataset have the same aspect ratio
        return 16/9 
        # sample_token = self.sample_tokens[image_index]
        # sample = self.nusc.get('sample', sample_token)

        # image_sample = self.load_sample_data(sample, camera_name)
        # return float(image_sample.shape[1]) / float(image_sample.shape[0])

    def load_image(self, image_index):
        """
        Returns the image plus from given image and radar samples.
        It takes the requested channels into account.

        :param sample_token: [str] the token pointing to a certain sample
        :returns: [tuple] imageplus, img
        """
        # Initialize local variables
        radar_name = self.radar_sensors[0]
        camera_name = self.camera_sensors[0]

        # Gettign data from nuscenes database
        sample_token = self.sample_tokens[image_index]
        sample = self.nusc.get('sample', sample_token)

        # Grab the front camera and the radar sensor.
        radar_token = sample['data'][radar_name]
        camera_token = sample['data'][camera_name]
        image_target_shape = (self.image_min_side, self.image_max_side)

        if self._image_plus_enabled():
            # Load samples from disk
            radar_sample = self.load_sample_data(sample, radar_name)
            image_sample = self.load_sample_data(sample, camera_name)
            
            # Parameters
            kwargs = {
            'pointsensor_token': radar_token,
            'camera_token': camera_token,
            'height': (0, 3), 
            'image_target_shape': image_target_shape,
            'clear_radar': False,
            'clear_image': False,
            'dropout_chance': self.dropout_chance
            }
    
            # Create image plus
            img_p_full = self.image_plus_creation(self.nusc, image_data=image_sample, radar_data=radar_sample, **kwargs)
            
            # reduce to requested channels
            img_p_reduced = np.asarray([img_p_full[:,:,index-1] for index in self.channels], dtype=np.float32)
            input_data = np.transpose(img_p_reduced, axes=(1,2,0))

        else: # We are not in image_plus mode
            # Load rgb-image from disk
            input_data = self.load_sample_data(sample, camera_name, image_target_shape=image_target_shape)

        return input_data


    def load_sample_data(self, sample, sensor_channel, image_target_shape=None, **kwargs):
        """
        This function takes the token of a sample and a sensor sensor_channel and returns the according data

        Radar format:
            - Shape: 18 x n
            - Semantics: x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0

        Image format:
            - Shape: h x w x 3
            - Channels: RGB
            - KWargs:
                - [int] image_target_shape to limit image size
                - [tuple[int]] image_target_shape to limit image size #TODO figure out directions
        """

        # Get filepath
        sd_rec = self.nusc.get('sample_data', sample['data'][sensor_channel])
        file_name = os.path.join(self.nusc.dataroot, sd_rec['filename'])

        # Check conditions
        assert os.path.exists(file_name), "nuscenes data must be located in %s" %file_name

        # Read the data
        if "RADAR" in sensor_channel:
            points = PointCloud.load_pcd_bin(file_name)  # Load radar points
            data = points.astype(self.DATATYPE)
        elif "CAM" in sensor_channel:
            i = Image.open(file_name)

            # resize if size is given
            if image_target_shape is not None:
                try:
                    _ = iter(image_target_shape)
                except TypeError: 
                    # not iterable size
                    # limit both dimension to size, but keep aspect ration
                    size = (image_target_shape, image_target_shape)
                    i.thumbnail(size=size)
                else:
                    # iterable size
                    size = image_target_shape[::-1]  # revert dimensions
                    i = i.resize(size=size)

            data = np.array(i, dtype=self.DATATYPE) / 255
        else:
            raise Exception("\"%s\" is not supported" % sensor_channel)

        return data

    def load_labels(self, sample_token, sensor_channels):
        """
        Create 2D bounding box labels from the given sample token.

        1 bounding box vector contains:


        :param sample_token: the sample_token to get the annotation for
        :param sensor_channels: list of channels for cropping the labels, e.g. ['CAM_FRONT', 'RADAR_FRONT']
            This works only for CAMERA atm

        :returns: [2-D list nx5] Labels (used for training) with following entries:
            [0]: box x_min (normalized to the image size)
            [1]: box y_min (normalized to the image size)
            [2]: box x_max (normalized to the image size)
            [3]: box y_max (normalized to the image size)
            [4]: class_label
        """

        if any([s for s in sensor_channels if 'RADAR' in s]):
            print("[WARNING] Cropping to RADAR is not supported atm")
            sensor_channels = [c for c in sensor_channels if 'CAM' in sensor_channels]

        sample = self.nusc.get('sample', sample_token)
        labels = [] # initialize counter for each category

        # Camera parameters
        for selected_sensor_channel in sensor_channels:
            sd_rec = self.nusc.get('sample_data', sample['data'][selected_sensor_channel])

            # Create Boxes:
            _, boxes, camera_intrinsic = self.nusc.get_sample_data(sd_rec['token'], box_vis_level=BoxVisibility.ANY)
            imsize = (sd_rec['height'], sd_rec['width'])
            
            category_indices = self._get_class_label_mapping()
            # Create labels for all boxes that are visible
            for box in boxes:

                 # Add labels to boxes 
                box.label = category_indices[box.name]

                # Check if box is visible and transform box to 1D vector
                if box_in_image(box=box, intrinsic=camera_intrinsic, imsize=imsize, vis_level=BoxVisibility.ANY):
                    # If visible, we create the corresponding label
                    # normalize=True, because we usually resize the image anyways
                    box2d = box.box2d(camera_intrinsic, imsize=imsize, normalize=True)
                    labels.append([*box2d, box.label])

        return labels

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        image_data = self.image_data[image_index]

        if image_data is None:
            sample_token = self.sample_tokens[image_index]
            image_data = self.load_labels(sample_token, self.camera_sensors)
            self.image_data[image_index] = image_data

        annotations = {'labels': np.empty((len(image_data),)), 'bboxes': np.empty((len(image_data), 4))}

        for idx, ann in enumerate(image_data):
            annotations['bboxes'][idx, 0] = ann[0] # x1
            annotations['bboxes'][idx, 1] = ann[1] # y1
            annotations['bboxes'][idx, 2] = ann[2] # x2
            annotations['bboxes'][idx, 3] = ann[3] # y2
            annotations['labels'][idx] = ann[-1] # number code for class

        if not self.normalize:
            for i, bbox in enumerate(annotations['bboxes']):
                bbox[0] *= self.image_max_side # x1
                bbox[1] *= self.image_min_side # y1
                bbox[2] *= self.image_max_side # x2
                bbox[3] *= self.image_min_side # y2

        return annotations
