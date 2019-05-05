import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenesExplorer
import cv2
import numpy as np
from typing import Tuple
from PIL import Image

# the area which will be used:y_min,y_max,x_min,x_max
VALID_AREA = [0, 900, 350, 1250]

class ImageProcessing:
    """ Class for processing image. """

    def __init__(self, sensor_token: str, nusc, mini_size: float):
        """
        Initialize a point cloud and check it has the correct dimensions.
        :param img_path: Path of image
        """
        self.nusc = nusc
        (img_path, self.boxes, self.camera_intrinsic) = self.nusc.get_sample_data(sensor_token)
        self.__sensor_token = sensor_token
        self.img = Image.open(img_path)
        self.height = self.img.height
        self.width = self.img.width
        self.pic_size = [self.height, self.width]

        # Resize shapes
        self.res_ratio = 1.0
        self.img = self.__img_resize2(mini_size)

        self.image = cv2.imread(img_path)
        self.height1, self.width1 = self.image.shape[:2]
        self.image = self.__img_resize1(mini_size)

        self.labels = self.classes_decode()
        # Get corcons list
        self.control_points_list = self.__control_points_list()
        self.width_seg = 9
        self.height_seg = 9
        # self.labels, self.control_points_list = self.annotations_in_net()

    # Resize image
    def __img_resize2(self, min_size: float = 416.0):
        """resize the picture
        Args:
            img: the origin image
            min_size: the minimum value of image after resizing
        Return:
            img_res: image after resizing
        """
        # resize
        max_loc = self.pic_size.index(max(self.pic_size))
        min_loc = self.pic_size.index(min(self.pic_size))
        self.res_ratio = min_size/self.pic_size[min_loc]
        self.pic_size[max_loc] = self.res_ratio * self.pic_size[max_loc]
        self.pic_size[min_loc] = min_size
        self.height = self.pic_size[0]
        self.width = self.pic_size[1]
        self.img = self.img.resize((int(self.pic_size[1]), int(self.pic_size[0])), Image.ANTIALIAS)

        return self.img


    # Resize image
    def __img_resize1(self, min_size: float = 416.0):
        """resize the picture
        Args:
            img: the origin image
            min_size: the minimum value of image after resizing
        Return:
            img_res: image after resizing
        """

        # cut left and right
        self.image = self.image[VALID_AREA[0]:VALID_AREA[1], VALID_AREA[2]:VALID_AREA[3]]

        self.res_ratio = min_size/self.height1
        self.image = cv2.resize(self.image, (min_size, min_size), interpolation=4)
        self.height1, self.width1 = self.image.shape[:2]

        return self.image


    def render_arbitrary(self, sensor_type: str='CAM_FRONT') -> None:
        """
        Renders box for arbitrary sizes of images.
        :param
        """
        # Init axes.
        _, ax = plt.subplots(1, 1, figsize=(9, 16))

        # Show image.
        ax.imshow(self.img)

        # Show boxes.
        for box in self.boxes:
            c = np.array(NuScenesExplorer.get_color(box.name)) / 255.0
            box.render(ax, view=self.camera_intrinsic,
                       normalize=True, colors=(c, c, c),
                       wlh_factor=self.res_ratio)

        # Limit visible range.
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)
        ax.axis('off')
        ax.set_title(sensor_type)
        ax.set_aspect('equal')
        plt.show()

    def render_arbitrary1(self, sensor_type: str='CAM_FRONT') -> None:
        """
        Renders box for arbitrary sizes of images.
        :param
        """
        # Init axes.
        _, ax = plt.subplots(1, 1, figsize=(9, 16))

        # Show image.
        ax.imshow(self.image)

        # Show boxes.
        for box in self.boxes:
            c = np.array(NuScenesExplorer.get_color(box.name)) / 255.0
            box.render(ax, view=self.camera_intrinsic,
                       normalize=True, colors=(c, c, c),
                       wlh_factor=self.res_ratio)

        # Limit visible range.
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)
        ax.axis('off')
        ax.set_title(sensor_type)
        ax.set_aspect('equal')
        plt.show()

    def __control_points_list(self):
        # Show boxes.
        control_points_list = []
        for box in self.boxes:
            control_points_list.append(box.control_points_loc(view=self.camera_intrinsic, 
                                                              wlh_factor=self.res_ratio))
        
        return control_points_list

    def render_rectangle(self, control_points_list,
                        colors: Tuple = ('b', 'r', 'k'),
                        linewidth: float = 2,
                        sensor_type: str='CAM_FRONT'):
        """
        Render rectangle by control_points_list.
        param: control_points_list
        """
        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                ax.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
                prev = corner

        # Init axes.
        _, ax = plt.subplots(1, 1, figsize=(9,16))
        # Show image.
        ax.imshow(self.img)
 
        for control_point in control_points_list:
            # Draw the sides
            for i in range(4):
                c = np.array(NuScenesExplorer.get_color('vehicle.car')) / 255.0
                colors = (c,c,c)
                ax.plot([control_point[i][0], control_point[i + 4][0]],
                          [control_point[i][1], control_point[i + 4][1]],
                          color=colors[2], linewidth=linewidth)

            # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
            draw_rect(control_point[:4], colors[0])
            draw_rect(control_point[4:8], colors[1])

        # Limit visible range.
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)
        ax.axis('off')
        ax.set_title(sensor_type)
        ax.set_aspect('equal')
        plt.show()


    def classes_decode(self):
        """
        Decode class name into number in a image
        :return: decode all object in a image
        """
        class_number = []
        for box in self.boxes:
            class_number.append(self.class_type(box.name))
        return class_number


    def class_type(self, box_name):
        """
        Decode class name into number
        :param box_name: the labels for object classes
        :return: class_number: the number for object classes
        """
        # switch = {
        #     "animal": 1,
        #     "human.pedestrian.adult": 2,
        #     "human.pedestrian.child": 3,
        #     "human.pedestrian.construction_worker": 4,
        #     "human.pedestrian.personal_mobility": 5,
        #     "human.pedestrian.police_officer": 6,
        #     "human.pedestrian.stroller": 7,
        #     "human.pedestrian.wheelchair": 8,
        #     "movable_object.barrier": 9,
        #     "movable_object.debris": 10,
        #     "movable_object.pushable_pullable": 11,
        #     "movable_object.trafficcone": 12,
        #     "vehicle.bicycle": 13,
        #     "vehicle.bus.bendy": 14,
        #     "vehicle.bus.rigid": 15,
        #     "vehicle.car": 16,
        #     "vehicle.construction": 17,
        #     "vehicle.emergency.ambulance": 18,
        #     "vehicle.emergency.police": 19,
        #     "vehicle.motorcycle": 20,
        #     "vehicle.trailer": 21,
        #     "vehicle.truck": 22,
        #     "static_object.bicycle_rack *": 23
        # }

        switch = {
            "animal": 1,
            "human.pedestrian.adult": 2,
            "human.pedestrian.child": 3,
            "human.pedestrian.construction_worker": 4,
            "human.pedestrian.personal_mobility": 5,
            "human.pedestrian.police_officer": 6,
            "human.pedestrian.stroller": 7,
            "human.pedestrian.wheelchair": 8,
            "movable_object.barrier": 9,
            "movable_object.debris": 10,
            "movable_object.pushable_pullable": 11,
            "movable_object.trafficcone": 12,
            "vehicle.bicycle": 13,
            "vehicle.bus.bendy": 14,
            "vehicle.bus.rigid": 15,
            "vehicle.car": 16,
            "vehicle.construction": 17,
            "vehicle.emergency.ambulance": 18,
            "vehicle.emergency.police": 19,
            "vehicle.motorcycle": 20,
            "vehicle.trailer": 21,
            "vehicle.truck": 22,
            "static_object.bicycle_rack *": 23
        }

        class_number = switch[box_name]
        return class_number


    def annotations_in_net(self):
        """
        get the annotation for net work
        :return:cell_class:<class 'tuple'>: (9, 9):
                            whether there is an object status of each cell
                corner_point_deta:<class 'tuple'>: (y_9, x_9, object_number_max, 9 conners*2(x,y)):
                                the control points deta about x,y of each cell
        """

        # set grid cell
        cell_width = self.width1/self.width_seg
        cell_height = self.height1/self.height_seg
        top_left_x = np.empty([1, self.width_seg])
        top_left_y = np.empty([1, self.height_seg])

        object_number_max = 5
        object_number_current = np.zeros((self.height_seg, self.width_seg, 1))
        cell_class = np.zeros((self.height_seg, self.width_seg, object_number_max, 1))

        for i_width in range(self.width_seg):
            top_left_x[0][i_width] = i_width*cell_width
        for i_height in range(self.height_seg):
            top_left_y[0][i_height] = i_height*cell_height


        # find control points into grid cell
        corner_number = 9
        corner_point_deta = np.zeros((self.height_seg, self.width_seg, object_number_max, corner_number*2))

        for control_point in self.control_points_list:
            if control_point[8][0]>VALID_AREA[2] and control_point[8][0]<VALID_AREA[3]:
                for i_corner in range(corner_number):
                    # resize the box
                    control_point[i_corner][0] = control_point[i_corner][0] - 350
                    control_point[i_corner][0] = control_point[i_corner][0] * self.res_ratio
                    control_point[i_corner][1] = control_point[i_corner][1] * self.res_ratio

                # find the center points location
                center_point_x = control_point[8][0]
                center_point_y = control_point[8][1]
                center_point_x_min, center_point_y_min = self.corner_loc(center_point_x, center_point_y,
                                                                         top_left_x, top_left_y)

                object_box_order = int(object_number_current[center_point_y_min][center_point_x_min][0])
                if object_box_order < 5:
                    # base point is at the left top, right is x ,down is y
                    cell_class[center_point_y_min][center_point_x_min][object_box_order][0] = 1

                    # find the offsets of other eight control points
                    for i_corner in range(corner_number):
                        corner_point_deta[center_point_y_min][center_point_x_min][object_box_order][i_corner] = \
                        control_point[i_corner][0] - top_left_x[0][center_point_x_min]
                        corner_point_deta[center_point_y_min][center_point_x_min][object_box_order][i_corner+1] = \
                        control_point[i_corner][1] - top_left_y[0][center_point_y_min]

                        if i_corner ==8:
                            assert control_point[i_corner][1]>0, 'center_point_y smaller than 0'
                            assert control_point[i_corner][0] > 0, 'center_point_x smaller than 0'

                    # next object box
                    object_number_current[center_point_y_min][center_point_x_min][0] += 1

                else:
                    # the object number is larger than 5 in this grid
                    pass
            else:
                # the box is not in the valid area
                pass

        return cell_class, corner_point_deta

    def corner_loc(self, point_x, point_y, top_left_x, top_left_y):
        """
        Get the gird location of each corner.
        :param point_x: x of corner
        :param point_y: y of corner
        :param top_left_x: gird rule
        :param top_left_y: grid rule
        :return: the location of corner of for the grid
        """
        _, center_point_x_min = np.where(top_left_x <= point_x)
        center_point_x_min = center_point_x_min[-1]
        _, center_point_y_min = np.where(top_left_y <= point_y)
        center_point_y_min = center_point_y_min[-1]

        return center_point_x_min, center_point_y_min