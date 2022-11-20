import rospy
import sys

import numpy as np
import cupy as cp
import ros_numpy
import matplotlib.pyplot as plt
from skimage.io import imshow

import message_filters
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge

from semantic_pointcloud.pointcloud_parameters import PointcloudParameter
from semantic_pointcloud.utils import resolve_model


class PointcloudNode:
    def __init__(self, sensor_name):
        # TODO: if this is going to be loaded from another package we might need to change namespace
        config = rospy.get_param("/semantic_pointcloud/subscribers")
        self.param: PointcloudParameter = PointcloudParameter.from_dict(
            config[sensor_name]
        )
        self.param.sensor_name = sensor_name
        print("--------------Pointcloud Parameters-------------------")
        print(self.param.dumps_yaml())
        print("--------------End of Parameters-----------------------")

        # setup custom dtype
        self.create_custom_dtype()
        # setup semantics
        self.feature_extractor = None
        self.semantic_model = None
        self.segmentation_channels = None
        self.feature_channels = None
        self.initialize_semantics()

        # setup pointcloud creation
        self.cv_bridge = CvBridge()
        self.P = None

        rospy.Subscriber(self.param.cam_info_topic, CameraInfo, self.cam_info_callback)
        rgb_sub = message_filters.Subscriber(self.param.image_topic, Image)
        depth_sub = message_filters.Subscriber(self.param.depth_topic, Image)
        if self.param.confidence:
            confidence_sub = message_filters.Subscriber(
                self.param.confidence_topic, Image
            )
            ts = message_filters.ApproximateTimeSynchronizer(
                [
                    rgb_sub,
                    depth_sub,
                    confidence_sub,
                ],
                queue_size=10,
                slop=0.5,
            )
        else:
            ts = message_filters.ApproximateTimeSynchronizer(
                [
                    rgb_sub,
                    depth_sub,
                ],
                queue_size=10,
                slop=0.5,
            )
        ts.registerCallback(self.image_callback)

        self.pcl_pub = rospy.Publisher(self.param.topic_name, PointCloud2, queue_size=2)
        if self.param.publish_segmentation_image:
            self.seg_pub = rospy.Publisher(
                self.param.segmentation_image_topic, Image, queue_size=2
            )
        if  self.segmentation_channels is not None:
            self.semseg_color_map = self.color_map(
                len(list(self.segmentation_channels.keys()))
            )
            self.color_map_viz()

    def color_map(self, N=256, normalized=False):
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        dtype = "float32" if normalized else "uint8"
        cmap = np.zeros((N + 1, 3), dtype=dtype)
        for i in range(N + 1):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap / 255 if normalized else cmap
        return cmap[1:]

    def color_map_viz(self):
        # labels = self.semantic_model["model"].get_classes()
        labels = list(self.segmentation_channels.keys())
        nclasses = len(labels)
        row_size = 50
        col_size = 500
        cmap = self.semseg_color_map
        array = np.empty(
            (row_size * (nclasses), col_size, cmap.shape[1]), dtype=cmap.dtype
        )
        for i in range(nclasses):
            array[i * row_size : i * row_size + row_size, :] = cmap[i]
        imshow(array)
        plt.yticks([row_size * i + row_size / 2 for i in range(nclasses)], labels)
        plt.xticks([])
        plt.show()
        # plt.savefig("./labels.png")

    def initialize_semantics(self):
        if self.param.semantic_segmentation:
            self.semantic_model = resolve_model(self.param.segmentation_model)
            class_to_idx = {
                cls: idx
                for (idx, cls) in enumerate(self.semantic_model["model"].get_classes())
            }
            print(
                "Semantic Segmentation possible channels: ",
                self.semantic_model["model"].get_classes(),
            )
            indices = []
            channels = []
            for it, chan in enumerate(self.param.channels):
                if chan in [cls for cls in list(class_to_idx.keys())]:
                    indices.append(class_to_idx[chan])
                    channels.append(chan)
                elif self.param.fusion[it] in ["class_average", "class_bayesian"]:
                    print(chan, " is not in the semantic segmentation model.")
            self.segmentation_channels = dict(zip(channels, indices))
        if self.param.feature_extractor:
            self.feature_channels = []
            for i, fusion in enumerate(self.param.fusion):
                if fusion == "average":
                    self.feature_channels.append(self.param.channels[i])
            assert len(self.feature_channels) > 0
            self.feature_extractor = resolve_model(
                self.param.feature_config.name, self.param.feature_config
            )

    def create_custom_dtype(self):
        self.dtype = [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
        ]
        for chan in self.param.channels:
            self.dtype.append((chan, np.float32))
        print(self.dtype)

    def cam_info_callback(self, msg):
        a = cp.asarray(msg.P)
        self.P = cp.resize(a, (3, 4))
        self.height = msg.height
        self.width = msg.width
        self.header = msg.header

    def image_callback(self, rgb_msg, depth_msg, confidence_msg=None):
        confidence = None
        if self.P is None:
            return
        image = cp.asarray(
            self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        )
        depth = cp.asarray(
            self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        )
        if confidence_msg is not None:
            confidence = cp.asarray(
                self.cv_bridge.imgmsg_to_cv2(
                    confidence_msg, desired_encoding="passthrough"
                )
            )

        pcl = self.create_pcl_from_image(image, depth, confidence)

        self.publish_pointcloud(pcl, depth_msg.header)

    def create_pcl_from_image(self, image, depth, confidence):
        u, v = self.get_coordinates(depth, confidence)

        # create pointcloud
        world_x = (u.astype(np.float32) - self.P[0, 2]) * depth[v, u] / self.P[0, 0]
        world_y = (v.astype(np.float32) - self.P[1, 2]) * depth[v, u] / self.P[1, 1]
        world_z = depth[v, u]
        points = np.zeros(world_x.shape, dtype=self.dtype)
        points["x"] = cp.asnumpy(world_x)
        points["y"] = cp.asnumpy(world_y)
        points["z"] = cp.asnumpy(world_z)
        self.process_image(image, u, v, points)
        return points

    def get_coordinates(self, depth, confidence):
        pos = cp.where(depth > 0, 1, 0)
        low = cp.where(depth < 8, 1, 0)
        if confidence is not None:
            conf = cp.where(confidence >= self.param.confidence_threshold, 1, 0)
        else:
            conf = cp.ones(pos.shape)
        fin = cp.isfinite(depth)
        temp = cp.maximum(cp.rint(fin + pos + conf + low - 2.6), 0)
        mask = cp.nonzero(temp)
        u = mask[1]
        v = mask[0]
        return u, v

    def publish_pointcloud(self, pcl, header):
        pc2 = ros_numpy.msgify(PointCloud2, pcl)
        pc2.header = header
        pc2.header.frame_id = self.param.cam_frame
        self.pcl_pub.publish(pc2)

    def process_image(self, image, u, v, points):
        if "color" in self.param.fusion:
            valid_rgb = image[v, u].get()
            r = np.asarray(valid_rgb[:, 0], dtype=np.uint32)
            g = np.asarray(valid_rgb[:, 1], dtype=np.uint32)
            b = np.asarray(valid_rgb[:, 2], dtype=np.uint32)
            rgb_arr = np.array((r << 16) | (g << 8) | (b << 0), dtype=np.uint32)
            rgb_arr.dtype = np.float32
            position = self.param.fusion.index("color")
            points[self.param.channels[position]] = rgb_arr

        if self.segmentation_channels is not None:
            self.perform_segmentation(image, points, u, v)
        if self.feature_channels is not None:
            # TODO is no return needed?
            self.extract_features(image, points, u, v)

    def perform_segmentation(self, image, points, u, v):
        prediction = self.semantic_model["model"](image)
        mask = prediction[list(self.segmentation_channels.values())]
        values = mask[:, v.get(), u.get()].cpu().detach().numpy()
        for it, channel in enumerate(self.segmentation_channels.keys()):
            points[channel] = values[it]
        if self.param.publish_segmentation_image:
            self.publish_segmentation_image(mask)

    def extract_features(self, image, points, u, v):
        prediction = self.feature_extractor["model"](image)
        values = prediction[:, v.get(), u.get()].cpu().detach().numpy()
        for it, channel in enumerate(self.feature_channels):
            points[channel] = values[it]

    def publish_segmentation_image(self, probabilities):
        probabilities = cp.asarray(probabilities)
        colors = cp.asarray(self.semseg_color_map)
        assert colors.ndim == 2 and colors.shape[1] == 3
        img = cp.argmax(probabilities, axis=0)
        img = colors[img].astype(cp.uint8)  # N x H x W x 3
        img = img.get()
        seg_msg = self.cv_bridge.cv2_to_imgmsg(img, encoding="rgb8")
        seg_msg.header.frame_id = self.header.frame_id
        self.seg_pub.publish(seg_msg)


if __name__ == "__main__":
    sensor_name = sys.argv[1]
    rospy.init_node("semantic_pointcloud_node", anonymous=True, log_level=rospy.INFO)
    node = PointcloudNode(sensor_name)
    rospy.spin()
