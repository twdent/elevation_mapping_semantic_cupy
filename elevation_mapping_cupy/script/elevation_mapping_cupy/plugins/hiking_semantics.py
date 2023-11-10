#!/usr/bin/env conda run -n thesis python

import rospy
from sensor_msgs.msg import CompressedImage, CameraInfo,Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import PIL


# Import Segformer model here
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from torch import nn

MODEL_PATH = '/home/twdenton/Documents/Thesis/segformer-b0-finetuned-HikingHD'


class SegmentationNode:
    def __init__(self):
        rospy.init_node('segmentation_node', anonymous=False)

        # Subscribers
        self.image_sub = rospy.Subscriber('/wide_angle_camera_rear/image_color_rect/compressed', CompressedImage, self.image_callback)
        self.info_sub = rospy.Subscriber('/wide_angle_camera_rear/camera_info', CameraInfo, self.info_callback)

        # Publisher
        self.segmented_image_pub = rospy.Publisher('/traversable_segmentation/segmentation', Image, queue_size=10)
        self.segmented_info_pub = rospy.Publisher('/traversable_segmentation/camera_info', CameraInfo, queue_size=10)
        # Initialize the CvBridge
        self.bridge = CvBridge()

        # Initialize model here
        self.processor = SegformerImageProcessor.from_pretrained(MODEL_PATH)
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH)

    def image_callback(self, msg):
        try:
            # Convert compressed image to OpenCV format
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Perform segmentation model
            segmented_image = self.segment_image(cv_image)

            # Convert the segmented image back to Image format
            segmented_msg = self.bridge.cv2_to_imgmsg(segmented_image, encoding='passthrough')
            #set the frame id
            segmented_msg.header.frame_id = msg.header.frame_id
            #set the timestamp
            segmented_msg.header.stamp = rospy.Time.now()
            # Publish the segmented image and camera info
            self.segmented_image_pub.publish(segmented_msg)



        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")

    def info_callback(self, msg):
        # Publish the camera info
        self.segmented_info_pub.publish(msg)

    def segment_image(self, image):
        # Perform segmentation 
        #conver to PIL image
        image = PIL.Image.fromarray(image)

        # Preprocess image
        inputs = self.processor(image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits # shape (batch_size, num_labels, height/4, width/4)

        # First, rescale logits to original image size
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1], # (height, width)
            mode='bilinear',
            align_corners=False
        )

        # Second, apply argmax on the class dimension
        pred_seg = upsampled_logits.argmax(dim=1)[0]

        #ovelay the segmentation on the original image
        seg_overlay = self.get_seg_overlay(image, pred_seg)



        return seg_overlay
    
    def get_seg_overlay(self, image, seg):
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
        palette = np.array([
                            [0, 0, 0],          # unlabeled
                            [0, 155, 255],      # traversable
                            [255, 255, 0],      # untraversable
                            ])
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color

        # Show image + mask
        img = np.array(image) * 0.7 + color_seg * 0.3
        img = img.astype(np.uint8)

        return img

if __name__ == '__main__':
    #test the segmentation node

    try:
        segmentation_node = SegmentationNode()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
