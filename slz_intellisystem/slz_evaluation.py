import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from mavros_msgs.msg import Altitude
from slz_msgs.msg import Evaluation, Images

from cv_bridge import CvBridge

import numpy as np
from tensorflow import keras
import segmentation_models as sm

from .submodules import b_level
from .submodules.params import IMG_SHAPE


class SLZ_Evaluation(Node):

    def __init__(self):
        super().__init__('slz_evaluation')
        self.pub_est = self.create_publisher(Evaluation, '/slz_intellisystem/evaluation', 10)
        self.pub_imgs = self.create_publisher(Images, '/slz_intellisystem/images', 10)

        self.pub_img_forward = self.create_publisher(Image, '/slz_intellisystem/image/forward', 10)
        self.pub_img_direct = self.create_publisher(Image, '/slz_intellisystem/image/direct', 10)
        self.pub_img_slz = self.create_publisher(Image, '/slz_intellisystem/image/slz', 10)
        self.pub_img_slp = self.create_publisher(Image, '/slz_intellisystem/image/slp', 10)

        timer_period = 1.5 # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.subscription = self.create_subscription(
            Image,
            '/camera/aligned_depth_to_color/image_raw',
            self._depth_callback,
            10)
        
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self._color_callback,
            10)
        
        self.subscription = self.create_subscription(
            Altitude,
            '/mavros/global_position/rel_alt',
            self._altitude_callback,
            10)
        
        self.bridge = CvBridge()
        self.model = self._get_model()

        # input data 
        self.color = np.zeros(IMG_SHAPE)
        self.depth = np.zeros(IMG_SHAPE)
        self.altitude = 0.0

        # output Images.msg
        self.forward_mask = np.zeros(IMG_SHAPE)
        self.direct_mask = np.zeros(IMG_SHAPE)
        self.slz_mask = np.zeros(IMG_SHAPE)
        self.slp_mask = np.zeros(IMG_SHAPE)

        # output Evaluation.msg
        self.x_coord = 0.0
        self.y_coord = 0.0
        self.radius = 0.0
        self.valid = False
        self.etype = 0


    def __dice_loss_plus_1focal_loss():
        dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 2, 0.5]))
        focal_loss = sm.losses.CategoricalFocalLoss()
        return dice_loss + (1 * focal_loss)


    def _get_model(self, threshold=0.5):
        return keras.models.load_model(
            'model.h5', custom_objects = {
                'dice_loss_plus_1focal_loss': self.__dice_loss_plus_1focal_loss,
                'iou_score': sm.metrics.IOUScore(threshold=threshold),
                'f1-score': sm.metrics.FScore(threshold=threshold)
            }
        )


    def timer_callback(self):
        self.forward_mask, self.direct_mask, self.slz_mask, self.slp_mask,\
            self.x_coord, self.y_coord, self.radius, self.valid, self.etype =\
                b_level.disp(self.model, self.color, self.depth, self.altitude)
        
        msg_est = Evaluation()
        msg_est.x_coord = float(self.x_coord)
        msg_est.y_coord = float(self.y_coord)
        msg_est.radius = self.radius
        msg_est.valid = self.valid
        msg_est.etype = self.etype

        img_forward = self.bridge.cv2_to_imgmsg(self.forward_mask * 255)
        img_direct = self.bridge.cv2_to_imgmsg(self.direct_mask * 255)
        img_slz = self.bridge.cv2_to_imgmsg(self.slz_mask * 255)
        img_slp = self.bridge.cv2_to_imgmsg(self.slp_mask)

        msg_imgs = Images()
        msg_imgs.forward_mask = img_forward
        msg_imgs.direct_mask = img_direct
        msg_imgs.slz_mask = img_slz
        msg_imgs.slp_mask = img_slp

        # publish slz_msgs messages
        self.pub_est.publish(msg_est)
        self.pub_imgs.publish(msg_imgs)

        # publish image messages
        self.pub_img_forward.publish(img_forward)
        self.pub_img_direct.publish(img_direct)
        self.pub_img_slz.publish(img_slz)
        self.pub_img_slp.publish(img_slp)

        self.get_logger().info(f'Distance: {self.altitude} metres')


    def _depth_callback(self, msg):
        depth_image = self.bridge.imgmsg_to_cv2(msg, msg.encoding)
        self.depth = np.array(depth_image, dtype=np.float32) / 1000
        
        center_idx = np.array(self.depth.shape) // 2
        self.altitude = self.depth[center_idx[0], center_idx[1]]


    def _color_callback(self, msg):
        color_image = self.bridge.imgmsg_to_cv2(msg)
        self.color = np.array(color_image, dtype=np.uint8)

    
    def _altitude_callback(self, msg):
        pass
        #self.altitude = msg.data


def main(args=None):
    rclpy.init(args=args)

    slz_evaluation = SLZ_Evaluation()
    rclpy.spin(slz_evaluation)
    
    slz_evaluation.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

