import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image
from mavros_msgs.msg import Altitude

from cv_bridge import CvBridge

import numpy as np
from tensorflow import keras
import segmentation_models as sm

import b_level

from params import IMG_SHAPE


class SLZ_Evaluation(Node):

    def __init__(self):
        super().__init__('slz_evaluation')
        self.publisher_ = self.create_publisher(String, '/slz/evaluation', 10)
        timer_period = 2.0  # seconds
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
            '../include/model.h5', custom_objects = {
                'dice_loss_plus_1focal_loss': self.__dice_loss_plus_1focal_loss,
                'iou_score': sm.metrics.IOUScore(threshold=threshold),
                'f1-score': sm.metrics.FScore(threshold=threshold)
            }
        )


    def timer_callback(self):
        msg = String()
        msg.data = 'This is SLZ IntelliSystem!'
        self.publisher_.publish(msg)

        #Self code
        res = b_level.disp(self.model, self.color, self.model, self.altitude)

        msg.data = f'This is SLZ IntelliSystem: {res[1]}, {res[2]}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')


    def _depth_callback(self, msg):
            depth_image = self.bridge.imgmsg_to_cv2(msg)
            self.depth = np.array(depth_image, dtype=np.float32)


    def _color_callback(self, msg):
            self.color = self.bridge.imgmsg_to_cv2(msg)

    
    def _altitude_callback(self, msg):
            self.altitude = msg.data


def main(args=None):
    rclpy.init(args=args)

    slz_evaluation = SLZ_Evaluation()
    rclpy.spin(slz_evaluation)
    
    slz_evaluation.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

