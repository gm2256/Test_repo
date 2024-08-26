import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
# yolo 추가 

class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')
        self.publisher_ = self.create_publisher(Image, 'raw_video_frames', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)
        self.model = YOLO('/home/sineunji/open_study/YOLOcode/runs/detect/train/weights/best.pt') 
        self.cups_detected = 0
    
    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning('Failed to capture image')
            return
        
        # 밝기 조절 # 모델 예측
        brightness_adjusted = cv2.convertScaleAbs(frame, alpha=0.68, beta=0)
        predictions = self.model.predict(brightness_adjusted[0:330, 68:570], conf=0.5)
        #바운드 박스 값 송신 
        results = predictions
        # 모델 돌린 결과 값 
        allbox = results[0].boxes.xyxy.cpu().detach().numpy().tolist() 
        #값이 존재?
        if not allbox:
            return                
        #컵 개수를 알 수 있는 값
        for result in results:
            boxes = result.boxes
            self.cups_detected += len(boxes)       
        #좌표 송신 

        
       
        #비디오 송신
        img_msg = self.bridge.cv2_to_imgmsg(plots, "bgr8")
        self.publisher_.publish(img_msg)
        self.get_logger().info('Published video frame')


def main(args=None):
    rclpy.init(args=args)
    video_publisher = VideoPublisher()
    rclpy.spin(video_publisher)
    video_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
