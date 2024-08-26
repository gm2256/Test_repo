import rclpy as rp
from rclpy.node import Node
from my_first_package_msgs.srv import Check
import time

class Bincupclient(Node):
    
    def __init__(self):
        super().__init__('XyGrapereceive')
    
        self.client = self.create_client(Check, 'get_coord') # 같은 서비스 이름 타입
     
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        
        self.request = Check.Request()
        self.timer_period = 20 # 요청을 보낼 주기 (초 단위)
        self.timer = self.create_timer(self.timer_period, self.send_request)
        self.future = None
     
    # 요청 보내기 함수 
    def send_request(self):    
        self.request.signal = 1
        self.future = self.client.call_async(self.request)
        self.get_logger().info('Request sent')
    
    def check_response(self):
        if self.future.done():
            try:
                response = self.future.result()
            except Exception as e:
                self.get_logger().info(f'Service call failed: {e}')
            else:
                self.get_logger().info(f'Result: x={response.x}, y={response.y}')
            # 다시 요청을 보내기 위해 future 초기화
            self.future = None

def main(args=None):
    rp.init(args=args)
    coordinate_client = Bincupclient() # 고객 노드 생성 
    
    while rp.ok():
        rp.spin_once(coordinate_client, timeout_sec=1.0)
        if coordinate_client.future:
            coordinate_client.check_response()
            time.sleep(5) 
            
    coordinate_client.destroy_node()
    rp.shutdown()

if __name__ == '__main__':
    main()
      
