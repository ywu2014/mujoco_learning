from common.mujoco_viewer import CameraViewer
import cv2

class CameraDemo(CameraViewer):
    def __init__(self):
        # 模型路径
        model_path = "./model/samples/camera_demo1.xml"
        super().__init__(model_path)

        self.camera_window_exist = {
            'test_camera1': False,
            'test_camera2': False,
        }

    def image_process_callback(self, name:str, color_img):
        print(f'name: {name}, shape: {color_img.shape}')
        if not self.camera_window_exist[name]:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, 1080, 720)
            self.camera_window_exist[name] = True
        cv2.imshow(name, color_img)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = CameraDemo()
    demo.init_camera('test_camera1')
    demo.init_camera('test_camera2')

    demo.start()