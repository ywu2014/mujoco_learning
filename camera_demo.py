from common.mujoco_viewer import CameraViewer
import cv2

class CameraDemo(CameraViewer):
    def __init__(self):
        # 模型路径
        model_path = "./model/samples/camera_demo1.xml"
        super().__init__(model_path)

        self.camera_window_exist = False

    def image_process_callback(self, name, color_img, depth_img):
        # print(color_img.shape)
        if not self.camera_window_exist:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, 1280, 720)
        cv2.imshow(name, color_img)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = CameraDemo()
    demo.init_camera('test_camera')

    demo.start()