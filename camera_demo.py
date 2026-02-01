from common.mujoco_viewer import CameraViewer
import cv2

class CameraDemo(CameraViewer):
    def __init__(self):
        # 模型路径
        model_path = "./model/samples/camera_demo1.xml"
        super().__init__(model_path)

    def image_callback(self, color_img):
        print(color_img.shape)
        cv2.imshow('MuJoCo Camera Output', color_img)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = CameraDemo()
    demo.init_camera('test_camera')

    demo.start()