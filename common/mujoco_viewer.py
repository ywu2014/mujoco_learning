import time
import mujoco
import mujoco.viewer
import glfw
import cv2
import numpy as np

class BaseViewer:
    def __init__(self, model_path, sleep_time:float=None):
        """
        mujoco viewer
        
        :param self: Description
        :param model_path: MuJoCo模型XML文件的路径
        """
        self.model_path = model_path
        # 获取模型对象, 包含了物理场景的静态描述, 如：物体的质量、几何形状、关节类型、连接方式、传感器定义
        self.model = mujoco.MjModel.from_xml_path(model_path)
        # 获取运行时数据对象, 包含了物理场景的动态状态, 如：关节的角度、速度、加速度、激活状态、传感器测量值、仿真时间等
        self.data = mujoco.MjData(self.model)
        # 图形窗口句柄
        self.handle = None

        # 每个step运行等待时间
        self.sleep_time = None

    def is_running(self):
        """
        模拟器是否还在运行
        """
        return self.handle.is_running()

    def sync(self):
        """
        将物理引擎（data）中计算出的最新状态，同步并渲染到图形窗口中
        """
        self.handle.sync()

    def _get_obj_id(self, obj_type, obj_name:str):
        """
        获取物体(body、camera等)的运行时id
        
        :param obj_type: 物体类型, mujoco.mjtObj.mjOBJ_xxx
        :param obj_name: 物体名称, MJCF中物体的name属性
        """
        return mujoco.mj_name2id(self.model, obj_type, obj_name)
    
    def get_body_id(self, name:str):
        """
        获取body物体的运行时id
        
        :param name: body 名称
        """
        return self._get_obj_id(mujoco.mjtObj.mjOBJ_BODY, name)

    def start(self):
        """
        起动 MuJoCo 模型的主循环
        """
        self.handle = mujoco.viewer.launch_passive(self.model, self.data)

        # 前置处理, 循环前的一些自定义初始化工作
        self.pre_process()

        while self.is_running():
            # 执行一次前向动力学模拟
            mujoco.mj_forward(self.model, self.data)

            self.step_callback()

            # 将物理仿真世界向前推进一步（通常是一个时间步长）
            mujoco.mj_step(self.model, self.data)

            self.sync()

            # 时间等待
            if self.sleep_time:
                time.sleep(self.sleep_time)
    
    def pre_process(self):
        """
        MuJoCo 模型主循环前置处理
        """
        pass

    def step_callback(self):
        """
        每个时间步处理回调
        """
        pass

class CameraViewer(BaseViewer):
    def __init__(self, model_path, sleep_time = None):
        super().__init__(model_path, sleep_time)

        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)

        self.cameras = {}

    def get_camera_id(self, name:str):
        """
        获取camera物体的运行时id
        
        :param name: camera 名称
        """
        camera_id = self._get_obj_id(mujoco.mjtObj.mjOBJ_CAMERA, name)
        if camera_id == -1:
            raise ValueError(f"Camera '{name}' not found")
        return camera_id
    
    def pre_process(self):
        super().pre_process()
        return 
    
    def get_camera_type(self, mode:int):
        if mode == 0:
            return mujoco.mjtCamera.mjCAMERA_FIXED
        
        raise Exception(f'invalid mode value {mode}')
    
    def add_camera(self, camera, name, cam_type, context, resolution):
        self.cameras[name] = {
            'name': name,
            'camera': camera,
            'type': cam_type,
            'context': context,
            'resolution': resolution
        }

    def init_camera(self, name:str):
        camera = mujoco.MjvCamera()
        cam_id = self.get_camera_id(name)
        cam_mode = self.model.cam_mode[cam_id]
        cam_type = self.get_camera_type(cam_mode)
        cam_resolution = self.model.cam_resolution[cam_id]  # 分辨率
        if cam_resolution is None:
            print("Camera resolution is not set. Using default resolution.")
            cam_resolution = np.array([640, 480])
        print(f'初始化相机, 名称: {name}, 类型: {cam_type}, 分辨率: {cam_resolution}')

        camera.fixedcamid = cam_id
        camera.type = cam_type

        # 初始化 GLFW
        if not glfw.init():
            return False

        window = glfw.create_window(cam_resolution[0], cam_resolution[1], 'Dobot Sim Environment', None, None)
        if not window:
            glfw.terminate()
            return False
        glfw.make_context_current(window)

        context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        self.add_camera(camera, name, cam_type, context, cam_resolution)
    
    def step_callback(self):
        for name, cam in self.cameras.items():
            camera = cam['camera']
            context = cam['context']
            resolution = cam['resolution']
            color_img, depth_img = self.get_image(camera, context, resolution[0], resolution[1])
            self.image_process_callback(name, color_img, depth_img)
    
    def get_image(self, camera, context, w, h):
        # 定义视口大小
        viewport = mujoco.MjrRect(0, 0, w, h)
        # 更新场景
        mujoco.mjv_updateScene(
            self.model, self.data, mujoco.MjvOption(), 
            None, camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene
        )
        # 渲染到缓冲区
        mujoco.mjr_render(viewport, self.scene, context)
        # 读取 RGB 数据（格式为 HWC, uint8）
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        depth = np.zeros((h, w), dtype=np.float64)
        mujoco.mjr_readPixels(rgb, depth, viewport, context)
        cv_image = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)

        # 参数设置
        min_depth_m = 0.0  # 最小深度（0米）
        max_depth_m = 8.0  # 最大深度（8米）
        near_clip = 0.1    # 近裁剪面（米）
        far_clip = 50.0    # 远裁剪面（米）
        # 将非线性深度缓冲区值转换为线性深度（米）
        # 公式: linear_depth = far * near / (far - (far - near) * depth)
        linear_depth_m = far_clip * near_clip / (far_clip - (far_clip - near_clip) * depth)
        # 裁剪深度到0-8米范围
        depth_clipped = np.clip(linear_depth_m, min_depth_m, max_depth_m)
        # 映射0-8米到0-255像素值（距离越小越亮）
        # 反转映射：距离越小值越大（越亮）
        inverted_depth = max_depth_m - depth_clipped
        # 计算缩放因子：255/(max_depth_m - min_depth_m)
        scale = 255.0 / (max_depth_m - min_depth_m)
        depth_visual = (inverted_depth * scale).astype(np.uint8)
        # 翻转图像（MuJoCO坐标系到OpenCV坐标系）
        depth_visual = np.flipud(depth_visual)
        return cv_image, depth_visual
    
    def image_process_callback(self, name, color_img, depth_img):
        """
        图象处理回调
        
        :param color_img: 相机名称
        :param color_img: 彩色图象数据
        :param depth_img: 深度图象数据
        """
        pass
    
    # def update_camera(self):
    #     if not glfw.window_should_close(self.window):
    #         img, depth_img = self.get_image(self.resolution[0], self.resolution[1])
            

    #         # 交换前后缓冲区
    #         glfw.swap_buffers(self.window)
    #         glfw.poll_events()