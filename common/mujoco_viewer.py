import time
import mujoco
import mujoco.viewer

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