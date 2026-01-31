from common.mujoco_viewer import BaseViewer

class BasicDemoV1(BaseViewer):
    def __init__(self):
        # 模型路径
        model_path = "./model/franka_emika_panda/scene.xml"
        super().__init__(model_path)

    def pre_process(self):
        # 目标关节位置
        self.target_positions = [1.13, -0.811, 0.811, -1.05, -1.65, 1.08, 1.45, 0.04, 0.04]
        # 运行到目标位置总共消耗的时间步长
        self.total_step = 30000
        # 计算每个时间步各关节需要移动的增量
        self.increments = [target_position / self.total_step for target_position in self.target_positions]
        print(f'increments: {self.increments}')

        # 当前时间步
        self.step = 0
        # 下一时刻关节位置
        self.next_positions = []

    def step_callback(self):
        # 获取关节位置
        last_positions = self.data.qpos[:9]
        print(f'last_positions: {last_positions}')

        # 动态计算下一时刻关节位置
        self.next_positions.clear()
        for i in range(len(self.target_positions)):
            self.next_positions.append(self.increments[i] * (self.step + 1))
        # 将下一时刻关节位置更新回data对象
        self.data.qpos[:9] = self.next_positions

        self.step += 1
        if self.step > self.total_step:
            print("机械臂已到达目标位置")
            self.handle.close()

if __name__ == "__main__":
    demo = BasicDemoV1()
    demo.start()