import mujoco
import mujoco.viewer
import time

# 模型路径
model_path = "./model/franka_emika_panda/scene.xml"
# 获取模型对象, 包含了物理场景的静态描述, 如：物体的质量、几何形状、关节类型、连接方式、传感器定义
model = mujoco.MjModel.from_xml_path(model_path)
# 获取运行时数据对象, 包含了物理场景的动态状态, 如：关节的角度、速度、加速度、激活状态、传感器测量值、仿真时间等
data = mujoco.MjData(model)

# 目标关节位置
target_positions = [1.13, -0.811, 0.811, -1.05, -1.65, 1.08, 1.45, 0.04, 0.04]
# 运行到目标位置总共消耗的时间步长
total_step = 30000
# 计算每个时间步各关节需要移动的增量
increments = [target_position / total_step for target_position in target_positions]
print(f'increments: {increments}')

# 启动 MuJoCo 物理模拟器
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 当前时间步
    step = 0
    # 下一时刻关节位置
    next_positions = []
    while viewer.is_running():
        # 执行一次前向动力学模拟
        mujoco.mj_forward(model, data)

        # 获取关节位置
        last_positions = data.qpos[:9]
        print(f'last_positions: {last_positions}')

        # 动态计算下一时刻关节位置
        next_positions.clear()
        for i in range(len(target_positions)):
            next_positions.append(increments[i] * (step + 1))
        # 将下一时刻关节位置更新回data对象
        data.qpos[:9] = next_positions
        
        # 将物理仿真世界向前推进一步（通常是一个时间步长）
        mujoco.mj_step(model, data)

        # 将物理引擎（data）中计算出的最新状态，同步并渲染到图形窗口中
        viewer.sync()

        # time.sleep(0.1)

        step += 1
        if step > total_step:
            print("机械臂已到达目标位置")
            break
