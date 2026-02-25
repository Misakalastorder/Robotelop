import xrobotoolkit_sdk as xrt
xrt.init()

while True:
    if xrt.get_A_button():
        print("A button pressed")
        break
xrt.close()

"""
Pico设备连接测试脚本
用于验证Pico设备的连接和数据获取功能
"""

# import sys
# import time
# import xrobotoolkit_sdk as xrt
# import numpy as np

# # 关节名称映射表
# JOINT_NAMES = [
#     "Pelvis", "Left Hip", "Right Hip", "Spine1", "Left Knee", "Right Knee",
#     "Spine2", "Left Ankle", "Right Ankle", "Spine3", "Left Foot", "Right Foot",
#     "Neck", "Left Collar", "Right Collar", "Head", "Left Shoulder", "Right Shoulder",
#     "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Hand", "Right Hand"
# ]

# def test_pico_connection():
#     """
#     测试Pico设备连接
#     """
#     print("开始Pico设备连接测试...")
    
#     try:
#         # 初始化SDK
#         print("正在初始化SDK...")
#         xrt.init()
#         print("SDK初始化成功")
        
#         # 简单测试循环
#         print("开始数据获取测试...")
#         frame_count = 0
#         max_frames = 10  # 只测试10帧数据
        
#         while frame_count < max_frames:
#             print(f"\n--- 帧 {frame_count + 1}/{max_frames} ---")
            
#             # 检查是否可以获取A按钮状态
#             try:
#                 a_button_pressed = xrt.get_A_button()
#                 print(f"A按钮状态: {a_button_pressed}")
#             except Exception as e:
#                 print(f"获取A按钮状态失败: {e}")
            
#             # 检查体感数据是否可用
#             if xrt.is_body_data_available():
#                 print("体感数据可用")
                
#                 # 获取关节姿态数据
#                 try:
#                     body_poses = xrt.get_body_joints_pose()
#                     print(f"获取到 {len(body_poses)} 个关节点数据")
                    
#                     # 显示部分关节数据作为测试
#                     for idx in [0, 16, 17, 18, 19]:  # 显示Pelvis, 左右肩膀, 左右肘部
#                         if idx < len(body_poses):
#                             joint_data = body_poses[idx]
#                             print(f"{JOINT_NAMES[idx]}: 位置{joint_data[:3]}, 四元数{joint_data[3:]}")
#                 except Exception as e:
#                     print(f"获取体感数据失败: {e}")
#             else:
#                 print("体感数据不可用，请检查Pico设备是否正确佩戴")
            
#             # 检查触发器状态
#             try:
#                 left_trigger = xrt.get_left_trigger()
#                 right_trigger = xrt.get_right_trigger()
#                 print(f"左触发器: {left_trigger}, 右触发器: {right_trigger}")
#             except Exception as e:
#                 print(f"获取触发器状态失败: {e}")
            
#             frame_count += 1
#             time.sleep(0.1)  # 每0.1秒获取一次数据
            
#         print("\nPico设备连接测试完成！")
        
#     except Exception as e:
#         print(f"Pico设备连接测试失败: {e}")
#         return False
    
#     finally:
#         # 关闭SDK
#         try:
#             xrt.close()
#             print("SDK已关闭")
#         except Exception as e:
#             print(f"关闭SDK时出错: {e}")
    
#     return True

# def main():
#     """
#     主函数
#     """
#     print("=" * 50)
#     print("Pico设备连接测试工具")
#     print("=" * 50)
    
#     success = test_pico_connection()
    
#     if success:
#         print("\n✓ Pico设备连接测试成功！")
#         print("设备工作正常，可以继续开发。")
#     else:
#         print("\n✗ Pico设备连接测试失败！")
#         print("请检查设备连接和驱动安装。")
    
#     print("=" * 50)

# if __name__ == "__main__":
#     main()