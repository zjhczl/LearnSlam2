# 旋转表示方法的转换
import numpy as np
from scipy.spatial.transform import Rotation as R

# # 从四元素构造旋转
# r = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])

# # 从旋转矩阵构造旋转
# r = R.from_matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

# # 从旋转向量构造旋转
# r = R.from_rotvec(np.pi/2 * np.array([0, 0, 1]))

# 从欧拉角构造旋转
# r = R.from_euler("z", 90, degrees=True)
r = R.from_euler("xz", [0, 90], degrees=True)

# 四元素形式
print(r.as_quat())
# 矩阵形式
print(r.as_matrix())
# 旋转向量
print(r.as_rotvec())
# 欧拉角
print(r.as_euler("zyx", degrees=True))
################################################

# 旋转向量
v = [1, 1, 0]
print(r.apply(v))
# 两次旋转
print(r.apply(r.apply(v)))
r2 = r * r
print(r2.apply(v))

# 反旋转
r3 = r.inv()
print(r3.apply(v))
