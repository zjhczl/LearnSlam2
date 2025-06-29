# 矩阵的基本运算
import numpy as np
from scipy import linalg

A = np.array([1, 2, 3])
# 参数还可以是一个已有的list类型
list_b = [1, 2, 3]
B = np.array(list_b)
C = np.array([[1, 2, 3], [2, 3, 4]])

M33 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 8]])
print(M33)

# 转置
print(M33.T)
# 各元素和
print(M33.sum())
# 迹
# 矩阵A的迹 tr(A) 是其主对角线元素之和tr(A)=a11 +a22 +⋯+ann
trace = np.trace(M33)
print("迹 tr(A):", trace)


# 求解行列式a.
print("------------行列式------------")
A = np.array([[1, 2], [3, 4]])
x = linalg.det(A)
print(x)


# 逆
# 矩阵的逆（Inverse）矩阵 A 的逆矩阵 A−1满足：A⋅A−1=A−1⋅A=IA⋅A −1 =A −1 ⋅A=I
# 其中 I 是单位矩阵。条件：仅当矩阵是方阵（行数=列数）且行列式不为零（非奇异矩阵）时，逆矩阵存在。
# 1. 求逆矩阵
try:
    print("M33矩阵的行列式：" + str(linalg.det(M33)))
    M33_inv = np.linalg.inv(M33)  # 可能触发异常（当矩阵不可逆时）
    print("M33逆矩阵 A^{-1}:\n", M33_inv)
except np.linalg.LinAlgError:
    print("M33矩阵不可逆（奇异矩阵）")


# 求特征值和特征向量
# https://blog.csdn.net/webor2006/article/details/120909642
# 定义：满足 Av=λv Av=λv 的标量 λ 和向量 v
# 性质：


print("------------特征值和特征向量------------")
eigenvalues, eigenvectors = np.linalg.eig(M33)
print("\n特征值:", eigenvalues)
print("特征向量:\n", eigenvectors)
# 特征值之和 = 迹
print(eigenvalues.sum())
# 特征值之积 = 行列式
print(np.prod(eigenvalues))

# 矩阵和向量相乘
print(np.dot(M33, np.array([1, 2, 3]).T))


# 方程组求解
print("-------------方程组求解--------------")
a = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])  # 系数矩阵
print(a)
b = np.array([2, 4, -1])  # 结果
print(b)
print(linalg.solve(a, b))
# 左乘M的逆矩阵
print(np.dot(np.linalg.inv(a), b.T))

# 奇异值分解
# 奇异值分解(SVD)可以被认为是特征值问题扩展到非矩阵的矩阵。
# scipy.linalg.svd将矩阵'a'分解为两个酉矩阵'U'和'Vh'，以及一个奇异值(实数，非负)的一维数组's'，使得a == U * S * Vh，其中'S'是具有主对角线's'的适当形状的零点矩阵。
# 让我们来看看下面的例子。
print("-------------奇异值分解--------------")
# Declaring the numpy array
a = np.random.randn(3, 2) + 1.0j * np.random.randn(3, 2)

# Passing the values to the eig function
U, s, Vh = linalg.svd(a)

# printing the result
print(U, Vh, s)
