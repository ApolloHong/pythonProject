import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as sgn


# def smooth_xy(lx, ly):
#     """数据平滑处理
#
#     :param lx: x轴数据，数组
#     :param ly: y轴数据，数组
#     :return: 平滑后的x、y轴数据，数组 slx, sly
#     """
#     x_smooth = np.linspace(lx.min(), lx.max(), 100)
#     y_smooth = scipy.imake_interp_spline(lx, ly)(x_smooth)
#     return x_smooth, y_smooth
def smooth_it(lly):
    y_smooth = sgn.savgol_filter(lly, 5, 3)
    y_smooth2 = sgn.savgol_filter(lly, 79, 3, mode='nearest')
    return y_smooth2

# if __name__ == '__main__':
#     x_raw = [6, 7, 8, 9, 10, 11, 12]
#     y_raw = [1.53, 5.92, 2.04, 7.24, 2.72, 1.10, 4.70]
#     xy_s = smooth_xy(x_raw, y_raw)
#
#     # 原始折线图
#     plt.plot(x_raw, y_raw)
#     plt.show()
#
#     # 处理后的平滑曲线
#     plt.plot(xy_s[0], xy_s[1])
#     plt.show()
