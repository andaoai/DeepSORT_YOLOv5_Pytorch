import numpy as np
import cv2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


# 定义直线拟合函数
def fit_line(points):
    # 使用np.polyfit()进行一次线性拟合，得到斜率和截距
    k, b = np.polyfit(points[:,0], points[:,1], 1)
    # 返回斜率和截距
    return k, b

def draw_boxes(img, bbox,tracking_point_list, identities=None, offset=(0,0)):

    for i,box in enumerate(bbox):
        tracking_point = tracking_point_list[i]
        # 对tracking_point进行线性拟合
        # 对数据点进行直线拟合
        tracking_point = np.array(tracking_point)
        k, b = fit_line(tracking_point)
        # 计算拟合直线与实际数据点之间的残差
        residuals = tracking_point[:, 1] - (k * tracking_point[:, 0] + b)

        # 计算损失值（残差平方和）
        loss = np.sum(residuals)

        # 计算起点和终点坐标
        x1_t = int(tracking_point[0, 0])
        y1_t = int(k * x1_t + b)
        x2_t = int(tracking_point[-1, 0])
        y2_t = int(k * x2_t + b)


        
        # print([arr.tolist() for arr in tracking_point])
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        # 计算拟合直线与实际数据点之间的残差
        residuals = tracking_point[:, 1] - (k * tracking_point[:, 0] + b)
        # 将所有负数转换为正数
        residuals[residuals < 0] *= -1
        # 计算损失值,越大，说明并没有按照指定路线进行驾驶
        loss = np.sum(residuals)

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        # for point in tracking_point:
        #     cv2.circle(img, tuple(point), 5, color, -1, lineType=cv2.LINE_AA, shift=0)
        cv2.arrowedLine(img, (x1_t, y1_t),(x2_t, y2_t), color, 3)
        # cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        # cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label+','+"{:.2f}".format(loss),(x2_t,y2_t), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 1)
    return img



if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
