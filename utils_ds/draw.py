import numpy as np
import cv2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

IN = set()
OUT = set()

# 定义直线拟合函数
def fit_line(points):
    # 使用np.polyfit()进行一次线性拟合，得到斜率和截距
    k, b = np.polyfit(points[:,0], points[:,1], 1)
    # 返回斜率和截距
    return k, b

# 计算直线的方程式
def line_equation(line_start, line_end):
    m = (line_end[1] - line_start[1]) / (line_end[0] - line_start[0])  # 斜率
    c = line_start[1] - m * line_start[0]  # 截距
    return m, c

# 判断点是否在直线上方还是下方
def point_position(point, m, c):
    y_on_line = m * point[0] + c
    if point[1] < y_on_line:
        return "Above"
    elif point[1] > y_on_line:
        return "Below"
    else:
        return "On"

# 判断从上往下穿过直线还是从下往上穿过直线
def crossing_direction(tracking_points, line_start, line_end):


    # ['Above',..,'On','Below',...]     属于从上往下穿过 "Crossing","From_above"
    # ['Below',..,'On','Above',...]     属于从下往上穿过 "Not_crossing","From_above"
    # ['Above',..,'Below',...]          属于从上往下穿过 "Crossing","From_above"
    # ['Below',..,'On','Above',...]     属于从下往上穿过 "Not_crossing","From_above"
    # ['Below',..,'Below']              未穿过 "Not_crossing", None
    # ['Above',..,'Above']              未穿过 "Not_crossing", None
    # 根据上面的情况进行判断返回。
    # 判断是否穿过直线
    m, c = line_equation(line_start, line_end)
    above_line_count = 0
    below_line_count = 0
    
    for point in tracking_points:
        position = point_position(point, m, c)
        if position == "Above":
            above_line_count += 1
        elif position == "Below":
            below_line_count += 1

    if above_line_count > 0 and below_line_count > 0:
        # 穿过直线
        if tracking_points[-1][1] < tracking_points[0][1]:  # 最后一个点在上面，说明是从上往下穿过直线
            return "Crossing", "From_above"
        else:
            return "Crossing", "From_below"
    else:
        return "Not_crossing", None
    

def draw_boxes(img, bbox,tracking_point_list, identities=None, offset=(0,0)):
    # 画直线
    line_start = (0, int(img.shape[0]* 0.7))
    line_end = (img.shape[1], int(img.shape[0]*0.7))
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


        

        # 计算直线的斜率和截距
        slope = (line_end[1] - line_start[1]) / (line_end[0] - line_start[0])
        intercept = line_start[1] - slope * line_start[0]
        # cv2.line(img, line_start, line_end, (0, 0, 255), 2)
        # 判断目标跟踪点是从上面穿过直线还是从下面穿过直线
        crossing_status, direction = crossing_direction(tracking_point, line_start, line_end)
        if crossing_status == "Crossing":
            # print("Crossing direction:", direction)
            if direction == 'From_above':
                if not (int(identities[i]) if identities is not None else 0 )in IN:
                    IN.add(int(identities[i]) if identities is not None else 0)
                    cv2.line(img, line_start, line_end, (0, 255, 0), 2)
            elif direction == 'From_below':
                if not (int(identities[i]) if identities is not None else 0 )in OUT:
                    OUT.add(int(identities[i]) if identities is not None else 0)
                    cv2.line(img, line_start, line_end, (255, 0, 0), 2)
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

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        for point in tracking_point:
            cv2.circle(img, tuple(point), 1, color, -1, lineType=cv2.LINE_AA, shift=0)
        # cv2.arrowedLine(img, (x1_t, y1_t),(x2_t, y2_t), color, 3)
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)

        # 方向
        # 使用反正切函数计算角度（单位为弧度）
        angle_rad = np.arctan(k)

        # 将弧度转换为角度
        angle_deg = np.degrees(angle_rad)% 360
        # print('拟合直线的角度为：', angle)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
        cv2.putText(img,"{:.2f}".format(angle_deg),(x2_t,y2_t), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 1)
    # print(f'out:{len(OUT)},in:{len(IN)}')
    return img,len(OUT),len(IN)



if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
