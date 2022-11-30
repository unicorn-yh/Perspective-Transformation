import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

class set_oldpoints_newpoints():    # 提取图像的旧定点坐标（旧区域）和新定点坐标（新区域），用于作透视变换

      def __init__(self, image_path):
            old_points = self.select_region_points(image_path,mode=False)    # 选择旧区域的定点
            new_points = self.select_region_points(image_path,mode=True,points=old_points) # 选择新区域的定点
            image = plt.imread(image_path)
            cv2.polylines(image, np.int32([old_points]),True,(0,0,0),2)        # 旧区域-黑线条
            cv2.polylines(image, np.int32([new_points]),True,(255,255,255),2)  # 新区域-白线条
            plt.imshow(image,cmap="gray")
            self.old_points = old_points
            self.new_points = new_points

    
      def mouse_left_click(self, event, x, y, flag, param):   # 点击鼠标左键
            global left_click_X, left_click_Y 
            if event == cv2.EVENT_LBUTTONDOWN:
                  left_click_X, left_click_Y = x, y   

      def get_point_array(self, x, y):   # 定义所有点对
            x = x[1:]          
            y = y[1:]          
            points = np.empty((len(x),2), np.int32)    # 记录点的坐标
            for i in range(len(x)):   
                points[i,0]= x[i]     # points[0,0] = x[0], points[0,1] = y[0]
                points[i,1]= y[i]   
            return(points)
    
      def polydraw(self, points, image,closed, b,g,r):   # 用线连接所有点对
            cv2.polylines(image, [points], closed,(b,g,r), 2)

      def select_region_points(self, path, mode, points=None):   # 选择区域的定点
            image = cv2.imread(path)
            if mode == True:
                  self.polydraw(points,image,True, b=0,g=0,r=0)
            cv2.namedWindow('image')
            cv2.setMouseCallback('image', self.mouse_left_click)
            x = [0]   # 存放x坐标的数组
            y = [0]   # 存放y坐标的数组
        
            cv2.imshow('image',image)
            while(1):
                  k = cv2.waitKey(1) & 0xFF
                  if k == ord('1'):
                        if x[len(x)-1] == left_click_X and y[len(y)-1] == left_click_Y:
                              x = x
                              y = y                     
                        else:
                              x.append(left_click_X)
                              y.append(left_click_Y)
                              
                        print (left_click_X, left_click_Y)  

                        if len(x)<3:
                              pass
                        else:
                              points = self.get_point_array(x,y)
                              self.polydraw(points,image, False,b=255,g=255,r=255)
                              cv2.imshow('image',image)
                              #cv2.waitKey(0)
                        if len(x)==5:
                              self.polydraw(np.array([(x[1],y[1]),(x[4],y[4])],np.int32),image,False,b=255,g=255,r=255)

                  elif k == 13:    # ENTER 键退出滑鼠左键设置
                      break
            points = self.get_point_array(x,y)
            cv2.destroyAllWindows()
            return(points)
    
      

#Perspective transform and Perspective Warping
def get_perspective_transform1(old_points,new_points):
    M = cv2.getPerspectiveTransform(old_points.astype(np.float32),new_points.astype(np.float32))
    return(M)

def get_perspective_transform(src,new_points):
      '''
      利用原图需要变换的物体的四个顶点坐标和变换后的四个顶点坐标求出变换矩阵warpMatrix
      A * A_matrix = B
      :param src: 原图需要变换物体的四个顶点
      :param new_points: 新图对应的四个顶点
      :return: A_matrix
      '''

      B = np.zeros((8, 1))
      for i in range(4):
            # 先做一个2*8的矩阵
            A1 = np.zeros((2, 8))

            A1[0, 0] = src[i,0]
            A1[0, 1] = src[i,1]
            A1[0, 2] = 1
            A1[0, 6] = -src[i, 0] * new_points[i, 0]
            A1[0, 7] = -src[i, 1] * new_points[i, 0]
            B[2*i] = new_points[i, 0]
            A1[1, 3] = src[i, 0]
            A1[1, 4] = src[i, 1]
            A1[1, 5] = 1
            A1[1, 6] = -src[i, 0] * new_points[i, 1]
            A1[1, 7] = -src[i, 1] * new_points[i, 1]
            B[2*i+1] = new_points[i, 1]

            if i == 0:
                  A = A1
            else:
                  A = np.vstack([A, A1]) # 连接四个2*8的矩阵
                  pass
            pass
      A = np.mat(A)
      # print(A)
      A_matrix = A.I*B
      # print(A_matrix)
      # 已求出a11,a12,a13,a21,a22,a23,a31,a32  再插入a33 = 1
      A_matrix = np.array(A_matrix).T[0]
      A_matrix = np.insert(A_matrix, A_matrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
      A_matrix = A_matrix.reshape((3, 3))
      return A_matrix
    
def perspective_warping(A_matrix,new_points,image):
      '''shape = image.shape[:2]
      print(shape)
      new_image = cv2.warpPerspective(image, Camera_matrix, (shape[1],shape[0]), flags= cv2.INTER_LINEAR)
      return(new_image)'''

      '''
      :param A_matrix:
      :param new_points:
      :return:
      '''
      w = abs(int(new_points[1, 0] - new_points[0, 0])) +1  # 337
      h = abs(int(new_points[2, 1] - new_points[0, 1])) +1  # 488

      # 用opencv处理图像时，可以发现获得的矩阵类型都是uint8 无符8位整型（0-255）, uint8是专门用于存储各种图像的（包括RGB，灰度图像等），范围是从0–255
      result = np.zeros((h, w, 3), np.uint8)

      # 数组与数组运算，矩阵与矩阵运算，最好统一数据类型，实测数组与矩阵相乘结果会不一样
      # W = np.linalg.inv(A_matrix)
      W = np.mat(A_matrix)
      W = W.I
      for i in range(h):
            for j in range(w):

                  XY1 = np.array([[j], [i], [1]])
                  # 关于j,i 的位置问题，由给定的dst坐标排列决定，由src和dst算出的变换矩阵已经定好位置，所以在反求原图位置的时候需注意
                  XY1 = np.mat(XY1)
                  # XY1 = XY1.T    一维数组无法进行转置
                  # print(XY1)
                  x, y, _ = W.dot(XY1)

                  # 验证算出来的坐标是否和src里的相对应
                  if i == h-1 and j == 0:
                        #print(x, y)
                        print('==')

                  # 算出的索引超出原图大小时，令其等于边界
                  if y >=960:
                        y = 959
                        pass
                  if y < 0:
                        y = 0
                        pass
                  if x >=540:
                        x = 539
                        pass
                  if x < 0:
                        x = 0
                        pass
                  # print(int(xy1[0]))
                  # try:
                  #     result[i, j, 0] = image[int(xy1[0]), int(xy1[1]), 0]
                  #     result[i, j, 1] = image[int(xy1[0]), int(xy1[1]), 1]
                  #     result[i, j, 2] = image[int(xy1[0]), int(xy1[1]), 2]
                  # except Exception as msg:
                  #     print(xy1[0], xy1[1])

                  result[i, j] = image[int(y), int(x)]

                  '''result[i, j, 0] = image[int(y), int(x), 0]
                  result[i, j, 1] = image[int(y), int(x), 1]
                  result[i, j, 2] = image[int(y), int(x), 2]'''


      return result


def main():
      image = cv2.imread("image/cat.jpg")
      grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      cv2.imwrite("image/gray-cat.jpg", grayimg)
      point_setting = set_oldpoints_newpoints("image/gray-cat.jpg")
      old_points = point_setting.old_points
      new_points = point_setting.new_points
      #Cam_mat1 = get_perspective_transform1(old_points,new_points)
      A_matrix = get_perspective_transform(old_points,new_points)
      print('A matrix:\n', A_matrix)

      image = plt.imread("image/gray-cat.jpg")
      result_image = perspective_warping(A_matrix,new_points,image)
      print('Warp matrix:\n',result_image)
      image, (x0,x1) = plt.subplots(1,2,figsize=(20,40))
      x0.imshow(image)
      x1.imshow(result_image)
      plt.show()


if __name__ == '__main__':
      main()