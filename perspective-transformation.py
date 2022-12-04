import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
新旧区域各提取4个定点坐标。
每用鼠标点击一次图像中的坐标则按一次ENTER键将该坐标提取出来。
旧区域由黑线条包围；新区域由白线条包围。
'''

class set_oldpoints_newpoints():    # 提取图像的旧定点坐标（旧区域）和新定点坐标（新区域），用于作透视变换

      def __init__(self, image_path):
            self.output_st = ""
            old_points = self.select_region_points(image_path,mode=False)    # 选择旧区域的定点
            new_points = self.select_region_points(image_path,mode=True,points=old_points) # 选择新区域的定点
            image = plt.imread(image_path)
            cv2.polylines(image,np.int32([old_points]),True,(0,0,0),2)        # 旧区域-黑线条
            cv2.polylines(image,np.int32([new_points]),True,(255,255,255),2)  # 新区域-白线条
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
                points[i,0]= x[i]    
                points[i,1]= y[i]   
            return(points)
    
      def drawlines(self, points, image, closed, b,g,r):   # 用线连接所有点对
            cv2.polylines(image, [points], closed, (b,g,r), 2)

      def select_region_points(self, path, mode, points=None):   # 选择区域的定点
            image = cv2.imread(path)
            if mode == True:
                  self.output_st += "New Points:\n"
                  print("\nNew Points:")
                  self.drawlines(points,image,True, b=0,g=0,r=0)
            else:
                  self.output_st += "Old Points:\n"
                  print("Old Points:")
            cv2.namedWindow('Perspective Transformation')
            cv2.setMouseCallback('Perspective Transformation', self.mouse_left_click)
            x = [0]   # 存放x坐标的数组
            y = [0]   # 存放y坐标的数组
        
            cv2.imshow('Perspective Transformation',image)
            while(1):
                  k = cv2.waitKey(1) & 0xFF
                  if k == 13:      # ENTER 键来提取坐标
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
                              self.drawlines(points,image, False,b=255,g=255,r=255)
                              cv2.imshow('Perspective Transformation',image)
                        if len(x)==5:
                              self.drawlines(np.array([(x[1],y[1]),(x[4],y[4])],np.int32),image,False,b=255,g=255,r=255)
                              break

            points = self.get_point_array(x,y)
            self.output_st += str(points) + '\n\n'
            cv2.destroyAllWindows()
            return(points)


def perspectiveTransform(old_points,new_points):   # 透视变换
      '''
      利用原图需要变换的物体的四个顶点坐标和变换后的四个顶点坐标求出变换矩阵 warp_matrix
       A * warp_matrix = B
      :param old_points: 原图需要变换物体的四个顶点
      :param new_points: 新图对应的四个顶点
      :return: warp_matrix
      '''
      B = np.zeros((8, 1))
      A = np.zeros((8, 8))
      for i in range(4):
            k = 2*i
            A[k,0] = old_points[i,0]
            A[k,1] = old_points[i,1]
            A[k,2] = 1
            A[k,6] = -old_points[i,0] * new_points[i,0]
            A[k,7] = -old_points[i,1] * new_points[i,0]
            B[k] = new_points[i,0]
            A[k+1,3] = old_points[i,0]
            A[k+1,4] = old_points[i,1]
            A[k+1,5] = 1
            A[k+1,6] = -old_points[i,0] * new_points[i,1]
            A[k+1,7] = -old_points[i,1] * new_points[i,1]
            B[k+1] = new_points[i,1]

      np.savetxt('output/A_matrix.txt',A,fmt='%s')
      A = np.mat(A)
      warp_matrix = A.I*B    # 8×1 矩阵：a11,a12,a13,a21,a22,a23,a31,a32
      warp_matrix = np.array(warp_matrix).T[0]    
      warp_matrix = np.insert(warp_matrix, warp_matrix.shape[0], values=1.0, axis=0)  # 插入 a_33 = 1
      warp_matrix = warp_matrix.reshape((3, 3))
      return warp_matrix


def warpPerspective(warp_matrix, input_image):   # 透视扭曲
      
      # 将图像转换为矩阵
      h,w = input_image.shape
      input_matrix = np.zeros((w,h), dtype='int')
      for i in range(input_image.shape[0]):
            input_matrix[:,i] = input_image[i]

      w,h = (input_image.shape[:2][1],input_image.shape[:2][0])
      output_matrix = np.zeros((w,h))
      for i in range(input_matrix.shape[0]):
            for j in range(input_matrix.shape[1]):
                  mul1, mul2, mul3 = np.dot(warp_matrix, [i,j,1])    # 变换矩阵*[x,y,1]=[X,Y,1]
                  i_temp = (mul1/mul3 + 0.5).astype(int)
                  j_temp = (mul2/mul3 + 0.5).astype(int)
                  if i_temp >= 0 and i_temp < w:
                        if j_temp >= 0 and j_temp < h:
                              output_matrix[i_temp,j_temp] = input_matrix[i,j]

      # 将矩阵转换为图像
      w,h = output_matrix.shape
      output_image = np.zeros((h,w), dtype='int')
      for i in range(output_matrix.shape[0]):
            output_image[:,i] = output_matrix[i]

      return output_image


def main():
      image = cv2.imread("image/cat.jpg")
      grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      cv2.imwrite("image/gray-cat.jpg", grayimg)
      
      point_setting = set_oldpoints_newpoints("image/gray-cat.jpg")
      old_points = point_setting.old_points
      new_points = point_setting.new_points
      warp_matrix = perspectiveTransform(old_points,new_points)
      image = plt.imread("image/gray-cat.jpg")
      result_image = warpPerspective(warp_matrix,image)

      # 输出矩阵
      st = '\nWarp Matrix:\n' + str(warp_matrix) + '\n\nResult Image:\n'  + str(result_image) 
      print(st)
      point_setting.output_st += st
      outfile = open('output/result.txt','w')
      outfile.write(point_setting.output_st)
      np.savetxt('output/resultImage.txt',result_image,fmt='%s')

      image, (x0,x1) = plt.subplots(1,2,figsize=(20,40))
      x0.imshow(grayimg)
      x1.imshow(result_image)
      plt.show()


if __name__ == '__main__':
      main()