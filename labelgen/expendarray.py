import  numpy as np
def expend(oldarray ,newwidth ,newheight): #参数介绍：原二维数组、新宽度、旧宽度

    height, width = oldarray.shape
    nwith = newwidth - width  # 所需扩展宽度数量
    nw1 = int(nwith / 2)  # 宽度做扩展数量
    # nw2 = nwith - nw1  # 右填充数量
    nheight = newheight - height  # 需扩展高度大小
    nh1 = int(nheight / 2)  # 上填充数量
    # nh2 = nheight - nh1  # 下填充数量

    newarray = np.zeros((newheight, newwidth))
    newarray[nh1:nh1+height,nw1:nw1+width] = oldarray[:,:]
    '''
 
    num1 = 0 #初始化计数参数
    NoData_value = 0
    # newarray = [],[]
    newarray = np.zeros((newheight, newwidth))
    for ii in range(newheight):
        for jj in range(newwidth):
            if((nh1 - 1< ii < nh1 + height) and (nw1 - 1 < jj <nw1 +width)):
                newarray[ii][jj] = oldarray[num1]
                num1 += 1   
    
    oldarray = oldarray.flatten() #二位数组变成一维数组
    '''

    #print(oldarray)


           # else:
           #     newarray[ii][jj] = NoData_value
    #newarray = newarray.reshape(newheight,newwidth)
    #newarray = newarray.astype(int)
    newarray = newarray.astype(np.uint8)
    #new = newarray
    return newarray
'''
a = [[1,2,2],[2,3,6]]
print(a)
a = np.array(a)
b = expend(a,10,8)
print(b)
'''