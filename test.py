import  cv2
print("Test Ctypes testctypesfunction")
from ctypes import *
from numpy.ctypeslib import ndpointer
# 回调函数类型 None 返回值void 参数 c_int
FUNT = CFUNCTYPE(None, c_int)




try:
    lib = CDLL("VST")

    # 传递的数据
    arr = [1, 3, 55, 11, 34]
    ArrType = c_int * len(arr)
    # Ctype的数组对象
    carr = ArrType(*arr)
    lib.VST.argtypes = (ArrType, c_int, c_int)
    lib.VST.restype = ndpointer(dtype=c_float, shape=(len(arr),))
    # print(arr)

    for i in range(1):
        VST_img=lib.VST(carr, len(arr), 1)
        print("VST_img",VST_img)

except Exception as ex:
    print("testctypes error", ex)

# 等待用户输入，程序不退出
# input()