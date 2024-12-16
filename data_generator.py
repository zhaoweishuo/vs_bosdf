import time
import point_generate
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas

client = RemoteAPIClient()
sim = client.require('sim')

points = point_generate.generate_cube_point(side_length=0.12, num_samples=5)

target_handle = sim.getObject('/AUBO_i5_Base/target')
target_position = sim.getObjectPosition(target_handle)
target_orientation = sim.getObjectOrientation(target_handle)

sensor_handle = sim.getObject('/AUBO_i5_Base/Vision_sensor')

count = 1
for i in points:

    new_position = [target_position[0] + i[0], target_position[1] + i[1], target_position[2] + i[2]]
    new_orientation = [target_orientation[0] + i[3], target_orientation[1] + i[4], target_orientation[2] + i[5]]

    buildmatrix = sim.buildMatrix(new_position, new_orientation)
    sim.setObjectMatrix(target_handle, buildmatrix)

    print("new", new_position, new_orientation)
    time.sleep(2)

    """
    sim.setObjectPosition(target_handle, new_position)
    sim.setObjectOrientation(target_handle, new_orientation)
    time.sleep(0.1)  
    """


    # image, resolution = sim.getVisionSensorImg(sensor_handle)
    # image = np.frombuffer(image, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
    # image = np.flipud(image)
    # image = Image.fromarray(image)
    #
    # # image.show()
    #
    # # 保存图像采集信息
    # image.save("./dataset/5000/Image/"+str(count)+".png")
    #

    # label = {
    #     "name": str(count)+".png",
    #     "x": new_position[0],
    #     "y": new_position[1],
    #     "z": new_position[2],
    #     "a": new_orientation[0],
    #     "b": new_orientation[1],
    #     "g": new_orientation[2]
    # }
    # data = pandas.DataFrame(data=label, index=[0])  #
    # data.to_csv("./dataset/5000/label.csv", mode='a', index=False, header=False)  # 保存数据
    #

    # print("Image Name: {}".format(str(count)+".png"))
    # count += 1


sim.setObjectPosition(target_handle, target_position)
sim.setObjectOrientation(target_handle, target_orientation)

