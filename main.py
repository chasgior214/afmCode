from AFMImageCollection import AFMImageCollection
from datetime import timedelta
import matplotlib.pyplot as plt

folder_path = "D:/afmCode/DataFolder"
collection = AFMImageCollection(folder_path)
num_images = collection.get_number_of_images()
print("=================================================")
print(f"Number of images in the array: {num_images}")
print("=================================================")

collection.review_phase()


collection.export_deflection_time(3)
collection.export_shift(3)

t = collection.get_image(0).get_datetime()

T = [] 
max_height = [] 


for i in range(num_images):
    
    x,y,max_value, shift = collection.get_image(i).get_trimmed_trace_z(3)
    timediff = collection.get_image(i).get_datetime() - t 
    T.append(timediff.total_seconds()/3600)
    max_height.append(max_value)
    
x,y,max_value, shift = collection.get_image(0).get_trimmed_trace_z(3)


plt.scatter(T,max_height)
plt.show()

