import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
import pickle
import time
import pandas as pd

time_start = time.time()

file1 = open('test.txt','rb')
results_dict = pickle.load(file1)
file2 = open('input.txt','rb')
input_dict = pickle.load(file2)
data_list = list(results_dict.keys())
stored_data = list(results_dict.values())
image_input_path = list(input_dict.keys())
input_data = list(input_dict.values())


feature_input = list(input_dict.values())
feature_stored = list(results_dict.values())
i = len(data_list) # number of stored images
j = len(image_input_path) # number of input images
locations_input = []
locations_stored = []
descriptors_input = []
descriptors_stored = []
for k in range(j):
    locations_input.append(feature_input[k][0])
    descriptors_input.append(feature_input[k][1])
for index in range(i):
    locations_stored.append(feature_stored[index][0])
    descriptors_stored.append(feature_stored[index][1])

# print(locations_input[0])
# print(descriptors_input[0])

print(descriptors_input[0].shape)

# location = (feature[0][0]).tolist()
# descriptors = (feature[1][1]).tolist()
# print(location)
# print(descriptors)

# print(input_dict)
# a = pd.DataFrame(input_dict)
# print(a)
# b = pd.DataFrame(results_dict)
# print(b)
# # 测试value list 和 data list 当中的index 数据是否相符合
# print(len(image_input_path))
# a = input_dict[image_input_path[2]]
# b = input_data[2]
# print(a==b)

# # 测试input_data 的维度和取值
# locations_1, descriptors_1 = input_data[0]#input_dict[test_path]
# num_features_1 = locations_1.shape[0]
# # print(locations_1)
# # print(descriptors_1)
# # print(num_features_1)
# locations_2, descriptors_2 = stored_data[0]
# num_features_2 = locations_2.shape[0]



#@title TensorFlow is not needed for this post-processing and visualization
def match_images(results_dict, input_dict, image_1_path, image_2_path):
  distance_threshold = 0.8
  # Read features.
  locations_1, descriptors_1 = input_dict[image_1_path]
  num_features_1 = locations_1.shape[0]
  # print("Loaded image 1's %d features" % num_features_1)
  locations_2, descriptors_2 = results_dict[image_2_path]
  num_features_2 = locations_2.shape[0]
  # print("Loaded",image_2_path,"'s %dfeatures" % (num_features_2))

  # Find nearest-neighbor matches using a KD tree.
  d1_tree = cKDTree(descriptors_1)
  _, indices = d1_tree.query(
      descriptors_2, distance_upper_bound=distance_threshold)

  # Select feature locations for putative matches.
  locations_2_to_use = np.array([
      locations_2[i,]
      for i in range(num_features_2)
      if indices[i] != num_features_1
  ])
  locations_1_to_use = np.array([
      locations_1[indices[i],]
      for i in range(num_features_2)
      if indices[i] != num_features_1
  ])

  # print(locations_1_to_use)

  # Perform geometric verification using RANSAC.
  # Try to add a try function to solve errors
  if locations_1_to_use.shape[0]!=0:
      _, inliers = ransac(
          (locations_1_to_use, locations_2_to_use),
          AffineTransform,
          min_samples=3,
          residual_threshold=20,
          max_trials=1000)
      if inliers is None:
          return 0, locations_1_to_use, locations_2_to_use, inliers
      else:
          return sum(inliers), locations_1_to_use, locations_2_to_use, inliers
  else:
      return 0, locations_1_to_use, locations_2_to_use, None
  # print('Found %d inliers' % sum(inliers))
  # return sum(inliers)

index = []
for i,test_path in enumerate(image_input_path): # test input 也可以用rdd 尝试 然后和后面的rdd 重叠起来
    # print(i,test_path)
    matches = []
    test_feature = []
    data_feature = []
    inliers = []
    for data_path in data_list: # here to apply rdd may overlap rdd
        match_num, locations_1_to_use, locations_2_to_use, inlier = match_images(results_dict, input_dict, test_path, data_path)
        # print('has',match,'matches with',data_path)
        matches.append(match_num)
        test_feature.append(locations_1_to_use)
        data_feature.append(locations_2_to_use)
        inliers.append(inlier)
    # index[i] = matches.index(max(matches))
    index.append(matches.index(max(matches)))
    # print(test_path,matches[index[i]],data_list[index[i]])
    # print(inliers[index[i]])

    _, ax = plt.subplots()
    img_1 = mpimg.imread(test_path)
    img_2 = mpimg.imread(data_list[index[i]])
    inlier_idxs = np.nonzero(inliers[index[i]])[0]
    plot_matches(
      ax,
      img_1,
      img_2,
      test_feature[index[i]],
      data_feature[index[i]],
      np.column_stack((inlier_idxs, inlier_idxs)),
      matches_color='b')
    ax.axis('off')
    ax.set_title('DELF correspondences')
    test_name = test_path.replace('\\','').split('.')[0]
    data_name = data_list[index[i]].replace('\\','').split('.')[0]
    match_index = str(matches[index[i]])
    plt.savefig(test_name+'_'+data_name+'_'+'match'+match_index+'.jpg')
    # plt.show()

time_end = time.time()
print('time cost',time_end-time_start,'s')

