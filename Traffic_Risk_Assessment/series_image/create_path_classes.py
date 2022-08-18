from __future__ import print_function, division
import os
import sys
import subprocess
import random
def class_process(dir_path, class_name, class_dict):
  sub_folder = os.listdir(dir_path + '/' + class_name)
  train_file = open("trainlist.txt","a")
  test_file = open("testlist.txt","a")  
  for x in sub_folder:
    seed = random.randint(0,4)
    seed = 1
    if seed < 5 :
      test_file.write(class_name + '/'+ x + '.avi ' + str(class_dict[class_name]))
      test_file.write("\n")
    else:
      train_file.write(class_name + '/'+ x + '.avi ' + str(class_dict[class_name]))
      train_file.write("\n")




if __name__=="__main__":
  dir_path = sys.argv[1]
  print(dir_path)
  class_dict = {}
  classes = os.listdir(dir_path)
  print(os.listdir(dir_path))
  for x,y in zip(range(6),classes):
    class_dict[y] = x 
  print(class_dict)
  for class_name in os.listdir(dir_path):
    class_process(dir_path, class_name, class_dict)
