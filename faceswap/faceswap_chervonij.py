# This is taken from https://colab.research.google.com/github/chervonij/DFL-Colab/blob/master/DFL_Colab_Demo.ipynb#scrollTo=JG-f2WqT4fLK


import sys

#!git clone https://github.com/iperov/DeepFaceLab.git


#!pip install -r /content/DeepFaceLab/requirements-colab.txt
#!pip install --upgrade scikit-image

#!wget -q --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1hTH2h6l_4kKrczA8EkN6GyuXx4lzmCnK' -O pretrain_CelebA.zip
#!mkdir /content/pretrain
#!unzip -q /content/pretrain_CelebA.zip -d /content/pretrain/
#!rm /content/pretrain_CelebA.zip

#######################
#  Celeb Images are 256 X 256 Size
#  Number of Images 24710
######################


if not Path("/content/workspace").exists():
  sys.exit(-1)
 

Mode = "workspace" #@param ["workspace", "data_src", "data_dst", "data_src aligned", "data_dst aligned", "models"]
Archive_name = "workspace.zip" #@param {type:"string"}

#Mount Google Drive as folder
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

def zip_and_copy(path, mode):
  unzip_cmd=" -q "+Archive_name
  
  %cd $path
  copy_cmd = "/content/drive/My\ Drive/"+Archive_name+" "+path
  !cp $copy_cmd
  !unzip $unzip_cmd    
  !rm $Archive_name

if Mode == "workspace":
  zip_and_copy("/content", "workspace")
elif Mode == "data_src":
  zip_and_copy("/content/workspace", "data_src")
elif Mode == "data_dst":
  zip_and_copy("/content/workspace", "data_dst")
elif Mode == "data_src aligned":
  zip_and_copy("/content/workspace/data_src", "aligned")
elif Mode == "data_dst aligned":
  zip_and_copy("/content/workspace/data_dst", "aligned")
elif Mode == "models":
  zip_and_copy("/content/workspace", "model")
  



print("Done!")
