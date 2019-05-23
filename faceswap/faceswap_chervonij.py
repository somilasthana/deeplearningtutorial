# This is taken from https://colab.research.google.com/github/chervonij/DFL-Colab/blob/master/DFL_Colab_Demo.ipynb#scrollTo=JG-f2WqT4fLK

"""
Parameter Details:

Number of Dest Images > 2000

Step 3: Extracting Images From Videos Both Source and Destination
  Need FPS : Frames per Second
  Output Image format: png or jpeg ( png are bigger size )
  Cut video: drop video
  
Step 4: Denoising 
  Factor : The degree of noise suppression ( 5 )
  Removing Noise but maintaining crisp edges
  
Step 5: Extract Faces
  Algo used 
  MT
  S3FD
  
Step 6: Sorting using Histogram
  After sorting this person will be grouped by content basically a technique used to weed out unwanted people.
  
Result of Extraction 
-- Muddy person will be removed
-- Person closing something will also be removed


To make results better in dest pereizvlech on frames.

Observations
1. Narrow faces train better than Broad Faces
2. In cases where dest are in one light e.g shadow nose different from src light then NN doesnt do well.

"""

import sys
# STEP 1
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
 
# STEP 2
Mode = "workspace" #@param ["workspace", "data_src", "data_dst", "data_src aligned", "data_dst aligned", "models"]
Archive_name = "workspace.zip" #@param {type:"string"}

#Mount Google Drive as folder
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

def zip_and_copy(path, mode):
  unzip_cmd=" -q "+Archive_name
  
  
# %cd $path
# copy_cmd = "/content/drive/My\ Drive/"+Archive_name+" "+path
# !cp $copy_cmd
# !unzip $unzip_cmd    
# !rm $Archive_name

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
  
## STEP 3: @title Extract frames
Video = "data_src" #@param ["data_src", "data_dst"]

# %cd "/content"

cmd = "DeepFaceLab/main.py videoed extract-video"

if Video == "data_dst":
  cmd+= " --input-file workspace/data_dst.* --output-dir workspace/data_dst/"
else:
  cmd+= " --input-file workspace/data_src.* --output-dir workspace/data_src/"
  
#!python $cmd
####################
# Prompts for FPS : 0
#              Output image format : png
####################


## STEP 4: @title Denoise frames

Data = "data_src" #@param ["data_src", "data_dst"]
Factor = 1 #@param {type:"slider", min:1, max:20, step:1}

cmd = "DeepFaceLab/main.py videoed denoise-image-sequence --input-dir workspace/"+Data+" --factor "+str(Factor)

#%cd "/content"
#!python $cmd


#STEP 5: @title Detect faces
Data = "data_src" #@param ["data_src", "data_dst"]
Detector = "S3FD" #@param ["S3FD", "MT"]
Debug = False #@param {type:"boolean"}

detect_type = "s3fd"
if Detector == "S3FD":
  detect_type = "s3fd"
elif Detector == "MT":
  detect_type = "mt"

folder = "workspace/"+Data
folder_align = folder+"/aligned"
debug_folder = folder_align+"/debug"

cmd = "DeepFaceLab/main.py extract --input-dir "+folder+" --output-dir "+folder_align

if Debug:
  cmd+= " --debug-dir "+debug_folder

cmd+=" --detector "+detect_type
  
#%cd "/content"
#!python $cmd

#Step 6: @title Sort aligned
Data = "data_src" #@param ["data_src", "data_dst"]
sort_type = "hist" #@param ["hist", "hist-dissim", "face-yaw", "face-pitch", "blur", "final"]

cmd = "DeepFaceLab/main.py sort --input-dir workspace/"+Data+"/aligned --by "+sort_type

#%cd "/content"
#!python $cmd


print("Done!")


#Step 7: @title Training
Model = "SAE" #@param ["SAE", "H128", "LIAEF128", "DF", "DEV_FANSEG", "RecycleGAN"]
Backup_every_hour = True #@param {type:"boolean"}

#%cd "/content"

#Mount Google Drive as folder
from google.colab import drive
drive.mount('/content/drive')

import psutil, os, time

p = psutil.Process(os.getpid())
uptime = time.time() - p.create_time()

if (Backup_every_hour):
  if not os.path.exists('workspace.zip'):
    print("Creating workspace archive ...")
    !zip -r -q workspace.zip workspace
    print("Archive created!")
  else:
    print("Archive exist!")

if (Backup_every_hour):
  print("Time to end session: "+str(round((43200-uptime)/3600))+" hours")
  backup_time = str(3600)
  backup_cmd = " --execute-program -"+backup_time+" \"import os; os.system('zip -r -q workspace.zip workspace/model'); os.system('cp /content/workspace.zip /content/drive/My\ Drive/'); print(' Backuped!') \"" 
elif (round(39600-uptime) > 0):
  print("Time to backup: "+str(round((39600-uptime)/3600))+" hours")
  backup_time = str(round(39600-uptime))
  backup_cmd = " --execute-program "+backup_time+" \"import os; os.system('zip -r -q workspace.zip workspace'); os.system('cp /content/workspace.zip /content/drive/My\ Drive/'); print(' Backuped!') \"" 
else:
  print("Session expires in less than an hour.")
  backup_cmd = ""
    
cmd = "DeepFaceLab/main.py train --training-data-src-dir workspace/data_src/aligned --training-data-dst-dir workspace/data_dst/aligned --pretraining-data-dir pretrain/aligned --model-dir workspace/model --model "+Model
  
if (backup_cmd != ""):
  train_cmd = (cmd+backup_cmd)
else:
  train_cmd = (cmd)

#!python $train_cmd


"""
Step 1

Cloning into 'DeepFaceLab'...
remote: Enumerating objects: 9, done.
remote: Counting objects: 100% (9/9), done.
remote: Compressing objects: 100% (7/7), done.
remote: Total 3214 (delta 4), reused 7 (delta 2), pack-reused 3205
Receiving objects: 100% (3214/3214), 298.47 MiB | 37.11 MiB/s, done.
Resolving deltas: 100% (2069/2069), done.
Checking out files: 100% (122/122), done.
Collecting git+https://www.github.com/keras-team/keras-contrib.git (from -r /content/DeepFaceLab/requirements-colab.txt (line 10))
  Cloning https://www.github.com/keras-team/keras-contrib.git to /tmp/pip-req-build-q6rntvxy
  Running command git clone -q https://www.github.com/keras-team/keras-contrib.git /tmp/pip-req-build-q6rntvxy
Requirement already satisfied: numpy==1.16.3 in /usr/local/lib/python3.6/dist-packages (from -r /content/DeepFaceLab/requirements-colab.txt (line 1)) (1.16.3)
Collecting h5py==2.9.0 (from -r /content/DeepFaceLab/requirements-colab.txt (line 2))
  Downloading https://files.pythonhosted.org/packages/30/99/d7d4fbf2d02bb30fb76179911a250074b55b852d34e98dd452a9f394ac06/h5py-2.9.0-cp36-cp36m-manylinux1_x86_64.whl (2.8MB)
     |████████████████████████████████| 2.8MB 8.9MB/s 
Requirement already satisfied: Keras==2.2.4 in /usr/local/lib/python3.6/dist-packages (from -r /content/DeepFaceLab/requirements-colab.txt (line 3)) (2.2.4)
Collecting opencv-python==4.0.0.21 (from -r /content/DeepFaceLab/requirements-colab.txt (line 4))
  Downloading https://files.pythonhosted.org/packages/37/49/874d119948a5a084a7ebe98308214098ef3471d76ab74200f9800efeef15/opencv_python-4.0.0.21-cp36-cp36m-manylinux1_x86_64.whl (25.4MB)
     |████████████████████████████████| 25.4MB 47.7MB/s 
Collecting tensorflow-gpu==1.13.1 (from -r /content/DeepFaceLab/requirements-colab.txt (line 5))
  Downloading https://files.pythonhosted.org/packages/7b/b1/0ad4ae02e17ddd62109cd54c291e311c4b5fd09b4d0678d3d6ce4159b0f0/tensorflow_gpu-1.13.1-cp36-cp36m-manylinux1_x86_64.whl (345.2MB)
     |████████████████████████████████| 345.2MB 53kB/s 
Collecting plaidml-keras==0.5.0 (from -r /content/DeepFaceLab/requirements-colab.txt (line 6))
  Downloading https://files.pythonhosted.org/packages/17/34/4102261e3d8867c31bae9f4def5d7e700fc25fff232fa1780040e8ed79b0/plaidml_keras-0.5.0-py2.py3-none-any.whl
Requirement already satisfied: scikit-image in /usr/local/lib/python3.6/dist-packages (from -r /content/DeepFaceLab/requirements-colab.txt (line 7)) (0.14.2)
Requirement already satisfied: tqdm in /usr/local/lib/python3.6/dist-packages (from -r /content/DeepFaceLab/requirements-colab.txt (line 8)) (4.28.1)
Collecting ffmpeg-python==0.1.17 (from -r /content/DeepFaceLab/requirements-colab.txt (line 9))
  Downloading https://files.pythonhosted.org/packages/3d/10/330cbc8e63d072d40413f4d470444a6a1e8c8c6a80b2a4ac302d1252ca1b/ffmpeg_python-0.1.17-py3-none-any.whl
Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from h5py==2.9.0->-r /content/DeepFaceLab/requirements-colab.txt (line 2)) (1.12.0)
Requirement already satisfied: pyyaml in /usr/local/lib/python3.6/dist-packages (from Keras==2.2.4->-r /content/DeepFaceLab/requirements-colab.txt (line 3)) (3.13)
Requirement already satisfied: keras-applications>=1.0.6 in /usr/local/lib/python3.6/dist-packages (from Keras==2.2.4->-r /content/DeepFaceLab/requirements-colab.txt (line 3)) (1.0.7)
Requirement already satisfied: keras-preprocessing>=1.0.5 in /usr/local/lib/python3.6/dist-packages (from Keras==2.2.4->-r /content/DeepFaceLab/requirements-colab.txt (line 3)) (1.0.9)
Requirement already satisfied: scipy>=0.14 in /usr/local/lib/python3.6/dist-packages (from Keras==2.2.4->-r /content/DeepFaceLab/requirements-colab.txt (line 3)) (1.2.1)
Requirement already satisfied: wheel>=0.26 in /usr/local/lib/python3.6/dist-packages (from tensorflow-gpu==1.13.1->-r /content/DeepFaceLab/requirements-colab.txt (line 5)) (0.33.4)
Requirement already satisfied: tensorboard<1.14.0,>=1.13.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow-gpu==1.13.1->-r /content/DeepFaceLab/requirements-colab.txt (line 5)) (1.13.1)
Requirement already satisfied: absl-py>=0.1.6 in /usr/local/lib/python3.6/dist-packages (from tensorflow-gpu==1.13.1->-r /content/DeepFaceLab/requirements-colab.txt (line 5)) (0.7.1)
Requirement already satisfied: protobuf>=3.6.1 in /usr/local/lib/python3.6/dist-packages (from tensorflow-gpu==1.13.1->-r /content/DeepFaceLab/requirements-colab.txt (line 5)) (3.7.1)
Requirement already satisfied: tensorflow-estimator<1.14.0rc0,>=1.13.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow-gpu==1.13.1->-r /content/DeepFaceLab/requirements-colab.txt (line 5)) (1.13.0)
Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow-gpu==1.13.1->-r /content/DeepFaceLab/requirements-colab.txt (line 5)) (1.1.0)
Requirement already satisfied: grpcio>=1.8.6 in /usr/local/lib/python3.6/dist-packages (from tensorflow-gpu==1.13.1->-r /content/DeepFaceLab/requirements-colab.txt (line 5)) (1.15.0)
Requirement already satisfied: gast>=0.2.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow-gpu==1.13.1->-r /content/DeepFaceLab/requirements-colab.txt (line 5)) (0.2.2)
Requirement already satisfied: astor>=0.6.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow-gpu==1.13.1->-r /content/DeepFaceLab/requirements-colab.txt (line 5)) (0.7.1)
Collecting plaidml (from plaidml-keras==0.5.0->-r /content/DeepFaceLab/requirements-colab.txt (line 6))
  Downloading https://files.pythonhosted.org/packages/48/b7/5c019a8c64e92fc69c77f077e9a1bc6ea7883eb8fd79c639082c7984bfb5/plaidml-0.5.0-py2.py3-none-manylinux1_x86_64.whl (15.0MB)
     |████████████████████████████████| 15.0MB 31.7MB/s 
Requirement already satisfied: matplotlib>=2.0.0 in /usr/local/lib/python3.6/dist-packages (from scikit-image->-r /content/DeepFaceLab/requirements-colab.txt (line 7)) (3.0.3)
Requirement already satisfied: PyWavelets>=0.4.0 in /usr/local/lib/python3.6/dist-packages (from scikit-image->-r /content/DeepFaceLab/requirements-colab.txt (line 7)) (1.0.3)
Requirement already satisfied: cloudpickle>=0.2.1 in /usr/local/lib/python3.6/dist-packages (from scikit-image->-r /content/DeepFaceLab/requirements-colab.txt (line 7)) (0.6.1)
Requirement already satisfied: networkx>=1.8 in /usr/local/lib/python3.6/dist-packages (from scikit-image->-r /content/DeepFaceLab/requirements-colab.txt (line 7)) (2.3)
Requirement already satisfied: dask[array]>=1.0.0 in /usr/local/lib/python3.6/dist-packages (from scikit-image->-r /content/DeepFaceLab/requirements-colab.txt (line 7)) (1.1.5)
Requirement already satisfied: pillow>=4.3.0 in /usr/local/lib/python3.6/dist-packages (from scikit-image->-r /content/DeepFaceLab/requirements-colab.txt (line 7)) (4.3.0)
Requirement already satisfied: future in /usr/local/lib/python3.6/dist-packages (from ffmpeg-python==0.1.17->-r /content/DeepFaceLab/requirements-colab.txt (line 9)) (0.16.0)
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.6/dist-packages (from tensorboard<1.14.0,>=1.13.0->tensorflow-gpu==1.13.1->-r /content/DeepFaceLab/requirements-colab.txt (line 5)) (3.1)
Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.6/dist-packages (from tensorboard<1.14.0,>=1.13.0->tensorflow-gpu==1.13.1->-r /content/DeepFaceLab/requirements-colab.txt (line 5)) (0.15.3)
Requirement already satisfied: setuptools in /usr/local/lib/python3.6/dist-packages (from protobuf>=3.6.1->tensorflow-gpu==1.13.1->-r /content/DeepFaceLab/requirements-colab.txt (line 5)) (41.0.1)
Requirement already satisfied: mock>=2.0.0 in /usr/local/lib/python3.6/dist-packages (from tensorflow-estimator<1.14.0rc0,>=1.13.0->tensorflow-gpu==1.13.1->-r /content/DeepFaceLab/requirements-colab.txt (line 5)) (3.0.5)
Requirement already satisfied: enum34>=1.1.6 in /usr/local/lib/python3.6/dist-packages (from plaidml->plaidml-keras==0.5.0->-r /content/DeepFaceLab/requirements-colab.txt (line 6)) (1.1.6)
Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=2.0.0->scikit-image->-r /content/DeepFaceLab/requirements-colab.txt (line 7)) (2.4.0)
Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=2.0.0->scikit-image->-r /content/DeepFaceLab/requirements-colab.txt (line 7)) (2.5.3)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=2.0.0->scikit-image->-r /content/DeepFaceLab/requirements-colab.txt (line 7)) (1.1.0)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=2.0.0->scikit-image->-r /content/DeepFaceLab/requirements-colab.txt (line 7)) (0.10.0)
Requirement already satisfied: decorator>=4.3.0 in /usr/local/lib/python3.6/dist-packages (from networkx>=1.8->scikit-image->-r /content/DeepFaceLab/requirements-colab.txt (line 7)) (4.4.0)
Requirement already satisfied: toolz>=0.7.3; extra == "array" in /usr/local/lib/python3.6/dist-packages (from dask[array]>=1.0.0->scikit-image->-r /content/DeepFaceLab/requirements-colab.txt (line 7)) (0.9.0)
Requirement already satisfied: olefile in /usr/local/lib/python3.6/dist-packages (from pillow>=4.3.0->scikit-image->-r /content/DeepFaceLab/requirements-colab.txt (line 7)) (0.46)
Building wheels for collected packages: keras-contrib
  Building wheel for keras-contrib (setup.py) ... done
  Stored in directory: /tmp/pip-ephem-wheel-cache-mjz0u18j/wheels/11/27/c8/4ed56de7b55f4f61244e2dc6ef3cdbaff2692527a2ce6502ba
Successfully built keras-contrib
ERROR: albumentations 0.1.12 has requirement imgaug<0.2.7,>=0.2.5, but you'll have imgaug 0.2.9 which is incompatible.
Installing collected packages: h5py, opencv-python, tensorflow-gpu, plaidml, plaidml-keras, ffmpeg-python, keras-contrib
  Found existing installation: h5py 2.8.0
    Uninstalling h5py-2.8.0:
      Successfully uninstalled h5py-2.8.0
  Found existing installation: opencv-python 3.4.5.20
    Uninstalling opencv-python-3.4.5.20:
      Successfully uninstalled opencv-python-3.4.5.20
Successfully installed ffmpeg-python-0.1.17 h5py-2.9.0 keras-contrib-2.0.8 opencv-python-4.0.0.21 plaidml-0.5.0 plaidml-keras-0.5.0 tensorflow-gpu-1.13.1
Collecting scikit-image
  Downloading https://files.pythonhosted.org/packages/d4/ab/674e168bf7d0bc597218b3bec858d02c23fbac9ec1fec9cad878c6cee95f/scikit_image-0.15.0-cp36-cp36m-manylinux1_x86_64.whl (26.3MB)
     |████████████████████████████████| 26.3MB 1.5MB/s 
Requirement already satisfied, skipping upgrade: networkx>=2.0 in /usr/local/lib/python3.6/dist-packages (from scikit-image) (2.3)
Requirement already satisfied, skipping upgrade: PyWavelets>=0.4.0 in /usr/local/lib/python3.6/dist-packages (from scikit-image) (1.0.3)
Requirement already satisfied, skipping upgrade: matplotlib!=3.0.0,>=2.0.0 in /usr/local/lib/python3.6/dist-packages (from scikit-image) (3.0.3)
Requirement already satisfied, skipping upgrade: pillow>=4.3.0 in /usr/local/lib/python3.6/dist-packages (from scikit-image) (4.3.0)
Requirement already satisfied, skipping upgrade: imageio>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from scikit-image) (2.4.1)
Requirement already satisfied, skipping upgrade: scipy>=0.17.0 in /usr/local/lib/python3.6/dist-packages (from scikit-image) (1.2.1)
Requirement already satisfied, skipping upgrade: decorator>=4.3.0 in /usr/local/lib/python3.6/dist-packages (from networkx>=2.0->scikit-image) (4.4.0)
Requirement already satisfied, skipping upgrade: numpy>=1.9.1 in /usr/local/lib/python3.6/dist-packages (from PyWavelets>=0.4.0->scikit-image) (1.16.3)
Requirement already satisfied, skipping upgrade: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image) (1.1.0)
Requirement already satisfied, skipping upgrade: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image) (0.10.0)
Requirement already satisfied, skipping upgrade: python-dateutil>=2.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image) (2.5.3)
Requirement already satisfied, skipping upgrade: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib!=3.0.0,>=2.0.0->scikit-image) (2.4.0)
Requirement already satisfied, skipping upgrade: olefile in /usr/local/lib/python3.6/dist-packages (from pillow>=4.3.0->scikit-image) (0.46)
Requirement already satisfied, skipping upgrade: setuptools in /usr/local/lib/python3.6/dist-packages (from kiwisolver>=1.0.1->matplotlib!=3.0.0,>=2.0.0->scikit-image) (41.0.1)
Requirement already satisfied, skipping upgrade: six in /usr/local/lib/python3.6/dist-packages (from cycler>=0.10->matplotlib!=3.0.0,>=2.0.0->scikit-image) (1.12.0)
ERROR: albumentations 0.1.12 has requirement imgaug<0.2.7,>=0.2.5, but you'll have imgaug 0.2.9 which is incompatible.
Installing collected packages: scikit-image
  Found existing installation: scikit-image 0.14.2
    Uninstalling scikit-image-0.14.2:
      Successfully uninstalled scikit-image-0.14.2
Successfully installed scikit-image-0.15.0
warning [/content/pretrain_CelebA.zip]:  456533 extra bytes at beginning or within zipfile
  (attempting to process anyway)
Done!


Step 2
Mounted at /content/drive
/content
Done!


Step 3
/content
Enter FPS ( ?:help skip:fullfps ) : 0
Output image format? ( jpg png ?:help skip:png ) : png
ffmpeg version 3.4.6-0ubuntu0.18.04.1 Copyright (c) 2000-2019 the FFmpeg developers
  built with gcc 7 (Ubuntu 7.3.0-16ubuntu3)
  configuration: --prefix=/usr --extra-version=0ubuntu0.18.04.1 --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --enable-gpl --disable-stripping --enable-avresample --enable-avisynth --enable-gnutls --enable-ladspa --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librubberband --enable-librsvg --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvorbis --enable-libvpx --enable-libwavpack --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzmq --enable-libzvbi --enable-omx --enable-openal --enable-opengl --enable-sdl2 --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-chromaprint --enable-frei0r --enable-libopencv --enable-libx264 --enable-shared
  libavutil      55. 78.100 / 55. 78.100
  libavcodec     57.107.100 / 57.107.100
  libavformat    57. 83.100 / 57. 83.100
  libavdevice    57. 10.100 / 57. 10.100
  libavfilter     6.107.100 /  6.107.100
  libavresample   3.  7.  0 /  3.  7.  0
  libswscale      4.  8.100 /  4.  8.100
  libswresample   2.  9.100 /  2.  9.100
  libpostproc    54.  7.100 / 54.  7.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '/content/workspace/data_src.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 19529854
    compatible_brands: mp42isom
    creation_time   : 2018-04-19T07:29:40.000000Z
  Duration: 00:00:27.36, start: 0.041708, bitrate: 4027 kb/s
    Stream #0:0(eng): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709), 1920x1080 [SAR 1:1 DAR 16:9], 4025 kb/s, 23.98 fps, 23.98 tbr, 24k tbn, 47.95 tbc (default)
    Metadata:
      creation_time   : 2018-04-19T07:29:40.000000Z
      handler_name    : Video Media Handler
      encoder         : AVC Coding
Stream mapping:
  Stream #0:0 -> #0:0 (h264 (native) -> png (native))
Press [q] to stop, [?] for help
Output #0, image2, to '/content/workspace/data_src/%5d.png':
  Metadata:
    major_brand     : mp42
    minor_version   : 19529854
    compatible_brands: mp42isom
    encoder         : Lavf57.83.100
    Stream #0:0(eng): Video: png, rgb24, 1920x1080 [SAR 1:1 DAR 16:9], q=2-31, 200 kb/s, 23.98 fps, 23.98 tbn, 23.98 tbc (default)
    Metadata:
      creation_time   : 2018-04-19T07:29:40.000000Z
      handler_name    : Video Media Handler
      encoder         : Lavc57.107.100 png
frame=  655 fps= 12 q=-0.0 Lsize=N/A time=00:00:27.31 bitrate=N/A speed=0.516x    
video:501791kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: unknown
Done.

Step 4
/content
ffmpeg version 3.4.6-0ubuntu0.18.04.1 Copyright (c) 2000-2019 the FFmpeg developers
  built with gcc 7 (Ubuntu 7.3.0-16ubuntu3)
  configuration: --prefix=/usr --extra-version=0ubuntu0.18.04.1 --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --enable-gpl --disable-stripping --enable-avresample --enable-avisynth --enable-gnutls --enable-ladspa --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librubberband --enable-librsvg --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvorbis --enable-libvpx --enable-libwavpack --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzmq --enable-libzvbi --enable-omx --enable-openal --enable-opengl --enable-sdl2 --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-chromaprint --enable-frei0r --enable-libopencv --enable-libx264 --enable-shared
  libavutil      55. 78.100 / 55. 78.100
  libavcodec     57.107.100 / 57.107.100
  libavformat    57. 83.100 / 57. 83.100
  libavdevice    57. 10.100 / 57. 10.100
  libavfilter     6.107.100 /  6.107.100
  libavresample   3.  7.  0 /  3.  7.  0
  libswscale      4.  8.100 /  4.  8.100
  libswresample   2.  9.100 /  2.  9.100
  libpostproc    54.  7.100 / 54.  7.100
Input #0, image2, from '/content/workspace/data_src/%5d.png':
  Duration: 00:00:26.20, start: 0.000000, bitrate: N/A
    Stream #0:0: Video: png, rgb24(pc), 1920x1080 [SAR 1:1 DAR 16:9], 25 fps, 25 tbr, 25 tbn, 25 tbc
Stream mapping:
  Stream #0:0 (png) -> hqdn3d
  hqdn3d -> Stream #0:0 (png)
Press [q] to stop, [?] for help
Output #0, image2, to '/content/workspace/data_src/%5d.png':
  Metadata:
    encoder         : Lavf57.83.100
    Stream #0:0: Video: png, rgb24, 1920x1080 [SAR 1:1 DAR 16:9], q=2-31, 200 kb/s, 25 fps, 25 tbn, 25 tbc
    Metadata:
      encoder         : Lavc57.107.100 png
frame=  655 fps=7.9 q=-0.0 Lsize=N/A time=00:00:26.20 bitrate=N/A speed=0.314x    
video:533830kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: unknown
Done.

Step 5
/content
Performing 1st pass...
Running on Tesla T4.
Using TensorFlow backend.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
100% 655/655 [00:54<00:00, 12.11it/s]/usr/lib/python3.6/multiprocessing/semaphore_tracker.py:143: UserWarning: semaphore_tracker: There appear to be 1 leaked semaphores to clean up at shutdown
  len(cache))
100% 655/655 [00:54<00:00, 12.08it/s]
Performing 2nd pass...
Running on Tesla T4.
Using TensorFlow backend.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
100% 655/655 [01:42<00:00,  6.41it/s]/usr/lib/python3.6/multiprocessing/semaphore_tracker.py:143: UserWarning: semaphore_tracker: There appear to be 1 leaked semaphores to clean up at shutdown
  len(cache))
100% 655/655 [01:42<00:00,  6.40it/s]
Performing 3rd pass...
Running on CPU0.
Running on CPU1.
100% 655/655 [00:13<00:00, 49.03it/s]/usr/lib/python3.6/multiprocessing/semaphore_tracker.py:143: UserWarning: semaphore_tracker: There appear to be 1 leaked semaphores to clean up at shutdown
  len(cache))
/usr/lib/python3.6/multiprocessing/semaphore_tracker.py:143: UserWarning: semaphore_tracker: There appear to be 1 leaked semaphores to clean up at shutdown
  len(cache))
100% 655/655 [00:13<00:00, 48.93it/s]
-------------------------
Images found:        655
Faces detected:      654
-------------------------
Done.


Step 6:
/content
Running sort tool.

Sorting by histogram similarity...
Running on CPU0.
Sorting: 100% 654/654 [00:01<00:00, 456.73it/s]/usr/lib/python3.6/multiprocessing/semaphore_tracker.py:143: UserWarning: semaphore_tracker: There appear to be 1 leaked semaphores to clean up at shutdown
  len(cache))
Sorting: 100% 654/654 [00:01<00:00, 447.74it/s]
Renaming: 100% 654/654 [00:00<00:00, 24386.36it/s]
Done.


Step 7
/content
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
Archive exist!
Time to end session: 11 hours
Running trainer.

Loading model...

Model first run. Enter model options as default for each run.
Write preview history? (y/n ?:help skip:n) : ?
Preview history will be writed to <ModelName>_history folder.
Write preview history? (y/n ?:help skip:n) : y
Target iteration (skip:unlimited/default) : 
0
Batch_size (?:help skip:0) : ?
Larger batch size is better for NN's generalization, but it can cause Out of Memory error. Tune this value for your videocard manually.
Batch_size (?:help skip:0) : 
0
Feed faces to network sorted by yaw? (y/n ?:help skip:n) : ?
NN will not learn src face directions that don't match dst face directions. Do not use if the dst face has hair that covers the jaw.
Feed faces to network sorted by yaw? (y/n ?:help skip:n) : 
n
Flip faces randomly? (y/n ?:help skip:y) : ?
Predicted face will look more naturally without this option, but src faceset should cover all face directions as dst faceset.
Flip faces randomly? (y/n ?:help skip:y) : y
Src face scale modifier % ( -30...30, ?:help skip:0) : ?
If src face shape is wider than dst, try to decrease this value to get a better result.
Src face scale modifier % ( -30...30, ?:help skip:0) : 
0
Resolution ( 64-256 ?:help skip:128) : ?
More resolution requires more VRAM and time to train. Value will be adjusted to multiple of 16.
Resolution ( 64-256 ?:help skip:128) : 
128
Half or Full face? (h/f, ?:help skip:f) : 
f
Learn mask? (y/n, ?:help skip:y) : ?
Learning mask can help model to recognize face directions. Learn without mask can reduce model size, in this case converter forced to use 'not predicted mask' that is not smooth as predicted. Model with style values can be learned without mask and produce same quality result.
Learn mask? (y/n, ?:help skip:y) : y
Optimizer mode? ( 1,2,3 ?:help skip:1) : ?
1 - no changes. 2 - allows you to train x2 bigger network consuming RAM. 3 - allows you to train x3 bigger network consuming huge amount of RAM and slower, depends on CPU power.
Optimizer mode? ( 1,2,3 ?:help skip:1) : 
1
AE architecture (df, liae ?:help skip:df) : ?
'df' keeps faces more natural. 'liae' can fix overly different face shapes.
AE architecture (df, liae ?:help skip:df) : df
AutoEncoder dims (32-1024 ?:help skip:512) : ?
All face information will packed to AE dims. If amount of AE dims are not enough, then for example closed eyes will not be recognized. More dims are better, but require more VRAM. You can fine-tune model size to fit your GPU.
AutoEncoder dims (32-1024 ?:help skip:512) : 512
Encoder dims per channel (21-85 ?:help skip:42) : ?
More encoder dims help to recognize more facial features, but require more VRAM. You can fine-tune model size to fit your GPU.
Encoder dims per channel (21-85 ?:help skip:42) : 42
Decoder dims per channel (10-85 ?:help skip:21) : ?
More decoder dims help to get better details, but require more VRAM. You can fine-tune model size to fit your GPU.
Decoder dims per channel (10-85 ?:help skip:21) : 21
Use multiscale decoder? (y/n, ?:help skip:n) : ?
Multiscale decoder helps to get better details.
Use multiscale decoder? (y/n, ?:help skip:n) : n
Use CA weights? (y/n, ?:help skip: n ) : ?
Initialize network with 'Convolution Aware' weights. This may help to achieve a higher accuracy model, but consumes a time at first run.
Use CA weights? (y/n, ?:help skip: n ) : n
Use pixel loss? (y/n, ?:help skip: n ) : ?
Pixel loss may help to enhance fine details and stabilize face color. Use it only if quality does not improve over time. Enabling this option too early increases the chance of model collapse.
Use pixel loss? (y/n, ?:help skip: n ) : n
Face style power ( 0.0 .. 100.0 ?:help skip:0.00) : ?
Learn to transfer face style details such as light and color conditions. Warning: Enable it only after 10k iters, when predicted face is clear enough to start learn style. Start from 0.1 value and check history changes. Enabling this option increases the chance of model collapse.
Face style power ( 0.0 .. 100.0 ?:help skip:0.00) : 
0.0
Background style power ( 0.0 .. 100.0 ?:help skip:0.00) : ?
Learn to transfer image around face. This can make face more like dst. Enabling this option increases the chance of model collapse.
Background style power ( 0.0 .. 100.0 ?:help skip:0.00) : 
0.0
Apply random color transfer to src faceset? (y/n, ?:help skip:n) : ?
Increase variativity of src samples by apply LCT color transfer from random dst samples. It is like 'face_style' learning, but more precise color transfer and without risk of model collapse, also it does not require additional GPU resources, but the training time may be longer, due to the src faceset is becoming more diverse.
Apply random color transfer to src faceset? (y/n, ?:help skip:n) : n
Pretrain the model? (y/n, ?:help skip:n) : ?
Pretrain the model with large amount of various faces. This technique may help to train the fake with overly different face shapes and light conditions of src/dst data. Face will be look more like a morphed. To reduce the morph effect, some model files will be initialized but not be updated after pretrain: LIAE: inter_AB.h5 DF: encoder.h5. The longer you pretrain the model the more morphed face will look. After that, save and run the training again.
Pretrain the model? (y/n, ?:help skip:n) : n
Using TensorFlow backend.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
Loading: 100% 654/654 [00:01<00:00, 587.27it/s]
Loading: 0it [00:00, ?it/s]
SalesVolumeChange
"""




