###########################################################################################################
# Libraries
###########################################################################################################

# General purpose and plotting
import numpy as np
import rosbag
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

import time
import math
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# EMG----------------------------------------------
from scipy.signal import butter, filtfilt, iirnotch, freqz
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA


# Motion capture-----------------------------------
from scipy.spatial.transform import Rotation as R
from scipy.optimize import linear_sum_assignment, linear_sum_assignment
from scipy.spatial.distance import cdist
from ikpy.chain import Chain
from ikpy.link import URDFLink
# ikpy library has been directly downloaded and slightly modified for our personal purposes, 
# can be found in the folders, optimization may be needed to avoid this process