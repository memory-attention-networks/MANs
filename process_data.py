import sys

import os
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import lmdb
from keras.utils import np_utils
from sklearn.decomposition import PCA
classes = {
    0: 'drink water',
    1: 'eat meal/snack',
    2: 'brushing teeth',
    3: 'brushing hair',
    4: 'drop',
    5: 'pickup',
    6: 'throw',
    7: 'sitting down',
    8: 'standing up (from sitting position)',
    9: 'clapping',
    10: 'reading',
    11: 'writing',
    12: 'tear up paper',
    13: 'wear jacket',
    14: 'take off jacket',
    15: 'wear a shoe',
    16: 'take off a shoe',
    17: 'wear on glasses',
    18: 'take off glasses',
    19: 'put on a hat/cap',
    20: 'take off a hat/cap',
    21: 'cheer up',
    22: 'hand waving',
    23: 'kicking something',
    24: 'put something inside pocket / take out something from pocket',
    25: 'hopping (one foot jumping)',
    26: 'jump up',
    27: 'make a phone call/answer phone',
    28: 'playing with phone/tablet',
    29: 'typing on a keyboard',
    30: 'pointing to something with finger',
    31: 'taking a selfie',
    32: 'check time (from watch)',
    33: 'rub two hands together',
    34: 'nod head/bow',
    35: 'shake head',
    36: 'wipe face',
    37: 'salute',
    38: 'put the palms together',
    39: 'cross hands in front (say stop)',
    40: 'sneeze/cough',
    41: 'staggering',
    42: 'falling',
    43: 'touch head (headache)',
    44: 'touch chest (stomachache/heart pain)',
    45: 'touch back (backache)',
    46: 'touch neck (neckache)',
    47: 'nausea or vomiting condition',
    48: 'use a fan (with hand or paper)/feeling warm',
    49: 'punching/slapping other person',
    50: 'kicking other person',
    51: 'pushing other person',
    52: 'pat on back of other person',
    53: 'point finger at the other person',
    54: 'hugging other person',
    55: 'giving something to other person',
    56: 'touch other persons pocket',
    57: 'handshaking',
    58: 'walking towards each other',
    59: 'walking apart from each other'
    }

training_subjects = [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]
training_views=[2,3]
joint_num = 50
n_classes = 60
scale_number=50

def draw_skeleton(feature1,feature2,feature3,show=1,save=0,outname=''):
  '''
  plot the skeleton sequence
  '''
  ax = plt.subplot(111,projection='3d')
  colors = ['black']*23
  colors.insert(1,'red')
  colors.insert(0,'red')


  for i in range(0,len(feature1)):
      if i%30!=0:
          continue
      for j in range(25):
        ax.scatter(float(feature1[i][j]),float(feature2[i][j]),float(feature3[i][j]),color=colors[int(j)],s=10,linewidths=0.1)
      connectivity = [(0,1),(1,20),(20,2),(2,3),(20,8),(8,9),(9,10),(10,11),(11,24),(24,23),
        (20,4),(4,5),(5,6),(6,7),(7,22),(22,21),(0,16),(16,17),(17,18),(18,19),(0,12),(12,13),(13,14),(14,15)]

      for connection in connectivity:
        t = connection[0]
        f = connection[1]

        ax.plot([float(feature1[i][t]),float(feature1[i][f])],[float(feature2[i][t]),
                                                               float(feature2[i][f])],[float(feature3[i][t]),float(feature3[i][f])])

  ax.view_init(elev=None,azim=45)
  ax.set_xlim(-1,1)
  ax.set_ylim(-1, 1)
  ax.set_zlim(-1, 1)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  if show:
    plt.show()
  if save:
    plt.savefig(outname+".png")


def vids_with_missing_skeletons():
  f = open('samples_with_missing_skeletons.txt','r')
  bad_files = []
  for line in f:
    bad_files.append(line.strip()+'.skeleton')
  f.close()
  return bad_files

def generate_data(argv):
  bad_files = vids_with_missing_skeletons() #get the missing skeleton
  skeleton_dir_root = "data_initial"  #store the initial *txt skeleton file

  skeleton_files = os.listdir(skeleton_dir_root)
  data_out_dir = 'data_ntu_subject/' #get the lmdb file
  batch_size = 1500
  lmdb_file_x = os.path.join(data_out_dir,'Xtrain_lmdb')
  lmdb_file_y = os.path.join(data_out_dir,'Ytrain_lmdb')

  lmdb_env_x = lmdb.open(lmdb_file_x, map_size=int(1e12))
  lmdb_env_y = lmdb.open(lmdb_file_y, map_size=int(1e12))
  lmdb_txn_x = lmdb_env_x.begin(write=True)
  lmdb_txn_y = lmdb_env_y.begin(write=True)

  lmdb_file_x1 = os.path.join(data_out_dir,'Xtest_lmdb')
  lmdb_file_y1 = os.path.join(data_out_dir,'Ytest_lmdb')

  lmdb_env_x1 = lmdb.open(lmdb_file_x1, map_size=int(1e12))
  lmdb_env_y1 = lmdb.open(lmdb_file_y1, map_size=int(1e12))
  lmdb_txn_x1 = lmdb_env_x1.begin(write=True)
  lmdb_txn_y1 = lmdb_env_y1.begin(write=True)


  count = 0
  count1=0
  count2=0
  for file_name in skeleton_files:

    if file_name in bad_files:
      continue

    action_class = int(file_name[file_name.find('A')+1:file_name.find('A')+4])
    subject_id = int(file_name[file_name.find('P')+1:file_name.find('P')+4])
    view_id = int(file_name[file_name.find('C')+1:file_name.find('C')+4])

    sf = open(os.path.join(skeleton_dir_root,file_name),'r')
    num_frames = int(sf.readline())
  
    feature1 = np.zeros((num_frames, joint_num))
    feature2 = np.zeros((num_frames, joint_num))
    feature3 = np.zeros((num_frames, joint_num))
    rotation=[]
    origin=[]
    b_first = 0 #record the body count
    n_first=0 #record the frame count

    ## generate the feature
    for n in range(0,num_frames):
      body_count = int(sf.readline())
      if body_count > 2:

        for b in range(0,body_count):
          body_info = sf.readline()
          joint_count = int(sf.readline())
          for j in range(0,joint_count):
            joint_info = sf.readline()
      else:
        body_information = {} # a dict to store all the transformed joint information

        for b in range(0,body_count):
          n_first += 1
          if b==1:
              b_first+=1
          body_info = sf.readline()

          joint_count = int(sf.readline())

          joint_dict = {} # a dict to store the joint coordinate
          joint_new={}   # a dict to store the transformed coordinate
          for j in range(0,joint_count):
            joint_info = sf.readline()
            jsp = joint_info.split()
            x = float(jsp[0])
            y = float(jsp[1])
            z = float(jsp[2])

            joint_dict[j] = (x,y,z)
          
          ## pre-processing, which obtains the origin and the rotation matrix, and transforms all the coordinates
          if n_first==1 or b_first==1:

              origin.append(np.array(((joint_dict[12][0] + joint_dict[16][0]) / 2, (joint_dict[12][1] + joint_dict[16][1]) / 2,
                                  (joint_dict[12][2] + joint_dict[16][2]) / 2)))
              ## use PCA to get the z-axis
              index = [0, 1, 4, 8, 12,
                       16, 20]
              j1 = np.array([joint_dict[i][0]-origin[b][0] for i in index]).reshape(-1,1)
              j2 = np.array([joint_dict[i][1]-origin[b][1] for i in index]).reshape(-1,1)
              j3 = np.array([joint_dict[i][2]-origin[b][2] for i in index]).reshape(-1,1)

              torso = np.concatenate([j1, j2, j3], axis=1)
              pca = PCA(n_components=3)
              pca.fit(torso)

              v3 = pca.components_[0]
              u=v3

              ## get the x-axis and y-axis, create a new orthogonal coordinate system
              v = np.array((joint_dict[12][0] - joint_dict[16][0], joint_dict[12][1] - joint_dict[16][1], joint_dict[12][2] - joint_dict[16][2]))

              v2=np.cross(v,u)
              v1=np.cross(v2,v3)
              v1_final=v1/sum(v1**2)**0.5
              v2_final=v2/sum(v2**2)**0.5
              v3_final=v3/sum(v3**2)**0.5
              v1_final.shape=(3,1)
              v2_final.shape=(3,1)
              v3_final.shape=(3,1)
              rotation.append(np.concatenate([v1_final,v2_final,v3_final],1))

          for i in range(0,joint_count):
              ## sub the origin and multiply with the rotation matrix, to get the transformed joint coordinate
              joint_new[i]=np.mat(rotation[b]).I*np.reshape((joint_dict[i]-origin[b]),[3,1])

          body_information[b] = joint_new

        sample_ind = 0
        sample1 = np.zeros((joint_num))
        sample2 = np.zeros((joint_num))
        sample3 = np.zeros((joint_num))
        ## construct the n-th frame
        for bind, body in body_information.items():
          for jind, joint in body.items():
            sample1[sample_ind] = joint[0] #x
            sample2[sample_ind] = joint[1] #y
            sample3[sample_ind] = joint[2] #z
            sample_ind += 1

        feature1[n] = sample1
        feature2[n] = sample2
        feature3[n] = sample3

    if feature3[0][4] < 0:
        feature3 = -feature3
    find_y1 = np.mean(feature2[:, 15]) - np.mean(feature2[:, 14])
    find_y2 = np.mean(feature2[:, 19]) - np.mean(feature2[:, 18])

    if find_y1 < 0 or find_y2 < 0:
        feature2 = -feature2
    if np.mean(feature1[:,16])-np.mean(feature1[:,12])<0:
        feature1=-feature1

    ## construct the final feature, using padding and down-sampling
    pad1=int((scale_number-num_frames)/2)
    pad2=scale_number-num_frames-pad1
    pad3=int((scale_number-joint_num)/2)
    pad4=scale_number-joint_num-pad3
    if pad1>=0 and pad2>=0:
      feature1_new=np.lib.pad(feature1, ((pad1,pad2),(pad3,pad4)), 'constant',constant_values=0).reshape(scale_number,scale_number,1)
      feature2_new =np.lib.pad(feature2, ((pad1,pad2),(pad3,pad4)), 'constant',
                 constant_values=0).reshape(scale_number,scale_number,1)
      feature3_new =np.lib.pad(feature3, ((pad1,pad2),(pad3,pad4)), 'constant',
                 constant_values=0).reshape(scale_number,scale_number,1)
    else:
      feature1_index = np.linspace(0, num_frames - 1, scale_number, dtype=int)

      feature1_new = np.lib.pad(feature1[feature1_index], ((0, 0), (pad3, pad4)), 'constant',
                                  constant_values=0).reshape(scale_number, scale_number, 1)
      feature2_new = np.lib.pad(feature2[feature1_index], ((0, 0), (pad3, pad4)), 'constant',
                                  constant_values=0).reshape(scale_number, scale_number, 1)
      feature3_new = np.lib.pad(feature3[feature1_index], ((0, 0), (pad3, pad4)), 'constant',
                                  constant_values=0).reshape(scale_number, scale_number, 1)
    feature=np.concatenate([feature1_new,feature2_new,feature3_new],axis=2) # the final feature, (scale_number,scale_number,3)
    if body_count <= 2:
      if subject_id in training_subjects:
        X=feature
        Y = np_utils.to_categorical(action_class-1, n_classes)
        keystr = '{:0>8d}'.format(count1)
        count1+=1
        lmdb_txn_x.put(keystr.encode(), X.tobytes())
        lmdb_txn_y.put(keystr.encode(), Y.tobytes())
      else:
        X=feature
        Y = np_utils.to_categorical(action_class - 1, n_classes)
        keystr = '{:0>8d}'.format(count2)
        count2+=1
        lmdb_txn_x1.put(keystr.encode(), X.tobytes())
        lmdb_txn_y1.put(keystr.encode(), Y.tobytes())
    sf.close()
    count += 1
    print(count,file_name)
    if count1 % batch_size == 0:

        lmdb_txn_x.commit()
        lmdb_txn_x = lmdb_env_x.begin(write=True)
        lmdb_txn_y.commit()
        lmdb_txn_y = lmdb_env_y.begin(write=True)
        print(count1)

    if count2%batch_size==0:

        lmdb_txn_x1.commit()
        lmdb_txn_x1 = lmdb_env_x1.begin(write=True)
        lmdb_txn_y1.commit()
        lmdb_txn_y1 = lmdb_env_y1.begin(write=True)
        print(count2)

  # write last batch
  if count1 % batch_size != 0:
    lmdb_txn_x.commit()
    lmdb_txn_y.commit()
    print ('last train batch')


  # write last batch
  if count2 % batch_size != 0:
    lmdb_txn_x1.commit()
    lmdb_txn_y1.commit()
    print ('last test batch')

  print ("Training samples number: ",count1, "Testing samples number:", count2)
  
  

if __name__ == "__main__":
  generate_data(sys.argv)
