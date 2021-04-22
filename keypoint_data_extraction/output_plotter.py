import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import copy


def distanceBetweenKeypoints(kpt_1_u, kpt_1_v, kpt_2_u, kpt_2_v):
    v = (kpt_1_u-kpt_2_u, kpt_1_v-kpt_2_v)
    d  = np.linalg.norm(v)
    return d

def elbowAngle(d_e_w, d_s_w, d_e_s):
    theta = np.arccos((d_e_w*d_e_w+d_e_s*d_e_s-d_s_w*d_s_w)/(2*d_e_w*d_e_s))
    return theta*180/np.pi

def computeAugmentedData(detected_keypoints):
    

class keypointsDataLogger():
    def __init__(self):

plt.style.use('ggplot')

# Image plane: The reference frame for the pixel values is situated on the top-left corner
#        u
#    o------> 
#    |   ________
#  v |  |        |
#    |  | __()__ | 
#    v  |   ||   |
#       |   /\   | 
#       |________|

class openPoseOutputLogger:
   
    def __init_figure(self, keypoint, frame_size):
        """Each keypoint's data will be ploted in a different figure. This functions takes care of initializing said figure

        Args:
            keypoint (int): Used to define the title of the fire to identify which keypoints is representing
            frame_size ([type]): size of the image frame used by OpenPose, this is used to define the maximum values for the u and v coordinates

        Returns:
            tuple: list containing the figure and axis objects used for plotting the keypoint data.
        """
        fig = plt.figure('Keypoint: ' + str(keypoint))
        ax_x = plt.subplot(1, 2, 1)
        ax_y = plt.subplot(1, 2, 2)
        ax_x.set_xlabel('time')
        ax_x.set_title('u')
        ax_x.set_ylim(0, frame_size[0])
        ax_y.set_xlabel('time')
        ax_y.set_title('v')
        ax_y.set_ylim(0, frame_size[1])

        return [fig, ax_x, ax_y]

    def __init__(self, keypoint_list = [], frame_size = (200,200)):  
        """ Constructor of the logger class. It will log all the keypoints data into a time series. Presenting the data as to pandas dataframes, one for each image coordinate.

        Args:
            keypoint_list (list, optional): List of the keypoints to be ploted. Defaults to [].
            frame_size (tuple, optional): size of the image frame used by OpenPose, this is used to define the maximum values for the u and v coordinates. Defaults to (200,200).
        """

        # Initialize figures, axis and plotting lines for each keypoint
        self.keypoint_list = keypoint_list
        plt.ion()
        self.fig_list, self.ax_x_list , self.ax_y_list  = zip(*[ self.__init_figure(keypoint, frame_size) for keypoint in self.keypoint_list])
        self.line_x_list, _ = zip(*[ self.ax_x_list[i].plot(0, 0, 0.8) for i in range(len(self.keypoint_list))])
        self.line_y_list, _ = zip(*[ self.ax_y_list[i].plot(0, 0, 0.8) for i in range(len(self.keypoint_list))])
      
        # Containers for the time series data for the u and v coordinates of the keypoints
        self.t0 = time.time()
        self.kpt_data = []

        self.add_derivative_data = False
        self.add_keypoints_pair_distances_data = True

        if self.add_derivative_data:
            self.line_x_fod_list, _ = zip(*[ self.ax_x_list[i].plot(0, 0, 0.8) for i in range(len(self.keypoint_list))])
            self.line_x_sod_list, _ = zip(*[ self.ax_x_list[i].plot(0, 0, 0.8) for i in range(len(self.keypoint_list))])
            self.line_y_fod_list, _ = zip(*[ self.ax_y_list[i].plot(0, 0, 0.8) for i in range(len(self.keypoint_list))])
            self.line_y_sod_list, _ = zip(*[ self.ax_y_list[i].plot(0, 0, 0.8) for i in range(len(self.keypoint_list))])


    def __updateAuxiliary(self, data_uv, kpts_data):
        """[summary]

        Args:
            data_uv (list): List containing the u and v coordinates data for the keypoints during one timestep
            kpts_data (list): keypoint data to be added to the lists
        """

        if kpts_data == []:
            point = [0, 0]
        else:
            point = [kpts_data[0][0], kpts_data[0][1]]
        
        data_uv.append(point[0])
        data_uv.append(point[1])


    def updateKeypointTimeSeries(self, detected_kpts):
        """Add keypoints data to the time series containers

        Args:
            detected_kpts (list): data for the detected keypoints
        """
        # Create list for the current keypoints data
        t = time.time() - self.t0 
        data_uv = [t] 
        [ self.__updateAuxiliary(data_uv, detected_kpts[i]) for i in range(len(detected_kpts))]

        # Add current data to the time series containers
        self.kpt_data.append(data_uv)

    def __loggedDataToPandas(self):
        """Convert the time series containers to a pandas dataframe.

        Returns:
            tuple: pandas dataframe for the u and v coordinates data
        """
        # Convert data to pandas dataframe
        columns_names = ['t', 'K_0_u', 'K_0_v', 
                              'K_1_u', 'K_1_v',  
                              'K_2_u', 'K_2_v',
                              'K_3_u', 'K_3_v', 
                              'K_4_u', 'K_4_v',
                              'K_5_u', 'K_5_v',
                              'K_6_u', 'K_6_v', 
                              'K_7_u', 'K_7_v', 
                              'K_8_u', 'K_8_v', 
                              'K_9_u', 'K_9_v', 
                              'K_10_u', 'K_10_v', 
                              'K_11_u', 'K_11_v', 
                              'K_12_u', 'K_12_v', 
                              'K_13_u', 'K_13_v', 
                              'K_14_u', 'K_14_v', 
                              'K_15_u', 'K_15_v', 
                              'K_16_u', 'K_16_v', 
                              'K_17_u', 'K_17_v']
        uv_data = pd.DataFrame(self.kpt_data, columns=columns_names)
        return uv_data

    def __updatePlot(self, uv_data, selected_kpt_index):
        """Auxiliary function to update the keypoints data plots

        Args:
            uv_data (pandas dataframe): Data for the u and v coordinates of all the keypoints
            selected_kpt_index (int): [description]
        """
        # u coordinate plot
        self.line_x_list[selected_kpt_index].set_data(uv_data['t'].tolist(), uv_data['K_'+str(selected_kpt_index)+'_u'].tolist())
        self.ax_x_list[selected_kpt_index].set_xlim(0, time.time() - self.t0)
        # v coordinate plot
        self.line_y_list[selected_kpt_index].set_data(uv_data['t'].tolist(), uv_data['K_'+str(selected_kpt_index)+'_v'].tolist())
        self.ax_y_list[selected_kpt_index].set_xlim(0, time.time() - self.t0) 
  
        if self.add_derivative_data:
            self.line_y_fod_list[selected_kpt_index].set_data(uv_data['t'].tolist(), uv_data['fod_K_'+str(selected_kpt_index)+'_v'].tolist()) 
            self.line_y_sod_list[selected_kpt_index].set_data(uv_data['t'].tolist(), uv_data['sod_K_'+str(selected_kpt_index)+'_v'].tolist())       
            self.line_x_fod_list[selected_kpt_index].set_data(uv_data['t'].tolist(), uv_data['fod_K_'+str(selected_kpt_index)+'_u'].tolist()) 
            self.line_x_sod_list[selected_kpt_index].set_data(uv_data['t'].tolist(), uv_data['sod_K_'+str(selected_kpt_index)+'_u'].tolist())       

       
    def plot(self):   
        """Graph the data for the u and v coordinates of the selected keypoints
        """
        # Get u and v coordinates data for all the keypoints
        uv_data = self.getData()    
      
        # Update visualization        
        [self.__updatePlot(uv_data, selected_kpt_index) for selected_kpt_index in range(len(self.keypoint_list))]

        plt.pause(0.001)

    def __addFirstOrderDerivatives(self, keypoints_data): 

        for selected_kpt_index in range(0,18):
            keypoints_data['sod_K_'+str(selected_kpt_index)+'_u'] = keypoints_data['K_'+str(selected_kpt_index)+'_u'] - 2*keypoints_data['K_'+str(selected_kpt_index)+'_u'].shift(1) + keypoints_data['K_'+str(selected_kpt_index)+'_u'].shift(2)
            keypoints_data['sod_K_'+str(selected_kpt_index)+'_v'] = keypoints_data['K_'+str(selected_kpt_index)+'_v'] - 2*keypoints_data['K_'+str(selected_kpt_index)+'_v'].shift(1) + keypoints_data['K_'+str(selected_kpt_index)+'_v'].shift(2)

        return keypoints_data


    def __addSecondOrderDerivatives(self, keypoints_data): 

        for selected_kpt_index in range(0,18):
            keypoints_data['fod_K_'+str(selected_kpt_index)+'_u'] = keypoints_data['K_'+str(selected_kpt_index)+'_u'].diff()
            keypoints_data['fod_K_'+str(selected_kpt_index)+'_v'] = keypoints_data['K_'+str(selected_kpt_index)+'_v'].diff()
        
        return keypoints_data

    def __addKeypointPairDistances(self, keypoints_data):
        # Compute the distance from the left elbow to the left shoulder
        keypoints_data['d_le_ls'] = keypoints_data.apply(lambda row : distanceBetweenKeypoints(row['K_6_u'],row['K_6_v'], row['K_5_u'],row['K_5_v']), axis = 1)
        # Compute the distance from the right elbow to the right shoulder
        keypoints_data['d_re_rs'] = keypoints_data.apply(lambda row : distanceBetweenKeypoints(row['K_3_u'],row['K_3_v'], row['K_2_u'],row['K_2_v']), axis = 1)
        # Compute the distance from the right elbow to the right wrist
        keypoints_data['d_re_rw'] = keypoints_data.apply(lambda row : distanceBetweenKeypoints(row['K_3_u'],row['K_3_v'], row['K_4_u'],row['K_4_v']), axis = 1)
        # Compute the distance from the left elbow to the left wrist
        keypoints_data['d_le_lw'] = keypoints_data.apply(lambda row : distanceBetweenKeypoints(row['K_6_u'],row['K_6_v'], row['K_7_u'],row['K_7_v']), axis = 1)
        # Compute the distance from the left shoulder to the left wrist
        keypoints_data['d_ls_lw'] = keypoints_data.apply(lambda row : distanceBetweenKeypoints(row['K_5_u'],row['K_5_v'], row['K_7_u'],row['K_7_v']), axis = 1)
        # Compute the distance from the right shoulder to the right wrist
        keypoints_data['d_rs_rw'] = keypoints_data.apply(lambda row : distanceBetweenKeypoints(row['K_2_u'],row['K_2_v'], row['K_4_u'],row['K_4_v']), axis = 1)
        # Compute the distance from the right shoulder to the left shoulder
        keypoints_data['d_rs_ls'] = keypoints_data.apply(lambda row : distanceBetweenKeypoints(row['K_5_u'],row['K_5_v'], row['K_2_u'],row['K_2_v']), axis = 1)
        # Angle left elbow
        keypoints_data['theta_le'] = keypoints_data.apply(lambda row : elbowAngle(row['d_le_lw'], row['d_ls_lw'], row['d_le_ls']), axis = 1)
        # Angle right elbow 
        keypoints_data['theta_re'] = keypoints_data.apply(lambda row : elbowAngle(row['d_re_rw'], row['d_rs_rw'], row['d_re_rs']), axis = 1)

        print(keypoints_data['theta_re'])
        return keypoints_data
                     
                     
             
    def getData(self):
        """Withdraw the keypoints coordinates data

        Returns:
            (pandas dataframe): Data for the time series of all the keypoints, currently: u,v coordinates
        """
        keypoints_data = self.__loggedDataToPandas()

        if self.add_derivative_data:
            keypoints_data = self.__addFirstOrderDerivatives(keypoints_data)
            keypoints_data = self.__addSecondOrderDerivatives(keypoints_data)

        if self.add_keypoints_pair_distances_data:
            keypoints_data = self.__addKeypointPairDistances(keypoints_data)

        return keypoints_data

    def saveData(self, u_data_filename):
        """Dump the keypoints coordinates data into a csv file

        Args:
            u_data_filename (str): Filename for the .csv file for the u coordinate data
            v_data_filename (str): Filename for the .csv file for the v coordinate data
        """
        self.getData().to_csv(u_data_filename)
