"""
After unzipping the data `Traj Label Each C.rar` from `https://github.com/RadetzkyLi/3P-MSPointNet/tree/master/data`,
we obtain user's trajectory files (in csv format) with corrected annotations, where each file represents a trajectory.
The unzipped data is a folder of name `Traj Label Each -C`.
Then, 
    step1: divide or segment the trajectory into trips if time interval between two successive GPS points exceeding 20 minutes;
    step2: remove outliers and errors of the trip;
    step3: save each trip in a seperate csv file, results stored in folder `Traj Lable Trip`.
"""


from geopy.distance import geodesic
import numpy as np
import math
import os
import time
import copy
import pickle
from matplotlib import pyplot as plt
from matplotlib import rcParams

import sys
sys.path.append('D:/Anaconda/documents/TransModeDetection/')
from utils import util


## 0. Basic functions
def compute_distance(p1,p2):
    '''
    Computing distance of two positions in earth using Vencenty distance.
    In addition,coordinate is WGS-84. 
    :param p1 : (lat,lng) of position 1 in degreen,such as (41.00,113.9);
    :param p2 : (lat,lng) of position 2 in degreen;
    :return : distance of two positions in meters.
    '''
    lat_lng_1 = (p1[0],p1[1])
    lat_lng_2 = (p2[0],p2[1])
    return geodesic(lat_lng_1,lat_lng_2).meters
    
def compute_delta_time(p1,p2):
    return (p2[2]-p1[2])*24*3600

def compute_speed(distance,delta_time):
    return distance / delta_time

def compute_speed_point(p1,p2):
    return compute_distance(p1,p2) / compute_delta_time(p1,p2)

def compute_acceleration(speed1,speed2,delta_time):
    return (speed2 - speed1) / delta_time

def compute_jerk(acc1,acc2,delta_time):
    return (acc2 - acc1) / delta_time

def compute_bearing(p1, p2):
    '''
    Compute bearing of north and line p1-p2.
    :param p1: location of tuple or list,[lat,lng,...];
    '''
    y = math.sin(math.radians(p2[1]) - math.radians(p1[1])) * math.radians(math.cos(p2[0]))
    x = math.radians(math.cos(p1[0])) * math.radians(math.sin(p2[0])) - \
        math.radians(math.sin(p1[0])) * math.radians(math.cos(p2[0])) \
        * math.radians(math.cos(p2[1]) - math.radians(p1[1]))
    # Convert radian from -pi to pi to [0, 360] degree
    return (math.atan2(y, x) * 180. / math.pi + 360) % 360

def compute_bearing_rate(bearing1, bearing2):
    return abs(bearing1 - bearing2)

def merge_modes(point):
    '''
    Merge car and taxi into one mode,merge subway and train into one mide.
    :param point : location point of (lat,lng,timestamp,mode),such as
        (39.975666,116.331158,39882.5505671296,0)
    :return : same location point with new mode (lat,lng,timestamp,new_mode).
    '''
    if point[3] == 0 or point[3] == 1 or point[3] == 2:
        return point
    elif point[3] == 3 or point[3] == 4:
        new_point = point
        new_point[3] = 3
        return new_point
    elif point[3] == 5 or point[3] == 6 or point[3] == 7:
        new_point = point
        new_point[3] = 4
        return new_point
    else:
        raise ValueError('Unexpected mode that will be merged,expected one of 0~7 ,got ',point[3])
        
        
        
##  1. Functions for removing errors and outliers
def remove_error_seg(traj_seg):
    '''
    Remove point whose timestamp is less than its former.
    :param traj_seg : trajectory list of a segment.
    :return : trajectory after removing error.
    '''
    traj_save = [traj_seg[0]]
    for i in range(1,len(traj_seg)):
        if compute_delta_time(traj_seg[i-1],traj_seg[i]) > 0.1:
            traj_save.append(traj_seg[i])
    return traj_save
    
def merge_index(index_list):
    '''
    Merge neighboring index pairs where difference of first value of latter
    and second value of former is no larger than 1.
    :param index_list : one dim list of ascending order with dual length;
    :return : a list after merging.
    '''
    index_list_new = []
    inx_srt = index_list[0]
    inx_end = index_list[1]
    for i in range(2,len(index_list) - 1,2):
        if index_list[i] - inx_end <= 1:
            inx_end = index_list[i+1]
            continue
        index_list_new.append(inx_srt)
        index_list_new.append(inx_end)
        inx_srt = index_list[i]
        inx_end = index_list[i+1]
    if not index_list_new or inx_srt > index_list_new[-1]:
        index_list_new.append(inx_srt)
        index_list_new.append(inx_end)
    return index_list_new

def explore_index(index_list,traj_seg,max_velocity,max_acc):
    length_traj = len(traj_seg)
    has_explored = False
    for i in range(0,len(index_list) - 1,2):
        left = max(0,index_list[i] - 1)
        right = min(length_traj - 1,index_list[i+1] + 1)
        speed_1 = 0 if left == 0 else compute_speed_point(traj_seg[left-1],traj_seg[left])
        dist = compute_distance(traj_seg[left],traj_seg[right])
        delta_time = compute_delta_time(traj_seg[left],traj_seg[right])
        try:
            speed_2 = compute_speed(dist,delta_time)
        except Exception:
            raise ValueError('error:',i,left,right,index_list)
        acc = compute_acceleration(speed_1,speed_2,delta_time)
        if speed_2 > max_velocity or abs(acc) > max_acc:
            if index_list[i] != left or index_list[i+1] != right:
                has_explored = True
            index_list[i] = left
            index_list[i+1] = right
    return has_explored
    
def remove_outlier_seg(traj_seg,max_velocity=80,max_acc=10):
    '''
    filter outliers whose velocity or acc is too large.
    :param traj_seg : segment trajecotory;
    :param output_dir : such as '.../010/';
    :param max_velocity : max velocity of land transportation mode,
        default is 80 m/s;
    :param max_acc : max acceleration of land transportation mode,
        default is 10 m/s^2.
    '''    
    if len(traj_seg) < 3:
        return traj_seg
    delta_time_1 = compute_delta_time(traj_seg[0],traj_seg[1])
    dist_1 = compute_distance(traj_seg[0],traj_seg[1])
    speed_1 = compute_speed(dist_1,delta_time_1)
    del_index_list = []
    if speed_1 > max_velocity:
        del_index_list.append(0)
        del_index_list.append(1)
    for i in range(2,len(traj_seg)):
        delta_time_2 = compute_delta_time(traj_seg[i-1],traj_seg[i])
        dist_2 = compute_distance(traj_seg[i-1],traj_seg[i])
        speed_2 = compute_speed(dist_2,delta_time_2)
        acc = compute_acceleration(speed_1,speed_2,delta_time_2)
        if speed_2 > max_velocity or abs(acc) > max_acc:
            del_index_list.append(i-1)
            del_index_list.append(i)
        speed_1 = speed_2 
    # merge and explore around outlier,
    if del_index_list:
        while True:
            del_index_list = merge_index(del_index_list)
            if not explore_index(del_index_list,traj_seg,max_velocity,max_acc):
                break
    # delete found outlier
    del_index_list = list(set(del_index_list))
    del_index_list.sort()
    traj_save = copy.deepcopy(traj_seg)
    cnt_del = 0
    for i in range(0,len(del_index_list)-1,2):
        del traj_save[(del_index_list[i]-cnt_del):(del_index_list[i+1]+1-cnt_del)]
        cnt_del += del_index_list[i+1] - del_index_list[i] + 1
    return traj_save
    
    
def is_legal_seg(traj_seg):
    '''
    Judging whether a segment should be saved.  Legal means
    total number of points of segment is not less than 20 and
    total distance of segment is not less than 150 meters and
    total trip time is not less than 1 minitues.
    :param traj_seg : trajectory of segment;
    :return : True if segment satisfies all three conditions else False.
    '''
    if len(traj_seg) < 20:
        return False
    if (traj_seg[-1][2] - traj_seg[0][2])*24*60 < 1:
        return False
    dist_total = 0
    for i in range(len(traj_seg) - 1):
        dist_total += compute_distance(traj_seg[i],traj_seg[i+1])
    if dist_total < 150:
        return False
    return True
    
    
## 3. Divide or segment GPS trajectories into trips
def seg_one_trip(traj_one_trip,output_dir,file_name,trip_time_gap = 20*60,unit='segment'):
    '''
    Segment a trip into single one mode segments and save the trip as a file.
    :param traj_one_trip : trajectory data of a trip;
    :param output_dir : such as '.../010/';
    :param file_name : output file name which should be same as original trip
        such as '20110801101010';
    :param trip_time_gap : threshold time gap of two trips; 
    :param unit: if 'segment',segment trip into segments;elif 'trip',segment trip
        into more trips.
    '''
    def process_one_seg(traj_one_seg,file_path):
        # remove error point and outlier,then save segment as csv if
        # trajectory of segment is not null.
        traj_one_seg = remove_error_seg(traj_one_seg)
        traj_one_seg = remove_outlier_seg(traj_one_seg)
        if not traj_one_seg:
            return False
        trip_mode.append([file_name,traj_one_seg[0][3]])
        if is_legal_seg(traj_one_seg):
            util.save_list2csv(file_path,columns,traj_one_seg)
        return True
        
    traj_one_seg = []
    seg_cnt = 0
    columns = ['latitude','longitude','timestamp','mode']
    trip_mode = []
    for i in range(len(traj_one_trip) - 1):
        delta_time = compute_delta_time(traj_one_trip[i],traj_one_trip[i+1])
        mode_not_change = (traj_one_trip[i][3] == traj_one_trip[i+1][3])
        if unit == 'segment':
            cond1 = delta_time < trip_time_gap and mode_not_change and traj_one_trip[i][3] <= 7
            cond2 = delta_time >= trip_time_gap or not mode_not_change
        elif unit == 'trip':
            cond1 = delta_time < trip_time_gap and traj_one_trip[i][3] <= 7
            cond2 = delta_time >= trip_time_gap
        else:
            raise ValueError('Unexpected kinds of unit:',uint)
        # process a segment or trip
        if cond1:
            traj_one_seg.append(merge_modes(traj_one_trip[i]))
        elif cond2:
            traj_one_seg.append(merge_modes(traj_one_trip[i]))
            seg_file_name = file_name + "_" + str(seg_cnt) + '.csv'
            if process_one_seg(traj_one_seg,output_dir + seg_file_name):
                seg_cnt += 1
                traj_one_seg = []
    # deal with unsegmented trip and last segment of a trip
    if traj_one_seg:
        seg_file_name = file_name + '.csv' if seg_cnt==0 else file_name + "_" + str(seg_cnt) + '.csv'
        if process_one_seg(traj_one_seg,output_dir + seg_file_name):
            seg_cnt += 1
    return trip_mode
    
def seg_all_user(output_dir,data_dir,unit='segment'):
    start_time1 = time.process_time()
    users_list = os.listdir(data_dir)
    seg_num_cnt = [0 for i in range(7)]
    user_cnt = 1
    for user in users_list:
        if user_cnt <= 0:
            user_cnt += 1
            continue
        start_time2 = time.process_time()
        user_mode_change = []
        user_output_dir = output_dir + user + '/'
        if not os.path.exists(user_output_dir):
            os.mkdir(user_output_dir)
        user_data_dir = data_dir + user + '/'
        file_list = os.listdir(user_data_dir)
        for file in file_list:
            traj_one_trip = util.load_csv2list(user_data_dir + file)
            trip_mode = seg_one_trip(traj_one_trip,user_output_dir,file.rstrip('.csv'),unit=unit)
            user_mode_change.extend(trip_mode)
            seg_num = min(len(trip_mode),6)
            seg_num_cnt[seg_num] += 1
        # if segment trahectory into segments,then save mode change list
        if unit == 'segment':
            save_user_mode_change(user,user_output_dir,user_mode_change)
        print(user_cnt,' th user:',user,', consuming time:',time.process_time()-start_time2,' second.')
        user_cnt += 1
    print('seg num cnt:',seg_num_cnt)
    print('Total ',user_cnt-1," users' trip are segmented,consuming time:",time.process_time()-start_time1,' second')
    
## 4. Excuting
if __name__ == '__main__':
    data_dir = '../data/Traj Label Each - C/'
    seg_all_user(output_dir='../data/Traj Label Trip/',data_dir=data_dir,unit='trip')