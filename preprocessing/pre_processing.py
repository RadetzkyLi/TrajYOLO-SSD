#!/usr/bin/env python
# coding: utf-8

'''
This file process raw but cleanned GPS data into a eight-channnel format, where each
channel represents distance,speed,acc,jerk,delta_lat,delta_lng,delta_time,
and bearing_rate respectively. Features of each trip are stored in a csv file,
where the first eight columns represent the eight attributes, and the last column 
represents the corresponding tarnsportation modes (i.e., label).
Inputs:
    `Traj Label Trip` from `data_cleaning.py`;
Outputs:
    a folder `Feats Trip` where each GPS point is represented by 8 features mentoined-above.
'''

from geopy.distance import geodesic
import numpy as np
import math
import os
import time
import copy
import pickle
from matplotlib import pyplot as plt
from matplotlib import rcParams


#  1. Basic fucntions 
##  1.1 compute motion attributes
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



#   2. Extracting features of trip
def extract_features_seg(traj_seg,
                         extra_features=[]):
    '''
    Extracting point-wise features of trajectory of segment,
    each point will be extarcted speed,acceleration,jerk and 
    relative distance. Of which , distance and speed of first 
    point are set to zero, standing for each single mode trip 
    starting from still.
    :param traj_seg : trajectory of segment, each element of it
        contains [lat,lng,timestamp/delta time,...];
    :param extra_features: extra features list whose element
        must be one of ['delta_lat','delta_lng','delta_time','bearing_rate'],
        default is [];
    :return feat_seg: point-wise features of trajectory of a segment,
        each element contains one feature of all points.
    '''
    def compute_delta_time_2(p1,p2):
        # Used when p[3] is delta time from first pointx 
        return p2[2] - p1[2]
    
    if abs(traj_seg[0][2]) > 0.01:
        get_delta_time = compute_delta_time
    else:
        get_delta_time = compute_delta_time_2
    num_pts = len(traj_seg)
    delta_time_arr = [0]
    dist_arr = [0]
    speed_arr = [0]
    acc_arr = [0]
    jerk_arr = [0]
    for i in range(1,num_pts):
        delta_time = get_delta_time(traj_seg[i-1],traj_seg[i])
        delta_time_arr.append(delta_time)
        dist = compute_distance(traj_seg[i-1],traj_seg[i])
        dist_arr.append(dist)
        speed_arr.append(compute_speed(dist,delta_time))
        acc_arr.append(compute_acceleration(speed_arr[-2],speed_arr[-1],delta_time))
        jerk_arr.append(compute_jerk(acc_arr[-2],acc_arr[-1],delta_time))
    if extra_features == []:
        return dist_arr,speed_arr,acc_arr,jerk_arr
    # compute extra
    feats_arr = [dist_arr,speed_arr,acc_arr,jerk_arr]
    if 'delta_lat' in extra_features:
        delta_lat_arr = [traj_seg[i][0]-traj_seg[i-1][0] for i in range(1,num_pts)]
        delta_lat_arr.insert(0,0)
        feats_arr.append(delta_lat_arr)
    if 'delta_lng' in extra_features:
        delta_lng_arr = [traj_seg[i][1]-traj_seg[i-1][1] for i in range(1,num_pts)]
        delta_lng_arr.insert(0,0)
        feats_arr.append(delta_lng_arr)
    if 'delta_time' in extra_features:
        feats_arr.append(delta_time_arr)
    if 'bearing_rate' in extra_features:
        bearing_rate_arr = [0]
        for i in range(1,num_pts-1):
            bearing_rate_arr.append(compute_bearing_rate(compute_bearing(traj_seg[i-1],traj_seg[i]),
                                                        compute_bearing(traj_seg[i],traj_seg[i+1])))
        bearing_rate_arr.append(0)
        feats_arr.append(bearing_rate_arr)
    return feats_arr
    
    
def save_feats_trip(feats_trip,file_path,columns):
    '''
    Save features of a trip into a csv file.
    :param feats_trip: list of shape [n_features,n_pts],which
        need to be converted into [n_pts,n_features];
    '''
    feats = []
    for i in range(len(feats_trip[0])):
        feats_point = []
        for j in range(len(columns)):
            feats_point.append(feats_trip[j][i])
        feats.append(feats_point)
    util.save_list2csv(file_path,columns,feats)
    
    
    
def extract_features_all_user(data_dir,output_dir):
    '''
    Extract eight point-wise features of all user,but don't interpolate 
    or smooth here,which are left to next preprocess step depending 
    on demand. Total eight features can be extracted.
    '''
    users_list = os.listdir(data_dir)
    cnt_user = 0

    extra_features = ['delta_lat','delta_lng','delta_time','bearing_rate']
    columns = ['distance','speed','acc','jerk']
    columns.extend(extra_features)
    columns.append('label')
    start_time1 = time.process_time()
    for user in users_list:
        if cnt_user <= -1:
            cnt_user += 1
            continue
        start_time2 = time.process_time()
        user_data_dir = data_dir + user + '/'
        file_list = os.listdir(user_data_dir)
        if len(file_list) < 1:
            continue
        # make dir for saving 
        user_output_dir = output_dir + user + '/'
        if not os.path.exists(user_output_dir):
            os.mkdir(user_output_dir)
        # extract features of a trip
        for file in file_list:
            traj_one_trip = util.load_csv2list(user_data_dir + file)
            feats_one_trip = extract_features_seg(traj_one_trip,extra_features)
            label_one_trip = [point[3] for point in traj_one_trip]
            feats_one_trip.append(label_one_trip)
            save_feats_trip(feats_one_trip,user_output_dir + file,columns)
        print(cnt_user,' th user:',user,', consuming time:',time.process_time()-start_time2,' second.')
        cnt_user += 1
    print('Total ',cnt_user," users' trip are extracted features,consuming time:",time.process_time()-start_time1,' second')


if __name__ == "__main__":
    """ 
    
    """
    data_dir = '../data/Traj Label Trip/'
    output_dir = '../data/Feats Trip/'
    extract_features_all_user(data_dir,output_dir)