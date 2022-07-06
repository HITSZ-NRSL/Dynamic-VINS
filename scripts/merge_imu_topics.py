#!/usr/bin/env python2

# Copyright (C) <2019> Intel Corporation
# SPDX-License-Identifier: MIT
# Author: Xuesong Shi

"""
merge the gyro and accel topics into one (with linear interpolation)
"""
import rosbag
import sys
from sensor_msgs.msg import Imu

def main():
    files = sys.argv[1:]
    i=0
    for file in files:
        i=i+1
        outfile = file +"-imu"+".bag"
        outbag = rosbag.Bag(outfile, 'w')
        d400_accer=[]
        d400_accer_t=[]
        d400_gyr=[]
        d400_gyr_t=[]
        with rosbag.Bag(file) as inbag:
                print ("-------------------- Input: %s ------------------------------" % file)
                print (inbag)
                for topic,msg,t in inbag.read_messages():
                    if topic=="/d400/accel/sample":
                        d400_accer.append(msg)
                        d400_accer_t.append(t)
                    if topic=="/d400/gyro/sample":
                        d400_gyr.append(msg)
                        d400_gyr_t.append(t)
                    outbag.write(topic, msg, t)
        imu0_topic="/d400/imu0"
        imu0=[]
        imu0_t=[]
        print('merging acc')
        for acc, t in zip(d400_accer, d400_accer_t):
            print("merging acc:%d",t)
            imu=acc
            imu.angular_velocity=findAngular(d400_gyr, d400_gyr_t, t)
            outbag.write(imu0_topic, imu, t)
        print("merging gyr")
        for gyr, t in zip(d400_gyr, d400_gyr_t):
            print("merging gyr:%d",t)
            imu=gyr
            imu.linear_acceleration=findGAccer(d400_accer, d400_accer_t, t)
            outbag.write(imu0_topic, imu, t)
        print("merged")
        outbag.close()
        outbag = rosbag.Bag(outfile)
        print ("------------------- Output: %s ------------------------------" % outfile)
        print (outbag)

def findAngular(d400_gyr, d400_gyr_t, target_t):
    imu=Imu()
    pre_t=d400_gyr_t[0]
    last_t=d400_gyr_t[0]
    pre_gyr=d400_gyr[0]
    last_gyr=d400_gyr[0]
    for gyr, t in zip(d400_gyr, d400_gyr_t):
        last_t=t
        last_gyr=gyr
        if target_t >=pre_t and target_t<=last_t:
            ratio=(last_t-target_t)/(last_t-pre_t)
            imu.angular_velocity.x=ratio*pre_gyr.angular_velocity.x + (1.0-ratio)*last_gyr.angular_velocity.x
            imu.angular_velocity.y=ratio*pre_gyr.angular_velocity.y + (1.0-ratio)*last_gyr.angular_velocity.y
            imu.angular_velocity.z=ratio*pre_gyr.angular_velocity.z + (1.0-ratio)*last_gyr.angular_velocity.z
            break
        elif target_t<pre_t:
            break
        else:
            pre_t=last_t
            pre_gyr=last_gyr
    return imu.angular_velocity

def findGAccer(d400_accer, d400_accer_t, target_t):
    imu=Imu()
    pre_t=d400_accer_t[0]
    last_t=d400_accer_t[0]
    pre_accer=d400_accer[0]
    last_accer=d400_accer[0]
    for accer, t in zip(d400_accer, d400_accer_t):
        last_t=t
        last_accer=accer
        if target_t >=pre_t and target_t<=last_t:
            ratio=(last_t-target_t)/(last_t-pre_t)
            imu.linear_acceleration.x=ratio*pre_accer.linear_acceleration.x + (1.0-ratio)*last_accer.linear_acceleration.x
            imu.linear_acceleration.y=ratio*pre_accer.linear_acceleration.y + (1.0-ratio)*last_accer.linear_acceleration.y
            imu.linear_acceleration.z=ratio*pre_accer.linear_acceleration.z + (1.0-ratio)*last_accer.linear_acceleration.z
            break
        elif target_t<pre_t:
            break
        else:
            pre_t=last_t
            pre_accer=last_accer
    return imu.linear_acceleration

if __name__ == '__main__':
    main()
