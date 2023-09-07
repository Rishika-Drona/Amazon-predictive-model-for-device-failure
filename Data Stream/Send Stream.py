#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Generating a stream to Kafka from  the predictibe maintenance file.

import argparse
import csv
import json
import sys
import time
from dateutil.parser import parse
from confluent_kafka import Producer
import socket


# In[2]:


def acked(err, data):
    if err is not None:
        print("Failed to deliver data: %s: %s" % (str(data.value()), str(err)))
    else:
        print("Data generated: %s" % (str(data.value())))


# In[3]:


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filename', type=str,
                        help='predictive_maintenance csv file.')
    parser.add_argument('topic', type=str,
                        help='Data to stream')
    parser.add_argument('--speed', type=float, default=1, required=False,
                        help='Speed up time series of predictive maintenance by the factor.')
    args = parser.parse_args()

    topic = args.topic
    p_key = args.filename

    conf = {'bootstrap.servers': "localhost:9092",
            'client.id': socket.gethostname()}
    producer = Producer(conf)

    rdr = csv.reader(open(args.filename))
    next(rdr)  
    firstline = True


# In[ ]:




