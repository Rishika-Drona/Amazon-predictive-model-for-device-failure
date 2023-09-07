#!/usr/bin/env python
# coding: utf-8

# In[6]:


import argparse
import json
import sys
import time
import socket


# In[7]:


pip install confluent_kafka


# In[2]:


from confluent_kafka import Consumer, KafkaError, KafkaException


# In[14]:


def data_process(data):
    
    # Printing the current time and that relevant data.
    time_start = time.starttime("%Y-%m-%d %H:%M:%S")
    val = data.value()
    dval = json.loads(val)
    print(time_start, dval)


# In[15]:


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('topic', type=str,
                        help='Data to stream')

    args = parser.parse_args()

    conf = {'bootstrap.servers': 'localhost:9092',
            'default.topic.config': {'auto.offset.reset': 'smallest'},
            'group.id': socket.gethostname()}
    
    consumer = Consumer(conf)

    running = True


# In[25]:


try:
    while running:
            consumer.subscribe([args.topic])

            data = consumer.poll(1)
            if data is None:
                continue

            if data.error():
                if data.error().code() == KafkaError._PARTITION_EOF:
                   
                # Ending of partition event
                    sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                     (data.topic(), data.partition(), data.offset()))
                elif data.error().code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
                    sys.stderr.write('Topic unknown, creating %s topic\n' %
                                     (args.topic))
                elif data.error():
                    raise KafkaException(data.error())
            else:
                data_process(data)

except KeyboardInterrupt:
        pass

    finally:
        # Closing down consumer to commit final offsets.
        consumer.close()


if __name__ == "__main__":
    main()


# In[ ]:




