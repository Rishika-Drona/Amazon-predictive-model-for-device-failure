# Amazon-predictive-model-for-device-failure

A company has a fleet of devices transmitting daily telemetry readings. They would like to create a
predictive maintenance solution to proactively identify when maintenance should be performed. This
approach promises cost savings over routine or time-based preventive maintenance, because tasks are
performed only when warranted.

# GOAL
# Part 1: 
building a predictive model using machine learning to predict the probability
of a device failure. When building this model, be sure to minimize false positives and false negatives.
The column you are trying to Predict is called failure with binary value 0 for non-failure and 1 for failure.
Please consider the following points for this part:
1. Clean Python or R Code
2. Data Exploration
3. Train/Test Splitting based on devices (device level)
4. Hyperparameter Tuning
5. Multiple Models Compared
6. Minimize False Positives and False Negatives
7. Model Performance
   
# Part2: 
Create a provider-consumer method to show how you can productionize your model.

First, generate stream data ( here is a sample code but you can use any other code:

https://towardsdatascience.com/make-a-mock-real-time-stream-of-data-with-python-and-kafka-
7e5e23123582 ). As data is limited repeat the old data.

Second, code should receive data from generator and predict the failure No model performance
evaluation is needed in this step.

# DATA
Download link: http://aws-proserve-data-science.s3.amazonaws.com/predictive_maintenance.csv
