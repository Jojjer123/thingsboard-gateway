# To specifically ignore ConvergenceWarnings:
import warnings

from collections import defaultdict
from functools import partial
from json import load
from os import path
import numpy as np
from tsmoothie import ExponentialSmoother, ConvolutionSmoother
import datetime
import threading
from time import sleep

import logging
import logging.config
import logging.handlers

log = logging.getLogger('service')
logclean = logging.getLogger("cleaning")



class DataCleaning:
	indexOfDeviceName = 0
	indexOfSensorName = 1
	indexOfTelemetry = 2
	indexOfTimeSeries = 3
	indexOfSeries = 4
	obviousOutlier = 999999
	cleaningConfig = None

	def __init__(self):
		# start thread
		t = threading.Thread(target=self.get_cleaning_config)
		t.start()

	def createDevice(self, deviceList, data):	
		try:
			newDeviceQueue = []
			# add each sensor in incoming device to sub array in list_of_devices (that exists in tb_gateway_service.py)
			for telemetry in data["telemetry"][0]["values"]:
				newDeviceQueue.append(self.getTelemetryData(data, telemetry))
			return newDeviceQueue
		except Exception as e:
			log.exception(e)
		
	def doesDeviceExist(self, deviceList, data):
		try:
			index = -1
			# if device exists in list_of_devices (in tb_gateway_service.py) return that index, otherwise return -1
			for i in range(len(deviceList)):
				if data["deviceName"] == deviceList[i][0][self.indexOfDeviceName]:
					index = i
			return index
		except Exception as e:
			log.exception(e)

	def getTelemetryData(self, data, telemetry):
		try:
			# create sub arrays
			deviceArray = []
			telemetryArray = []
			timeseriesArray = []

			# add name of device and name sensor
			deviceArray.append(data["deviceName"])
			deviceArray.append(telemetry)

			# add data point and the timestamp
			telemetryArray.append(self.check_type(data["telemetry"][0]["values"][telemetry]))			# return obvious deviation if data point is NaN
			timeseriesArray.append(data["telemetry"][0]["ts"])

			# add a dictionary in which cleaning related data is stored
			series = defaultdict(partial(np.ndarray, shape=(1, 1), dtype='float32'))

			deviceArray.append(telemetryArray)
			deviceArray.append(timeseriesArray)
			deviceArray.append(series)

			return deviceArray
		except Exception as e:
			log.exception(e)

	def addTelemetry(self, deviceList, data, deviceIndex):
		try:
			i = 0

			for telemetry in data["telemetry"][0]["values"]:
				series = deviceList[deviceIndex][i][self.indexOfSeries]

				# Control data type for observed value
				data_point = self.check_type(data["telemetry"][0]["values"][telemetry])
				# get method and other params from cleaning config
				cleaningMethod, window_len, std = self.get_cleaning_method(deviceList, deviceIndex, telemetry)
				debug_data = [0]

				# controls size of list
				while (len(deviceList[deviceIndex][i][self.indexOfTelemetry]) > window_len):
					self.removeFirstElements(deviceList, deviceIndex, telemetry)

				# cleaning begins when length is greater than specified windows_len
				if (len(deviceList[deviceIndex][i][self.indexOfTelemetry]) >= window_len):
					if(cleaningMethod == "exponentialSmoother"):
						data_point, debug_data = self._exponentialSmoother(data_point, series, window_len, std)
					elif(cleaningMethod == "convolutionSmoother"):
						data_point, debug_data = self._convolutionSmoother(data_point, series, window_len, std)

					if debug_data[0] == 1:
						time_of_cleaning = datetime.datetime.utcfromtimestamp(int(data["telemetry"][0]["ts"]) / 1000).strftime('%Y-%m-%d %H:%M:%S')
						strerror = "# -- Outlier detected - " + str(time_of_cleaning)+ " " + str(deviceList[deviceIndex][i][self.indexOfDeviceName])+ " " + str(deviceList[deviceIndex][i][self.indexOfSensorName]) +\
								   "\t\tObserved value- " + str(debug_data[1]) + " Lower boundary- " + str(debug_data[2]) + " Upper boundary- " + str(debug_data[3]) + " Corrected value to- " + str(debug_data[4])
						logclean.debug(str(strerror))


				series['original'] = np.insert(series['original'], series['original'].size, [[data_point]])

				if series['original'].size > window_len * 2:
					series['original'] = series['original'][series['original'].size - window_len * 2:]

				# add data point with timestamp to time series
				deviceList[deviceIndex][i][self.indexOfTelemetry].append(data_point)
				deviceList[deviceIndex][i][self.indexOfTimeSeries].append(data["telemetry"][0]["ts"])  # adding timeseries

				i += 1


		except Exception as e:
			log.exception(e)
	
	def removeFirstElements(self, deviceList, deviceIndex, telemetry):
		try:
			# remove first data point with its' timestamp
			i = 0
			for sensor in deviceList[deviceIndex]:
				if sensor[1] == telemetry:
					break
				else:
					i += 1
			deviceList[deviceIndex][i][self.indexOfTelemetry].pop(0)
			deviceList[deviceIndex][i][self.indexOfTimeSeries].pop(0)
		except Exception as e:
			log.exception(e)

	def check_type(self, data_point):
		# check if data point is valid type, if not make it valid but obvious outlier
		if isinstance(data_point, str):
			if data_point == "NaN":
				return self.obviousOutlier
			else:
				return float(data_point)
		else:
			return data_point

	def get_cleaning_config(self):
		try:
			# loads cleaning.json to update current cleaning config
			config_file = path.abspath("thingsboard_gateway/config/cleaning.json")
			with open(config_file) as conf:
				self.cleaningConfig = load(conf)
		except Exception as e:
			log.exception(e)
		sleep(60)
		self.get_cleaning_config()
	
	def check_if_cleaning_is_specified_for_all(self, deviceList):
		try:
			# checks if cleaning is specified
			i = 0
			for mpoint in deviceList:
				j = 0
				for attribute in deviceList[i][0]:
					if(isinstance(attribute, str) and j == 0):
						if(not self.check_if_cleaning_is_specified(attribute)):
							exceptionString = "Cleaning.json does does not specify cleaning for the endpoint " + attribute
							log.debug(exceptionString)
					j += 1
				i += 1
		except Exception as e:
			log.exception(e)

	def check_if_cleaning_is_specified(self, attribute):
		try:
			# checks if cleaning is specified for a specific device
			i = 0
			for mpoint in self.cleaningConfig["devicesWithCleaning"]:
				if(self.cleaningConfig["devicesWithCleaning"][i]["datatypeName"] != ""):
					nameOfDevice = self.cleaningConfig["devicesWithCleaning"][i]["mpointName"] + ", " + self.cleaningConfig["devicesWithCleaning"][i]["datatypeName"]
				else:
					nameOfDevice = self.cleaningConfig["devicesWithCleaning"][i]["mpointName"]
				if(nameOfDevice == attribute):
					return True

				i += 1
			return False
		except Exception as e:
			log.exception(e)
	
	def get_cleaning_method(self, deviceList, deviceIndex, telemetry):
		try:
			i = 0
			for sensor in deviceList[deviceIndex]:
				if sensor[1] == telemetry:
					break
				else:
					i += 1

			# returns the current cleaning method
			deviceNameToCheck = deviceList[deviceIndex][i][self.indexOfDeviceName]

			i = 0
			for mpoint in self.cleaningConfig["devicesWithCleaning"]:
				if (self.cleaningConfig["devicesWithCleaning"][i]["datatypeName"] != ""):
					nameOfDevice = self.cleaningConfig["devicesWithCleaning"][i]["mpointName"] + ", " + self.cleaningConfig["devicesWithCleaning"][i]["datatypeName"]
				else:
					nameOfDevice = self.cleaningConfig["devicesWithCleaning"][i]["mpointName"]
				if (nameOfDevice == deviceNameToCheck):
					j = 0
					for specificCleaning in self.cleaningConfig["devicesWithCleaning"][i]["sensorsWithSpecificCleaning"]:
						if (telemetry == specificCleaning["sensorName"] and j == 0):
							return specificCleaning["cleaningMethod"], specificCleaning["windowLen"], specificCleaning["standardDeviation"]
						j += 1
					break
				i += 1
			return self.cleaningConfig["devicesWithCleaning"][i]["defaultCleaning"], \
				   self.cleaningConfig["devicesWithCleaning"][i]["defaultWindowLen"], self.cleaningConfig["devicesWithCleaning"][i]["defaultStandardDeviation"]
		except Exception as e:
			log.exception(e)

	def _exponentialSmoother(self, data_point, series, window_len, std):
		# exponential smoothing algorithm
		smoother = ExponentialSmoother(window_len=window_len // 2, alpha=0.4)
		smoother.smooth(series['original'][-window_len:])

		series['smooth'] = np.insert(series['smooth'], series['smooth'].size, smoother.smooth_data[-1][-1])

		_low, _up = smoother.get_intervals('sigma_interval', n_sigma=std)
		series['low'] = np.insert(series['low'], series['low'].size, _low[-1][-1])
		series['up'] = np.insert(series['up'], series['up'].size, _up[-1][-1])

		debug_data = [0]

		if data_point > series['up'][-1]:
			debug_data = [1, data_point, series['low'][-1], series['up'][-1], series['up'][-1]]
			data_point = series['up'][-1]
			series['original'][-1] = data_point

		elif data_point < series['low'][-1]:
			debug_data = [1, data_point, series['low'][-1], series['up'][-1], series['low'][-1]]
			data_point = series['low'][-1]
			series['original'][-1] = data_point


		if series['smooth'].size > window_len:
			series['smooth'] = series['smooth'][series['smooth'].size - window_len:]
			series['low'] = series['low'][series['low'].size - window_len:]
			series['up'] = series['up'][series['up'].size - window_len:]

		return data_point, debug_data

	def _convolutionSmoother(self, data_point, series, window_len, std):
		smoother = ConvolutionSmoother(window_len=window_len, window_type='ones')
		smoother.smooth(series['original'][-window_len:])

		series['smooth'] = np.insert(series['smooth'], series['smooth'].size, smoother.smooth_data[-1][-1])

		_low, _up = smoother.get_intervals('sigma_interval', n_sigma=std)
		series['low'] = np.insert(series['low'], series['low'].size, _low[-1][-1])
		series['up'] = np.insert(series['up'], series['up'].size, _up[-1][-1])

		debug_data = [0]

		if data_point > series['up'][-1]:
			debug_data = [1, data_point, series['low'][-1], series['up'][-1], series['up'][-1]]
			data_point = series['up'][-1]
			series['original'][-1] = data_point
		elif data_point < series['low'][-1]:
			debug_data = [1, data_point, series['low'][-1], series['up'][-1], series['low'][-1]]
			data_point = series['low'][-1]
			series['original'][-1] = data_point

		if series['smooth'].size > window_len:
			series['smooth'] = series['smooth'][series['smooth'].size - window_len:]
			series['low'] = series['low'][series['low'].size - window_len:]
			series['up'] = series['up'][series['up'].size - window_len:]

		return data_point, debug_data
