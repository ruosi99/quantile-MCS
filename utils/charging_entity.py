import pandas as pd
from datetime import datetime, timedelta
from geopy.distance import geodesic
import requests

class ChargingEntity:
    def __init__(self, name, charging_power, latitude, longitude, electricity_fee, cutoff_time = "2023-06-20 07:00:00"):
        self.name = name
        self.charging_power = charging_power
        self.latitude = latitude
        self.longitude = longitude
        self.expected_available_time = None
        self.start_time = None  # 开始处理第一个请求的时间
        self.total_occupied_time = timedelta(0)  # 总的占用时间
        self.total_time = timedelta(0)  # 总的时间间隔
        self.cutoff_time = datetime.strptime(cutoff_time, "%Y-%m-%d %H:%M:%S")
        self.electricity_fee = electricity_fee

        if 'mcs' in name.lower():
            self.entity_type = 'mcs'
        elif 'fcs' in name.lower():
            self.entity_type = 'fcs'
        else:
            self.entity_type = 'unknown'

    @property
    def occupancy_rate(self):
        if self.total_time.total_seconds() == 0:
            return 0.0
        return self.total_occupied_time.total_seconds() / self.total_time.total_seconds()

    def calculate_distance(self, request):
        request_lat = request['latitude']
        request_lon = request['longitude']
        current_location = (self.latitude, self.longitude)
        request_location = (request_lat, request_lon)
        return geodesic(current_location, request_location).kilometers

    def is_available(self, request):
        request_time = pd.to_datetime(request['request_time'])
        travel_time = self.calculate_travel_time_simple(request)
        arrival_time = request_time + timedelta(minutes=travel_time)
        return self.expected_available_time is None or self.expected_available_time <= arrival_time

    def calculate_travel_time_simple(self, request):
        # 假设车辆行驶速度为 30 公里/小时
        speed = 30
        distance = self.calculate_distance(request)
        travel_time = distance / speed * 60  # 转换为分钟
        return travel_time
    
    def calculate_travel_time_osrm(self, request):
        # Replaced OSRM API with simple approximation for stability
        return self.calculate_travel_time_simple(request)

    def process_request(self, request, is_eval = False):
        request_time = pd.to_datetime(request['request_time'])
        request_lat = request['latitude']
        request_lon = request['longitude']
        energy_needed = request['energy_needed']

        # 初始化开始时间
        if self.start_time is None:
            self.start_time = request_time

        # 计算到达耗时
        travel_time = self.calculate_travel_time_osrm(request)
        arrival_time = request_time + timedelta(minutes=travel_time)

        # 计算排队耗时
        if self.expected_available_time is None or self.expected_available_time <= arrival_time:
            queue_time = 0
        else:
            queue_time = (self.expected_available_time - arrival_time).total_seconds() / 60

        # 计算充电耗时
        charging_time = energy_needed / self.charging_power * 60  # 转换为分钟

        # 计算等待时间
        waiting_time = travel_time + queue_time + charging_time

        # 更新预计可用时间
        if self.expected_available_time is None or self.expected_available_time <= arrival_time:
            new_available_time = arrival_time + timedelta(minutes=charging_time)
        else:
            new_available_time = self.expected_available_time + timedelta(minutes=charging_time)

        if not is_eval:
            # 更新占用时间
            if self.expected_available_time is not None and self.expected_available_time < self.cutoff_time:
                self.total_occupied_time += timedelta(minutes=charging_time)
                if new_available_time > self.cutoff_time:
                    self.total_occupied_time -= (new_available_time - self.cutoff_time)
                
            # 更新总时间
            self.total_time = new_available_time - self.start_time if new_available_time < self.cutoff_time \
                else self.cutoff_time - self.start_time

            # 更新预计可用时间
            self.expected_available_time = new_available_time

            # 对于 mcs，更新位置
            if self.entity_type == 'mcs':
                self.latitude = request_lat
                self.longitude = request_lon

        return {
            'travel_time': travel_time,
            'queue_time': queue_time,
            'charging_time': charging_time,
            'waiting_time': waiting_time,
            'end_time': new_available_time
        }

    def calculate_cost(self, request):
        process_result = self.process_request(request, is_eval=True)
        travel_time = process_result['travel_time']
        queue_time = process_result['queue_time']
        charging_time = process_result['charging_time']
        vot = request['vot']
        cost = vot * (travel_time + queue_time) / 60 + charging_time/60 * self.charging_power * self.electricity_fee
        if self.entity_type == 'mcs':
            cost += 20
        return cost