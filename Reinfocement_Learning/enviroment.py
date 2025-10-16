""" Generates a real-time (or stored data) stock investment 
environment for reinforcement learning models."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import yfinance as yf
from datetime import datetime, timedelta

class Simulate_Portfolio:
    def __init__(self, init_datetime, balance = 100.0, interval = 24):
        
        self.timestep_hours = interval
        
        if interval>=24:
            self.interval_str = '1d'
        else:
            self.interval_str = f'{interval}h'
        self.investments_timesteps = {} 
        self.investments_future_steps = {}
        # such that investment_timesteps[name][-1] is current close
        self.investments = {}
        self.investments_states = {} # :: Name : [price, holdings]
        self.datetime = init_datetime
        self.init_date = init_datetime
        self.timestep = 0 
        self.balance = balance
        self.init_balance = balance
        
    def add_investment(self, name):
        
        self.investments[name] = yf.Ticker(name)
        self.investments_timesteps[name] = [self.investments[name].history(start=self.datetime-timedelta(days=1), end=self.datetime, interval=self.interval_str)['Close'].iloc[-1]]
        self.investments_future_steps[name] = self.investments[name].history(start = self.datetime, end = datetime.now(), interval = self.interval_str)['Close']
        self.investments_states[name] = np.array([self.investments_timesteps[name][-1], 0])

    def get_current_timestep(self, ticker):
        return ticker.history(start=self.datetime-timedelta(hours=self.timestep_hours*5), end=self.datetime, interval=self.interval_str)['Close'].iloc[-1]

    def get_current_timestep_sim(self, name):
        
        df = self.investments_future_steps[name]
        
        dtime = self.datetime
        
        while True:
            try:
                val = df.loc[dtime.strftime('%Y-%m-%d')]
            except:
                dtime += timedelta(hours=self.timestep_hours)
            else:
                return val, dtime
        
    def next_timestep(self):
        
        self.datetime += timedelta(hours=self.timestep_hours)
        
        for name, ticker in self.investments.items():
            previous_state = self.investments_states[name]
            timestep, date = self.get_current_timestep_sim(name)
            
            self.datetime = date
            
            self.investments_timesteps[name].append(timestep)

            current_state =  timestep * previous_state/previous_state[0]
            #print(f"prev_state: {previous_state},  timestep: {timestep},  new_state:{current_state}")
            self.investments_states[name] = current_state
            
    def update_investment(self, name, value):
        
        value = min(value, self.balance + self.investments_states[name][1]) # cannot invest more than total portfolio value
        value = max(value, 0) # cannot have negative investment
        
        if value<=(self.balance + self.investments_states[name][1]): # check if new value is less or equal to balance + current holding
            prev_holding = self.investments_states[name][1]
            
            
            if self.balance + prev_holding - value < 0:
                raise ValueError(f"Value update error, value: {value}, prev_holding: {prev_holding}, balance: {self.balance}")  
            
            self.investments_states[name][1] = value
            self.balance += prev_holding - value # subtract increase in holding from balance so the increase is paid
        else:
            raise ValueError(f"Attempted investment exceeds porfolio's total, total:{self.balance + self.investments_states[name][1]}, attempted investment: {value}")
        
    def get_portfolio_net(self):
        net = self.balance
        for values in self.investments_states.values():
            net += values[1]
            
        return net
    
    def calc_profit(self):
        return self.get_portfolio_net() - self.init_balance
    
    def return_state(self, name):
        # :: name/str -> np.array( price, holdings, balance  )
        return np.concatenate([self.investments_states[name], [self.balance]])
    
    def return_all_states(self):
        states = []
        for name in self.investments_states.keys():
            states.append(self.investments_states[name])
        return np.array(states)

    def reset_to_start(self):
        """Resets the portfolio"""
        
        self.investments_timesteps = {} 
        self.investments = {}
        self.investments_future_steps = {}
        self.investments_states = {}
        self.datetime = self.init_date 
        self.timestep = 0 
        self.balance = self.init_balance
