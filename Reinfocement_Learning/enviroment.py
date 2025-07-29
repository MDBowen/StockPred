""" Generates a real-time (or stored data) stock investment 
environment for reinforcement learning models."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import yfinance as yf
from datetime import datetime, timedelta

class StockInvestment:
    def __init__(self, yf_name, start_date=datetime.now()-timedelta(days=1), start_investement = 100):
        self.ticker = yf.Ticker(yf_name)
        self.start_date = start_date
        self.start_price = self.fetch_current_price(start_date)
        self.start_investement = start_investement
        self.start_value_ratio = self.start_investement/self.start_price
        
    def fetch_current_price(self, time = datetime.now()):
        return self.ticker.history(start=time-timedelta(days=1), end=time, interval='1d')['Close'].iloc[-1]

    def fetch_current_value(self, time = datetime.now()):
        return self.fetch_current_price(time) * self.start_value_ratio
    
    def fetch_time_series(self, start_date, end_date, interval='1d'):
        return self.ticker.history(start=start_date, end=end_date, interval=interval) 

    def update_investment_value(self, new_value):
        self.start_investement = new_value
        self.start_date = datetime.now()
        self.start_price = self.fetch_current_price()
        self.start_value_ratio = self.start_investement / self.start_price
        
# A non real time version of StockInvestment
class FauxInvestment(StockInvestment):
    def __init__(self, name, start_date=datetime.now()-timedelta(days=1), start_investement=100, intervals = '1d'):
        super().__init__(name, start_date, start_investement)
        self.current_value = self.start_investement
        self.current_date = start_date
        self.start_date = start_date
        
        
        if intervals[-1] == 'd':
            self.interval_in_mins = 60*24
        elif intervals[-1] == 'h':
            self.interval_in_mins = 60
        elif intervals[-1] == 'm':
            self.interval_in_mins = 1
        else:
            raise ValueError("Interval must end with 'd', 'h', or 'm' for days, hours, or minutes respectively.")

    def next_time_step(self):
        if self.current_date > datetime.now():
            raise ValueError("Cannot fetch data for future dates.")
        else: 
            self.current_date += timedelta(minutes=self.interval_in_mins)
            self.current_value = self.fetch_current_value(self.current_date)
            return self.current_date
    
    def fetch_current_price(self, time=None):
        date= self.current_date
        return super().fetch_current_price(time=date)
    
    def fetch_time_series(self, start_date, interval='1d'):
        return super().fetch_time_series(start_date, self.current_date, interval)
        
class Portfolio:
    def __init__(self, investments = {}, balance=0.0):
        self.investments = investments
        self.balance = balance
        
    def add_investment(self, investiment, name = 'undefined'):
        self.investments[name]=investiment
        
    def fetch_investment_value(self, name):
        if name in self.investments:
            return self.investments[name].fetch_current_value()
        else:
            raise ValueError(f"Investment '{name}' not found in portfolio.")

    def update_investment(self, name, updated_value):
        previous_value = self.fetch_investment_value(name)
        if name in self.investments:
            self.investments[name].update_investment_value(updated_value)
        else:
            raise ValueError(f"Investment '{name}' not found in portfolio.")

        profit = previous_value - self.fetch_investment_value(name)
        self.balance += profit

        print(f"Updated investment '{name}': Previous Value: {previous_value}, New Value: {self.fetch_investment_value(name)}, Profit: {profit}, Balance: {self.balance}")

    def get_full_net_worth(self):
        net = self.balance
        for name, investment in self.investments.items():
            net += self.fetch_investment_value(name)
        return net
    
    def get_timeseries(self,start_date, end_date, name = 'GOOG', interval='1d'):
        if name in self.investments:
            return self.investments[name].fetch_time_series(start_date, end_date, interval)
        else:
            raise ValueError(f"Investment '{name}' not found in portfolio.")
        
        

class Simulate_Portfolio:
    def __init__(self, init_datetime, balance = 1000.0):
        
        
        self.investments_timesteps = {} 
        self.investments_future_steps = {}
        # such that investment_timesteps[name][-1] is current close
        self.investments = {}
        self.investments_states = {} # :: Name : [price, holdings]
        self.datetime = init_datetime
        self.init_date = init_datetime
        self.timestep = 0 
        self.balance = balance
        self.init_balane = balance
        
    def add_investment(self, name):
        
        self.investments[name] = yf.Ticker(name)
        self.investments_timesteps[name] = [self.investments[name].history(start=self.datetime-timedelta(days=1), end=self.datetime, interval='1d')['Close'].iloc[-1]]
        self.investments_future_steps[name] = self.investments[name].history(start = self.datetime, end = datetime.now(), interval = '1d')['Close']
        self.investments_states[name] = np.array([self.investments_timesteps[name][-1], 0])

    def get_current_timestep(self, ticker):
        return ticker.history(start=self.datetime-timedelta(days=10), end=self.datetime, interval='1d')['Close'].iloc[-1]
        
    def get_current_timestep_sim(self, name):
        
        df = self.investments_future_steps[name]
        
        dtime = self.datetime
        
        while True:
            try:
                val = df.loc[dtime.strftime('%Y-%m-%d')]
            except:
                dtime += timedelta(days=1)
            else:
                return val, dtime
            
        
        
    def next_timestep(self):
        
        self.datetime += timedelta(days=1)
        
        for name, ticker in self.investments.items():
            previous_state = self.investments_states[name]
            timestep, date = self.get_current_timestep_sim(name)
            
            self.datetime = date
            
            self.investments_timesteps[name].append(timestep)

            current_state =  timestep * previous_state/previous_state[0]
            #print(f"prev_state: {previous_state},  timestep: {timestep},  new_state:{current_state}")
            self.investments_states[name] = current_state
            
    def update_investment(self, name, value):
        
        if value<=self.balance:
            self.investments_states[name][1] += value
            self.balance -= value
        else:
            raise ValueError(f"Attempted investment exceeds porfolio's balance, balance:{self.balance}, attempted investment: {value}")
        
    def get_portfolio_net(self):
        net = self.balance
        for values in self.investments_states.values():
            net += values[1]
            
        return net
    
    def calc_profit(self):
        return self.get_portfolio_net() - self.init_balane
    
    def return_state(self, name):
        # :: name/str -> np.array( price, holdings, balance  )
        return np.concatenate([self.investments_states[name], [self.balance]])

    def reset_to_start(self):
        """Resets the portfolio"""
        
        self.investments_timesteps = {} 
        self.investments = {}
        self.investments_future_steps = {}
        self.investments_states = {}
        self.datetime = self.init_date 
        self.timestep = 0 
        self.balance = self.init_balane
        
