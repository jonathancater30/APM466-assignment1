import numpy as np
from numpy import linalg as LA
import sympy
import math
from sympy.solvers import solve
from sympy import Symbol
from sympy import Eq
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev


class FixedIncomeBond:
    def __init__(self, isin, price_series, mat_date, coupon, num_periods):
        self.isin = isin
        self.price_series = price_series
        self.mat_date = mat_date
        self.coupon = coupon
        self.num_periods = num_periods


# Define bond price series and maturity dates
price_series_2025_03_01 = [99.87, 99.88, 99.88, 99.88, 99.88, 99.88, 99.88, 99.88, 99.88, 99.88]
mat_date_2025_03_01 = datetime.date(2025, 3, 1)
bond_2025_03_01 = FixedIncomeBond("CA135087D929", price_series_2025_03_01, mat_date_2025_03_01, 0.75, 0)

price_series_2025_06_01 = [100.70, 100.70, 100.69, 100.69, 100.68, 100.66, 100.65, 100.65, 100.64, 100.64]
mat_date_2025_06_01 = datetime.date(2025, 6, 1)
bond_2025_06_01 = FixedIncomeBond("CA135087YZ11", price_series_2025_06_01, mat_date_2025_06_01, 1.75, 0)

price_series_2025_09_01 = [99.30, 99.26, 99.30, 99.25, 99.30, 99.26, 99.30, 99.25, 99.30, 99.28]
mat_date_2025_09_01 = datetime.date(2025, 9, 1)
bond_2025_09_01 = FixedIncomeBond("CA135087E596", price_series_2025_09_01, mat_date_2025_09_01, 0.375, 1)

price_series_2026_03_01 = [98.87, 98.91, 98.93, 98.92, 98.90, 98.90, 98.86, 98.88, 98.88, 98.91]
mat_date_2026_03_01 = datetime.date(2026, 3, 1)
bond_2026_03_01 = FixedIncomeBond("CA135087F254", price_series_2026_03_01, mat_date_2026_03_01, 0.375, 2)

price_series_2026_09_01 = [98.43, 98.47, 98.51, 98.48, 98.48, 98.45, 98.45, 98.40, 98.43, 98.44]
mat_date_2026_09_01 = datetime.date(2026, 9, 1)
bond_2026_09_01 = FixedIncomeBond("CA135087F585", price_series_2026_09_01, mat_date_2026_09_01, 0.375, 3)

price_series_2027_03_01 = [97.55, 97.65, 97.64, 97.67, 97.62, 97.62, 97.59, 97.59, 97.56, 97.63]
mat_date_2027_03_01 = datetime.date(2027, 3, 1)
bond_2027_03_01 = FixedIncomeBond("CA135087G328", price_series_2027_03_01, mat_date_2027_03_01, 0.25, 4)

price_series_2028_03_01 = [100.33, 100.40, 100.50, 100.43, 100.46, 100.33, 100.33, 100.25, 100.33, 100.36]
mat_date_2028_03_01 = datetime.date(2028, 3, 1)
bond_2028_03_01 = FixedIncomeBond("CA135087H490", price_series_2028_03_01, mat_date_2028_03_01, 0.875, 5)

price_series_2029_03_01 = [102.50, 102.67, 102.73, 102.60, 102.66, 102.55, 102.45, 102.48, 102.52, 102.66]
mat_date_2029_03_01 = datetime.date(2029, 3, 1)
bond_2029_03_01 = FixedIncomeBond("CA135087J546", price_series_2029_03_01, mat_date_2029_03_01, 1.125, 6)

price_series_2029_09_01 = [98.74, 98.93, 99.31, 99.09, 99.27, 98.97, 99.05, 99.04, 99.01, 99.08]
mat_date_2029_09_01 = datetime.date(2029, 9, 1)
bond_2029_09_01 = FixedIncomeBond("CA135087J967", price_series_2029_09_01, mat_date_2029_09_01, 0.75, 7)

price_series_2030_03_01 = [98.76, 98.71, 98.70, 98.76, 98.70, 98.78, 98.77, 98.72, 98.76, 98.75]
mat_date_2030_03_01 = datetime.date(2030, 3, 1)
bond_2030_03_01 = FixedIncomeBond("CA135087K528", price_series_2030_03_01, mat_date_2030_03_01, 0.625, 8)


# Build the collection of bonds
bond_collection = []
bond_collection.append(bond_2025_03_01)
bond_collection.append(bond_2025_09_01)
bond_collection.append(bond_2026_03_01)
bond_collection.append(bond_2026_09_01)
bond_collection.append(bond_2027_03_01)
bond_collection.append(bond_2028_03_01)
bond_collection.append(bond_2029_03_01)
bond_collection.append(bond_2029_09_01)
bond_collection.append(bond_2030_03_01)


# Observation record dates
record_dates = {}
record_dates[0] = datetime.date(2025, 1, 2)
record_dates[1] = datetime.date(2025, 1, 3)
record_dates[2] = datetime.date(2025, 1, 6)
record_dates[3] = datetime.date(2025, 1, 7)
record_dates[4] = datetime.date(2025, 1, 8)
record_dates[5] = datetime.date(2025, 1, 9)
record_dates[6] = datetime.date(2025, 1, 10)
record_dates[7] = datetime.date(2025, 1, 13)
record_dates[8] = datetime.date(2025, 1, 14)
record_dates[9] = datetime.date(2025, 1, 15)


def compute_present_value(bond_obj, obs_day):
    # Calculate the dirty price component for the bond at the given observation day
    dirty_val = bond_obj.coupon * (183 - (31 - obs_day) - 29) / 183
    # The present value is the sum of the dirty price and the observed price
    pres_val = round(dirty_val + bond_obj.price_series[obs_day], 4)
    return pres_val


def compute_year_diff(start_date, end_date):
    duration_seconds = (end_date - start_date).total_seconds()
    return round(divmod(duration_seconds, 86400)[0] / 365, 3)


def compute_year_fractions(rec_date, year_frac_dict, counter=0):
    for idx in range(len(bond_collection) - 1):
        frac = compute_year_diff(rec_date, bond_collection[idx].mat_date)
        year_frac_dict[counter] = frac

        if compute_year_diff(bond_collection[idx].mat_date, bond_collection[idx+1].mat_date) > 0.6:
            later_frac = compute_year_diff(rec_date, bond_collection[idx+1].mat_date)
            year_frac_dict[counter + 1] = (later_frac + frac) / 2
            counter += 1
        counter += 1
    year_frac_dict[counter] = compute_year_diff(rec_date, bond_collection[-1].mat_date)
    return year_frac_dict


# Map the bond period indicator to actual period count
period_map = {}
period_map[0] = 0
period_map[1] = 1
period_map[2] = 2
period_map[3] = 3
period_map[4] = 4
period_map[5] = 6
period_map[6] = 8
period_map[7] = 9
period_map[8] = 10


def compute_total_present_value(face_val, coupon_amt, period_key, disc_rate, year_frac_dict):
    total_val = 0
    total_periods = period_map[period_key]

    if period_key == 0:
        total_val += face_val * math.exp(-year_frac_dict[0] * disc_rate)
    else:
        for i in range(total_periods):
            total_val += coupon_amt * math.exp(-year_frac_dict[i] * disc_rate)
        total_val += face_val * math.exp(-year_frac_dict[i+1] * disc_rate)
    return total_val


def compute_ytm(bond_obj, obs_day, year_frac_dict):
    pres_val = compute_present_value(bond_obj, obs_day)
    face_val = bond_obj.coupon + 100
    coupon_amt = bond_obj.coupon
    period_key = bond_obj.num_periods

    ytm_val1 = coupon_amt / 100
    flag1 = True
    while flag1:
        if pres_val < face_val:
            ytm_val1 -= 0.000001
        else:
            ytm_val1 += 0.000001
 
        tot_val1 = compute_total_present_value(face_val, coupon_amt, period_key, ytm_val1, year_frac_dict)
 
        if pres_val < face_val:
            flag1 = tot_val1 < pres_val
        else:
            flag1 = tot_val1 > pres_val

    ytm_val2 = coupon_amt / 100
    flag2 = True
    while flag2:
        if pres_val < face_val:
            ytm_val2 += 0.000001
        else:
            ytm_val2 -= 0.000001
 
        tot_val2 = compute_total_present_value(face_val, coupon_amt, period_key, ytm_val2, year_frac_dict)
 
        if pres_val < face_val:
            flag2 = tot_val2 > pres_val
        else:
            flag2 = tot_val2 < pres_val

    return max(ytm_val1 * 100, ytm_val2 * 100)


def generate_ytm_daily(bond_collection):
    daily_raw_ytm = {}
    daily_full_ytm = {}
    daily_year_frac = {}

    for day_idx in range(10):
        raw_ytm = []
        full_ytm = []
        year_frac_input = {}
        yr_frac = compute_year_fractions(record_dates[day_idx], year_frac_input)
        
        j = 0
        while j < len(bond_collection) - 1:
            current_bond = bond_collection[j]
            next_bond = bond_collection[j+1]
            ytm_val = compute_ytm(current_bond, day_idx, yr_frac)
            raw_ytm.append(ytm_val)
            full_ytm.append(ytm_val)
            
            if compute_year_diff(current_bond.mat_date, next_bond.mat_date) > 0.6:
                next_ytm = compute_ytm(next_bond, day_idx, yr_frac)
                avg_ytm = (next_ytm + ytm_val) / 2
                full_ytm.append(avg_ytm)
            j += 1

        last_bond = bond_collection[-1]
        last_ytm = compute_ytm(last_bond, day_idx, yr_frac)
        raw_ytm.append(last_ytm)
        full_ytm.append(last_ytm)
        print(full_ytm)
        daily_raw_ytm[day_idx] = raw_ytm
        daily_full_ytm[day_idx] = full_ytm
        daily_year_frac[day_idx] = yr_frac

    return daily_raw_ytm, daily_full_ytm, daily_year_frac


daily_raw_ytm, daily_full_ytm, daily_year_frac = generate_ytm_daily(bond_collection)

dates_list = ['Jan 2', 'Jan 3', 'Jan 6', 'Jan 7', 'Jan 8', 'Jan 9', 'Jan 10', 'Jan 13', 'Jan 14', 'Jan 15']
plt.xlabel('Time to Maturity (Year/Month)')
plt.ylabel('Yield to Maturity')
plt.title('Five Year Yield Curve')
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], labels=['25/3','25/9','26/3','26/9','27/3','27/9','28/3','28/9','29/3','29/9','30/3'])
for key in range(10):
    plt.plot(daily_full_ytm[key], label=dates_list[key])
plt.legend(loc=1, prop={'size': 6})
plt.show()


def compute_total_spot_pv(face_val, coupon_amt, period_key, disc_rate, year_frac_dict, spot_list, gap_flag):
    total_val = 0
    total_periods = period_map[period_key]

    if not gap_flag:
        for i in range(total_periods):
            total_val += coupon_amt * math.exp(-year_frac_dict[i] * (spot_list[i] / 100))
        total_val += face_val * math.exp(-year_frac_dict[i+1] * disc_rate)
    else:
        for i in range(total_periods - 1):
            total_val += coupon_amt * math.exp(-year_frac_dict[i] * (spot_list[i] / 100))
        total_val += coupon_amt * math.exp(-year_frac_dict[i+1] * ((spot_list[-1] / 100 + disc_rate) / 2))
        total_val += face_val * math.exp(-year_frac_dict[i+2] * disc_rate)

    return total_val


def compute_spot_rate(bond_obj, obs_day, bond_idx, year_frac_dict, spot_list, ytm_list, gap_flag):
    pres_val = compute_present_value(bond_obj, obs_day)
    face_val = bond_obj.coupon + 100
    coupon_amt = bond_obj.coupon
    period_key = bond_obj.num_periods
    spot_val1 = ytm_list[bond_idx] / 100

    cond1 = True
    while cond1:
        if pres_val < face_val:
            spot_val1 -= 0.000001
        else:
            spot_val1 += 0.000001
 
        tot_val1 = compute_total_spot_pv(face_val, coupon_amt, period_key, spot_val1, year_frac_dict, spot_list, gap_flag)
 
        if pres_val < face_val:
            cond1 = tot_val1 < pres_val
        else:
            cond1 = tot_val1 > pres_val

    spot_val2 = ytm_list[bond_idx] / 100
    cond2 = True
    while cond2:
        if pres_val < face_val:
            spot_val2 += 0.000001
        else:
            spot_val2 -= 0.000001

        tot_val2 = compute_total_spot_pv(face_val, coupon_amt, period_key, spot_val2, year_frac_dict, spot_list, gap_flag)
        
        if pres_val < face_val:
            cond2 = tot_val2 > pres_val
        else:
            cond2 = tot_val2 < pres_val

    return max(spot_val1 * 100, spot_val2 * 100)


def generate_spot_rate_daily(bond_collection, daily_year_frac):
    spot_rate_collection = {}
    for day_key in range(10):
        spot_list = []
        for j in range(len(bond_collection)):
            ytm_vals = daily_raw_ytm[day_key]

            if j == 0:
                spot_list.append(ytm_vals[0])
            elif j >= 1:
                prev_bond = bond_collection[j-1]
                curr_bond = bond_collection[j]
                if compute_year_diff(prev_bond.mat_date, curr_bond.mat_date) > 0.6:
                    sp_rate = compute_spot_rate(curr_bond, day_key, j, daily_year_frac[day_key], spot_list, ytm_vals, True)
                    spot_list.append((spot_list[-1] + sp_rate) / 2)
                    spot_list.append(sp_rate)
                else:
                    sp_rate = compute_spot_rate(curr_bond, day_key, j, daily_year_frac[day_key], spot_list, ytm_vals, False)
                    spot_list.append(sp_rate)
        print(spot_list)
        spot_rate_collection[day_key] = spot_list
    return spot_rate_collection

spot_rate_collection = generate_spot_rate_daily(bond_collection, daily_year_frac)


dates_list = ['Jan 2', 'Jan 3', 'Jan 6', 'Jan 7', 'Jan 8', 'Jan 9', 'Jan 10', 'Jan 13', 'Jan 14', 'Jan 15']
plt.xlabel('Time to Maturity (Year/Month)')
plt.ylabel('Spot Rate')
plt.title('Five Year Spot Rate Curve')
axes = plt.gca()
axes.set_ylim([1.4, 2.8])
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], labels=['25/3','25/9','26/3','26/9','27/3','27/9','28/3','28/9','29/3','29/9','30/3'])
for key in range(10):
    plt.plot(spot_rate_collection[key], label=dates_list[key])
plt.legend(loc=1, prop={'size': 6})
plt.show()


def compute_forward_rate(spot_rate_collection):
    forward_rate_collection = {}
    for day_key in range(len(spot_rate_collection)):
        forward_list = []
        base_rate = spot_rate_collection[day_key][2]

        j = 2
        k = 4
        while k < len(spot_rate_collection[0]):
            fwd_rate = (spot_rate_collection[day_key][k] * j - base_rate) / (j - 1)
            forward_list.append(fwd_rate)        
            j += 1
            k += 2
            
        print(forward_list)
        forward_rate_collection[day_key] = forward_list
    return forward_rate_collection

forward_rate_collection = compute_forward_rate(spot_rate_collection)


plt.xlabel('Year')
plt.ylabel('Forward Rate')
plt.title('Five Year Forward Rate Curve')
axes = plt.gca()
axes.set_ylim([1.4, 1.72])
plt.xticks(ticks=[0, 1, 2, 3], labels=['1yr-1yr','1yr-2yr','1yr-3yr','1yr-4yr'])
for key in range(10):
    plt.plot(forward_rate_collection[key], label=dates_list[key])
plt.legend(loc=1, prop={'size': 6})
plt.show()


def compute_ytm_covariance(daily_full_ytm):
    ytm_matrix = []
    for day_key in range(10):
        temp_list = []
        j = 1
        while j < len(daily_full_ytm[0]):
            temp_list.append(daily_full_ytm[day_key][j])
            j += 2
        ytm_matrix.append(temp_list)

    ytm_matrix = np.array(ytm_matrix).transpose()

    log_ret = np.zeros((5, 9))
    for i in range(len(ytm_matrix)):
        for j in range(len(ytm_matrix[i]) - 1):
            log_ret[i][j] = math.log(ytm_matrix[i][j] / ytm_matrix[i][j+1])
    return np.cov(log_ret)

ytm_covariance = compute_ytm_covariance(daily_full_ytm)
print(ytm_covariance)


def compute_forward_covariance(forward_rate_collection):
    forward_matrix = []
    for day_key in range(10):
        temp_list = []
        j = 0
        while j < len(forward_rate_collection[0]):
            temp_list.append(forward_rate_collection[day_key][j])
            j += 1
        forward_matrix.append(temp_list)

    forward_matrix = np.array(forward_matrix).transpose()

    log_ret = np.zeros((4, 9))
    for i in range(len(forward_matrix)):
        for j in range(len(forward_matrix[i]) - 1):
            log_ret[i][j] = math.log(forward_matrix[i][j] / forward_matrix[i][j+1])
    print(log_ret)
    return np.cov(log_ret)

forward_covariance = compute_forward_covariance(forward_rate_collection)
print(forward_covariance)


ytm_eigvals, ytm_eigvecs = LA.eig(ytm_covariance)
print(ytm_eigvals)
print(ytm_eigvecs)

forward_eigvals, forward_eigvecs = LA.eig(forward_covariance)
print(forward_eigvals)
print(forward_eigvecs)