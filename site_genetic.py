import pandas as pd
import geopandas as gpd
import ast
import transbigdata as tbd  # 假设tbd库有必要的GPS_to_grid和area_to_grid函数
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms


# 定义处理充电站数据的函数
def process_charging_orders(station_info, step_length, params):
    station_info_table = station_info[
        ['station_id', 'lon', 'lat', 'max_capacity', 'charge_speed_station']].drop_duplicates().copy()

    station_info = station_info[station_info['num_current_car'] > 0]
    station_info['time'] = pd.to_datetime(station_info['time'])
    station_info['current_car'] = station_info['current_car'].apply(lambda a: ast.literal_eval(a))
    station_info['waiting_car'] = station_info['waiting_car'].apply(lambda a: ast.literal_eval(a))
    station_info.sort_values(by=['station_id', 'time'], inplace=True)

    current_car_infos = station_info[['station_id', 'time', 'current_car']].explode('current_car')
    current_car_infos = current_car_infos[~current_car_infos['current_car'].isnull()]
    current_car_infos = current_car_infos.sort_values(by=['current_car', 'time'])[['current_car', 'time', 'station_id']]

    waiting_car_infos = station_info[['station_id', 'time', 'waiting_car']].explode('waiting_car')
    waiting_car_infos = waiting_car_infos[~waiting_car_infos['waiting_car'].isnull()]
    waiting_car_infos = waiting_car_infos.sort_values(by=['waiting_car', 'time'])[['waiting_car', 'time', 'station_id']]

    return station_info_table, current_car_infos, waiting_car_infos


# 计算充电订单
def get_charging_order(station_info_table, current_car_infos, step_length):
    current_car_infos['timegap'] = current_car_infos['time'].diff().dt.total_seconds().fillna(1000000).astype(int)
    current_car_infos['order_id'] = (current_car_infos['timegap'] > step_length).cumsum()
    charge_info_s = current_car_infos.groupby(['current_car', 'order_id']).first().reset_index()
    charge_info_e = current_car_infos.groupby(['current_car', 'order_id']).last().reset_index()
    charging_order = pd.merge(charge_info_s, charge_info_e, on=['current_car', 'order_id', 'station_id'])
    charging_order = charging_order[['current_car', 'order_id', 'time_x', 'time_y', 'station_id']]
    charging_order.columns = ['carid', 'order_id', 'stime', 'etime', 'station_id']
    charging_order['duration'] = (charging_order['etime'] - charging_order['stime']).dt.total_seconds()
    charging_order = charging_order[charging_order['duration'] > 0]
    charging_orders = pd.merge(charging_order, station_info_table)
    # 计算充电时长
    charging_orders['duration'] = (charging_orders['etime'] - charging_orders['stime']).dt.total_seconds()
    return charging_orders


def get_order(charging_orders, params, grid):
    charging_order = charging_orders
    charging_order['LONCOL'], charging_order['LATCOL'] = tbd.GPS_to_grid(charging_order['lon'], charging_order['lat'],
                                                                         params=params)
    charging_order_agg = charging_order.groupby(['LONCOL', 'LATCOL'])['carid'].count().reset_index()
    charging_order_agg = pd.merge(grid[['LONCOL', 'LATCOL']], charging_order_agg, on=['LONCOL', 'LATCOL'], how="left")[
        ['LONCOL', 'LATCOL', 'carid']]

    charging_order_agg.fillna(0, inplace=True)
    return charging_order_agg


# 处理潜在充电需求
def process_potential_demand(car_infos, step_length, params):
    car_infos['time'] = pd.to_datetime(car_infos['time'])
    stay_infos = car_infos.sort_values(by=['carid', 'time'])
    stay_infos['timegap'] = (-stay_infos['time'].diff(-1).dt.total_seconds()).fillna(1000000).astype(int)
    stay_infos['etime'] = stay_infos['time'].shift(-1)
    stay_order = stay_infos[stay_infos['timegap'] > step_length][['carid', 'time', 'etime', 'soc', 'lon', 'lat']]
    stay_order.columns = ['carid', 'stime', 'etime', 'soc', 'lon', 'lat']
    stay_order['duration'] = (stay_order['etime'] - stay_order['stime']).dt.total_seconds()
    stay_order = stay_order[(stay_order["soc"] <= 50) & (stay_order["duration"] >= 60 * 5)]
    stay_order['LONCOL'], stay_order['LATCOL'] = tbd.GPS_to_grid(stay_order['lon'], stay_order['lat'], params=params)
    stay_order_agg = stay_order.groupby(['LONCOL', 'LATCOL']).size().reset_index()
    stay_order_agg.columns = ['LONCOL', 'LATCOL', "pdemand"]
    return stay_order_agg


# 处理 POI 数据
def process_poi(poi_data, grid, params, poi_types=['停车场', '加油站']):
    poi_data = poi_data[poi_data["pname"] == "上海市"]
    poi_data['ttype'] = poi_data['type'].apply(lambda x: x.split(';')[0])
    poi_data["lon"] = poi_data['location'].apply(lambda x: x.split(',')[0])
    poi_data["lat"] = poi_data['location'].apply(lambda x: x.split(',')[1])
    poi_data = poi_data[poi_data["type"].str.contains('|'.join(poi_types))]
    poi_data["lon"] = poi_data["lon"].astype("float")
    poi_data["lat"] = poi_data["lat"].astype("float")
    poi_data['LONCOL'], poi_data['LATCOL'] = tbd.GPS_to_grid(poi_data['lon'], poi_data['lat'], params=params)
    poi_agg = poi_data.groupby(['LONCOL', 'LATCOL']).size().reset_index()
    poi_agg = pd.merge(poi_agg, grid[['LONCOL', 'LATCOL']], on=['LONCOL', 'LATCOL'], how="right")
    poi_agg.columns = ['LONCOL', 'LATCOL', "park"]
    poi_agg.fillna(0, inplace=True)
    return poi_agg


# 处理充电站的利用率
def calculate_utilization(charging_orders, grid, params):
    """
    计算充电站的充电需求满足度（利用率）。

    参数:
    - charging_orders: 包含充电订单的DataFrame，至少应包含 ['stime', 'station_id', 'lon', 'lat', 'max_capacity', 'duration']
    - grid: 栅格数据，用于进行GPS坐标转换和聚合。
    - params: 栅格化参数。

    返回:
    - station_chargetime_agg: DataFrame, 包含聚合后的利用率数据，按栅格坐标（LONCOL, LATCOL）。
    """
    # 确定充电订单中的最早和最晚时间
    start_time = charging_orders["stime"].min()
    end_time = charging_orders["stime"].max()
    duration = end_time - start_time

    # 计算每个充电站的充电时长（duration）
    station_chargetime = charging_orders.groupby(["station_id", "lon", "lat", "max_capacity"])[
        "duration"].sum().reset_index()

    # 转换为timedelta格式
    station_chargetime["duration"] = pd.to_timedelta(station_chargetime["duration"], unit='s')

    # 计算充电站的利用率
    station_chargetime["uti"] = station_chargetime["duration"] / (duration * station_chargetime["max_capacity"])

    # 将站点的经纬度转换为栅格坐标
    station_chargetime['LONCOL'], station_chargetime['LATCOL'] = tbd.GPS_to_grid(station_chargetime['lon'],
                                                                                 station_chargetime['lat'],
                                                                                 params=params)

    # 按照栅格（LONCOL, LATCOL）聚合充电站利用率
    station_chargetime_agg = station_chargetime.groupby(['LONCOL', 'LATCOL']).mean().reset_index()

    # 将聚合后的利用率数据与栅格进行合并
    station_chargetime_agg = \
    pd.merge(station_chargetime_agg, grid[['LONCOL', 'LATCOL']], on=['LONCOL', 'LATCOL'], how="right")[
        ['LONCOL', 'LATCOL', 'uti']]

    # 处理空值情况
    station_chargetime_agg.fillna(0, inplace=True)

    return station_chargetime_agg


# 计算建站成本
def process_station_cost(price_data, grid, params):
    price_data["geometry"] = gpd.points_from_xy(price_data["lon"], price_data["lat"])
    price_data = gpd.GeoDataFrame(price_data, geometry=price_data["geometry"])
    price_data.crs = "EPSG:4326"
    price_data = price_data.to_crs("EPSG:32651")
    buffer = price_data.buffer(1000)
    price_data = gpd.GeoDataFrame(price_data, geometry=buffer)
    pricegrid = grid.to_crs("EPSG:32651")
    pricegrid = gpd.sjoin(pricegrid, price_data)
    pricegrid = pricegrid.groupby(["LONCOL", "LATCOL"])["price"].mean().reset_index()

    pricegrid["price"] = pricegrid["price"] * 100 * 0.02 * 0.1 * 20 + 200000 + 400000 + 0.4 * (
                pricegrid["price"] * 100 * 0.02 * 0.1 * 20 + 200000 + 400000)
    pricegrid = pd.merge(pricegrid, grid, how="right", on=['LONCOL', 'LATCOL'])
    pricegrid.fillna(pricegrid["price"].min(), inplace=True)
    pricegrid = pricegrid[['LONCOL', 'LATCOL', "price"]]
    return pricegrid


# 整合栅格
def merge_grid(station_info_path, taz_path, car_infos_path, poi_path, price_path, gridfile_path, gridgejson_path):
    # 加载数据
    station_info = pd.read_csv(station_info_path)
    taz = gpd.read_file(taz_path)
    car_infos = pd.read_csv(car_infos_path)
    poi = pd.read_excel(poi_path)
    price = pd.read_csv(price_path)
    gridfile = gridfile_path
    gridgejson = gridgejson_path

    # 设置栅格化参数
    paramssh = {'slon': 120.88125, 'slat': 30.7125, 'deltalon': 0.0125, 'deltalat': 0.008333, 'theta': 0,
                'method': 'rect', 'gridsize': 1000}
    grid, paramssh = tbd.area_to_grid(taz, params=paramssh)

    # 处理订单数据
    step_length = 5 * 60
    station_info_table, current_car_infos, waiting_car_infos = process_charging_orders(station_info, step_length,
                                                                                       paramssh)
    charging_orders = get_charging_order(station_info_table, current_car_infos, step_length)
    # 处理充电需求数据
    charging_order_agg = get_order(charging_orders, paramssh, grid)
    # 处理潜在充电需求
    stay_order_agg = process_potential_demand(car_infos, step_length, paramssh)
    # 处理POI
    poi_agg = process_poi(poi, grid, paramssh)

    # 处理充电站利用率
    station_chargetime_agg = calculate_utilization(charging_orders, grid, paramssh)

    # 计算建站成本
    pricegrid = process_station_cost(price, grid, paramssh)

    gridsum = pd.merge(charging_order_agg, stay_order_agg, on=['LONCOL', 'LATCOL'])
    gridsum = pd.merge(gridsum, poi_agg, on=['LONCOL', 'LATCOL'])
    gridsum = pd.merge(gridsum, station_chargetime_agg, on=['LONCOL', 'LATCOL'])
    gridsum = pd.merge(gridsum, pricegrid, on=['LONCOL', 'LATCOL'])

    gridsum.columns = ['LONCOL', 'LATCOL', "demand", "pdemand", "park", "uti", "price"]

    gridsum.to_csv(gridfile, index=False)

    grid.to_file(gridgejson)

    # 数据汇总（如果需要后续处理，可以在这里继续处理或保存）
    return gridsum


# 数据预处理函数
def preprocess_data(gridsum):
    df = gridsum.copy()
    df = df[df["uti"] <= 0.5]  # 保证利用率 <= 0.5
    df = df[df["park"] >= 1]  # 保证停车位数 >= 1
    df.reset_index(drop=True, inplace=True)
    df["demand"] = df["demand"].astype(float)
    df["pdemand"] = df["pdemand"].astype(float)
    return df


# 目标函数：只考虑充电需求和潜在充电需求
def objective_function(individual, df):
    selected_indices = [i for i in range(len(individual)) if individual[i] == 1]
    total_score = 0
    for i in selected_indices:
        D_it = df.at[i, 'demand']
        P_it = df.at[i, 'pdemand']
        total_score += (D_it + P_it)
    return total_score


# 约束条件检查函数
def satisfies_constraints(individual, df, max_cost, max_sites):
    selected_indices = [i for i in range(len(individual)) if individual[i] == 1]
    if not selected_indices or len(selected_indices) != max_sites:
        return False  # 如果没有选中的栅格或选中的栅格数不等于max_sites，则返回不满足约束

    total_cost = df.iloc[selected_indices]['price'].sum()

    # 条件1：用地约束
    land_availability = all(df.iloc[selected_indices]['park'] > 0)

    # 条件2：充电需求满足度约束
    demand_satisfaction = all(df.iloc[selected_indices]['uti'] <= 0.5)

    # 条件3：建站成本约束
    cost_constraint = total_cost <= max_cost

    return land_availability and demand_satisfaction and cost_constraint


# 评价函数
def evaluate(individual, df, max_cost, max_sites):
    if satisfies_constraints(individual, df, max_cost, max_sites):
        return objective_function(individual, df),
    else:
        return 0.0,  # 不满足约束条件的个体适应度设为0


# 初始化个体的函数
def init_individual(icls, df, size, num_ones, target_cost):
    individual = [0] * size
    df_sorted = df.copy()
    df_sorted['demand_pdemand_sum'] = df['demand'] + df['pdemand']

    df_high_cost = df_sorted[df_sorted['price'] > target_cost].sort_values(by=['demand_pdemand_sum', 'price'],
                                                                           ascending=[False, True])
    df_low_cost = df_sorted[df_sorted['price'] <= target_cost].sort_values(by=['demand_pdemand_sum', 'price'],
                                                                           ascending=[False, False])

    selected_indices = []

    high_cost_count = min(len(df_high_cost), num_ones // 2)
    selected_indices.extend(df_high_cost.index[:high_cost_count])

    remaining_count = num_ones - len(selected_indices)
    selected_indices.extend(df_low_cost.index[:remaining_count])

    for idx in selected_indices:
        individual[idx] = 1

    return icls(individual)


# 自定义变异函数，确保变异后仍然有固定数量的1
def mut_shuffle_indexes(individual, indpb):
    if np.random.random() < indpb:
        ones_indices = [i for i, bit in enumerate(individual) if bit == 1]
        zeros_indices = [i for i, bit in enumerate(individual) if bit == 0]
        if ones_indices and zeros_indices:
            swap_out = np.random.choice(ones_indices)
            swap_in = np.random.choice(zeros_indices)
            individual[swap_out], individual[swap_in] = individual[swap_in], individual[swap_out]
    return individual,


# 初始化遗传算法工具
def init_toolbox(df, cost, max_sites):
    # 注册工具
    toolbox = base.Toolbox()
    toolbox.register("individual", init_individual, creator.Individual, df=df, size=len(df), num_ones=max_sites,
                     target_cost=cost)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", mut_shuffle_indexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, df=df, max_cost=cost, max_sites=max_sites)
    return toolbox


# 遗传算法主函数
def genetic_algorithm(toolbox, population_size, generations, cxpb, mutpb):
    pop = toolbox.population(n=population_size)
    algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=generations, verbose=True)

    best_individual = tools.selBest(pop, k=1)[0]
    selected_indices = [i for i in range(len(best_individual)) if best_individual[i] == 1]
    return selected_indices


# 主调用函数
def run_site_selection(gridsum, cost=120 * 10000, population_size=900, generations=200, cxpb=0.5, mutpb=0.2,
                       max_sites=100):
    # 数据预处理

    df = preprocess_data(gridsum)

    # 初始化工具
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = init_toolbox(df, cost, max_sites)

    # 执行遗传算法
    optimal_sites = genetic_algorithm(toolbox, population_size, generations, cxpb, mutpb)

    # 筛选出最优站点数据
    optimal_grid_data = df.iloc[optimal_sites]
    return optimal_grid_data