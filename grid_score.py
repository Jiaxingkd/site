import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler
import transbigdata as tbd


# 读取数据文件，选择城市
def read_data(filename, select_city):
    """
    输入:
        filename (str): 文件路径
        select_city (str): 选择的城市名称
    输出:
        pd.DataFrame: 过滤后的数据集，仅包含选定城市的数据
    """
    df = pd.read_excel(filename)
    return df[df["市"] == select_city]


# 对每个列进行MinMax标准化，并计算得分
def calculate_scores(df, col, quantiles):
    """
    输入:
        df (pd.DataFrame): 输入的数据集
        col (pd.Index): 需要进行归一化的列
        quantiles (dict): 包含每列的0.96分位数的字典
    输出:
        pd.DataFrame: 返回包含标准化得分的新数据集，基于传入的0.96分位数作为最大值进行归一化
    """
    for c in col:
        # 获取对应列的0.96分位数作为max值
        max_val = quantiles[c]

        # 定义归一化公式：根据0.96分位数作为最大值
        new_col = c + "_score"
        df[new_col] = (df[c] / max_val) * 100

        # 如果大于100，归一化得分上限为100
        df[new_col] = df[new_col].clip(upper=100)

    # 计算总得分，汇总所有含有'score'字样的列
    df["total_score"] = df[[c for c in df.columns if "score" in c]].sum(axis=1)

    return df


# 提取经纬度坐标信息
def extract_coordinates(df):
    """
    输入:
        df (pd.DataFrame): 输入的数据集，包含栅格编号
    输出:
        pd.DataFrame: 返回提取了经纬度坐标的更新数据集
    """
    df["lon"] = df["栅格编号"].str.split("-", expand=True)[1].astype("float")
    df["lat"] = df["栅格编号"].str.split("-", expand=True)[2].astype("float")
    return df


# 将数据转换为GeoDataFrame
def create_geodataframe(df):
    """
    输入:
        df (pd.DataFrame): 包含经纬度的普通DataFrame
    输出:
        gpd.GeoDataFrame: 包含几何信息的GeoDataFrame
    """
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]))


# 对栅格评分，并生成格网
def score_grid(df, bounds):
    """
    输入:
        df (pd.DataFrame): 包含经纬度及评分的DataFrame
        bounds (tuple): 定义区域边界的坐标
    输出:
        pd.DataFrame: 包含总评分的格网数据
    """
    p = tbd.area_to_params(bounds, accuracy=5000)
    grid, p = tbd.area_to_grid(bounds, params=p)
    df["LONCOL"], df["LATCOL"] = tbd.GPS_to_grid(df["lon"], df["lat"], params=p)
    score = df.groupby(["LONCOL", "LATCOL"])["total_score"].sum().reset_index()
    score.columns = ["LONCOL", "LATCOL", "total_score"]
    grid = pd.merge(grid, score, on=["LONCOL", "LATCOL"], how="left")
    return grid[grid["total_score"].notna()]


# 根据评分将格网进行分类
def classify_grid(grid):
    """
    输入:
        grid (pd.DataFrame): 包含格网和评分的数据
    输出:
        pd.DataFrame: 排序后的格网数据，并按评分分类
    """
    grid.sort_values(by="total_score", ascending=False, inplace=True)
    grid["num"] = pd.cut(grid["total_score"], bins=7, labels=["0", "1", "2", "3", "4", "5", "6"])
    return grid


# 选择最佳站点
def select_best_sites(df, grid):
    """
    输入:
        df (pd.DataFrame): 原始数据集，包含坐标和评分
        grid (pd.DataFrame): 已经分类的格网数据
    输出:
        pd.DataFrame: 每个栅格中的最佳站点选择
    """
    select = pd.merge(df[["geometry", "LONCOL", "LATCOL", "total_score"]],
                      grid[["LONCOL", "LATCOL", "num"]])
    select["num"] = select["num"].astype(int)
    select = select.sort_values(by=["LONCOL", "LATCOL", "total_score"], ascending=[True, True, False])
    f = select.groupby(["LONCOL", "LATCOL"]).apply(lambda x: x.head(x["num"].iloc[0])).reset_index(drop=True)
    return f


# 主函数，用于调用所有步骤
def main(filename, select_city, bounds):
    """
    输入:
        filename (str): 文件路径
        select_city (str): 选择的城市
        bounds (tuple): 定义区域边界的坐标
    输出:
        pd.DataFrame: 最终的站点选择结果
    """
    df_gz = read_data(filename, select_city)
    cols = df_gz.columns[4:]
    df = pd.read_excel(filename)
    quantiles = {}
    for col in cols:
        quantiles[col] = df[col].quantile(0.96)
    df_gz = calculate_scores(df_gz, cols, quantiles)
    df_gz = extract_coordinates(df_gz)
    df_gz = create_geodataframe(df_gz)
    grid = score_grid(df_gz, bounds)
    grid = classify_grid(grid)
    best_sites = select_best_sites(df_gz, grid)
    return best_sites
