import pandas as pd
import geopandas as gpd
import transbigdata as tbd
from sklearn.preprocessing import MinMaxScaler
import os


# Function to process Haikou city data
def process_haikou_data(filename: str) -> pd.DataFrame:
    df = pd.read_excel(filename)
    df = df.fillna(method="ffill")
    df = df[df["84坐标系经纬度信息"].str.contains("N")]
    df["84坐标系经纬度信息"] = df["84坐标系经纬度信息"].str.replace("T", "1")
    df["latitude"] = df["84坐标系经纬度信息"].str.extract(r"(\d+\.\d+)[°N]")
    df["longitude"] = df["84坐标系经纬度信息"].str.extract(r"[，,.\s](\d+\.\d+)[°E]")
    df["longitude"] = df["longitude"].astype("float")
    df["latitude"] = df["latitude"].astype("float")
    return df


# Function to process data for other cities
def process_other_data(filename: str) -> pd.DataFrame:
    df = pd.read_excel(filename)
    return df


# Function to convert longitude and latitude to grid
def to_grid(bounds: list, df: pd.DataFrame) -> tuple:
    p = tbd.area_to_params(bounds, accuracy=500)
    grid, p = tbd.area_to_grid(bounds, accuracy=500)
    df["loncol"], df["latcol"] = tbd.GPS_to_grid(df["longitude"], df["latitude"], params=p)
    grid.columns = ["loncol", "latcol", "geometry"]
    return df, grid, p


# Function to process MR grid data
def process_mrgrid(mrfile: str, p) -> pd.DataFrame:
    mr = pd.read_csv(mrfile)
    mr["loncol"], mr["latcol"] = tbd.GPS_to_grid(mr["longitude"], mr["latitude"], params=p)
    return mr


# Function to process POI data
def process_poi(poifile: str, city: str, p) -> pd.DataFrame:
    merge_df = pd.DataFrame()
    for file in os.listdir(poifile):
        file_path = os.path.join(poifile, file)
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path, encoding_errors="ignore", encoding="gbk", skiprows=[618095])
            df = df[df["cityname"] == city]
            merge_df = pd.concat([merge_df, df])
    merge_df = merge_df[merge_df["page_publish_time"].str.contains("2018")]
    merge_df["ttype"] = merge_df["type"].str.split(";").str.get(0)
    merge_df["lon"] = merge_df["location"].str.split("，").str.get(0).astype("float")
    merge_df["lat"] = merge_df["location"].str.split("，").str.get(1).astype("float")
    merge_df["loncol"], merge_df["latcol"] = tbd.GPS_to_grid(merge_df["lon"], merge_df["lat"], params=p)
    poi = merge_df.groupby(["ttype", "loncol", "latcol"]).size().reset_index()
    poi.columns = ["ttype", "loncol", "latcol", "count"]
    poi_pivot = poi.pivot_table(index=["loncol", "latcol"], columns="ttype", values="count", fill_value=0).reset_index()
    return poi_pivot


# Function to scale columns and calculate total scores
def trancol(df: pd.DataFrame, col: str) -> pd.DataFrame:
    scaler = MinMaxScaler()
    new_col_name = f"{col}_score"
    df[new_col_name] = scaler.fit_transform(df[[col]]) * 100
    return df


# Function to calculate final scores
def calculate_scores(grid_score: pd.DataFrame, num: int) -> pd.DataFrame:
    c = grid_score.columns[3:]
    for col in c:
        grid_score = trancol(grid_score, col)
    grid_score["total_score"] = grid_score.filter(like="score").sum(axis=1)
    grid_score["num"] = pd.cut(grid_score["total_score"], bins=10, labels=[str(i) for i in range(1, 11)]).astype("int")
    grid_score["ratio"] = grid_score["num"] / grid_score["num"].sum()
    grid_score = grid_score.sort_values(by="total_score", ascending=False)
    grid_score["rank"] = grid_score["total_score"].rank(ascending=False, method="dense").astype(int)
    grid_score.reset_index(drop=True, inplace=True)
    grid_score["c_num"] = grid_score["ratio"] * (num * grid_score["ratio"].sum())
    return grid_score


# Main process
def main_process(filename: str, bounds: list, mrfile: str, poifile: str, city: str, num: int, is_haikou: bool):
    # Data reading and processing
    df = process_haikou_data(filename) if is_haikou else process_other_data(filename)

    # Convert to grid
    df, grid, p = to_grid(bounds, df)

    # Process MR grid data
    mr = process_mrgrid(mrfile, p)
    df = pd.merge(df, mr, on=["loncol", "latcol"], how="left")

    # Calculate grid scores
    score = df[["loncol", "latcol", "总平均"]]
    grid_score = pd.merge(grid, score, on=["loncol", "latcol"])

    # Process POI data
    poi_pivot = process_poi(poifile, city, p)
    grid_score = pd.merge(grid_score, poi_pivot, on=["loncol", "latcol"], how="left")
    grid_score.fillna(0, inplace=True)

    # Calculate final scores
    final_scores = calculate_scores(grid_score, num)

    return final_scores

