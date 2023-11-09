import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import sparse


def haversine_distance_fn(x, y, grad_to_rad=False):
    if grad_to_rad:
        x = np.radians(x)
        y = np.radians(y)
    W = 6353 * haversine_distances(x, y)  # 6353? Radius of earth is 6371
    return W


class StoresPredictor(object):
    """Class to propose new stores given current stores and sales in different locations
    Args:
        initial_stores (pd.DataFrame): Dataframe with current stores and their locations
        possible_stores (pd.DataFrame, optional): Dataframe with available locations for new stores to propose
        sales (pd.DataFrame): Dataframe with the online sales of different zip codes
        n (int): Number of stores to predict
        min_dist_stores (int,optional): Minimum distance between stores
        sales_radius (int,optional): Radius in which we try to maximize sales.
            The sales of a store are the sum of zipcode sales within the radius.
    """

    def __init__(
        self,
        initial_stores: pd.DataFrame,
        possible_stores: pd.DataFrame,
        sales: pd.DataFrame,
        n: int,
        min_dist_stores: int = 20,
        sales_radius: int = 10,
    ):
        self.initial_stores = initial_stores.copy(deep=True)
        self.possible_stores = possible_stores.copy(deep=True)
        self.sales = sales.copy(deep=True)
        self.n = n
        self.distance_fn = haversine_distance_fn
        self.min_dist_stores = min_dist_stores
        self.sales_radius = sales_radius

    def algorithm(self):
        for i in range(self.n):
            if len(self.predicted_stores) == 0:
                Wip = self.distance_fn(self.i, self.p, grad_to_rad=True)
                Wis = self.distance_fn(self.i, self.s, grad_to_rad=True)
            else:
                last_predicted = np.asarray(
                    self.predicted_stores.iloc[-1][["LONGITUDE", "LATITUDE"]],
                    dtype=np.float32,
                ).reshape(-1, 2)
                Wip = self.distance_fn(last_predicted, self.p, grad_to_rad=True)
                Wis = self.distance_fn(last_predicted, self.s, grad_to_rad=True)

            p_mask = (Wip > self.min_dist_stores).all(axis=0)
            self.p = self.p[p_mask]

            s_mask = (Wis > self.sales_radius).all(axis=0)
            self.units = self.units[s_mask]
            self.s = self.s[s_mask]

            if len(self.predicted_stores) == 0:
                self.Wps = self.distance_fn(self.p, self.s, grad_to_rad=True)
                self.Wps = self.Wps <= self.sales_radius
                self.Wps = sparse.csr_matrix(self.Wps * self.units[np.newaxis, :])
            else:
                self.Wps = self.Wps[p_mask, :][:, s_mask]

            p_sales = np.asarray(self.Wps.sum(axis=1))
            p_idx = np.argmax(p_sales)
            long, lat = self.p[p_idx]

            self.predicted_stores.loc[len(self.predicted_stores)] = {
                "STORE_NAME": f"predicted_store_{i+1}",
                "LONGITUDE": long,
                "LATITUDE": lat,
                "UNITS": p_sales[p_idx][0],
            }
        return self.predicted_stores

    def run(self):
        self.preprocess()
        self.algorithm()
        self.postprocess()
        return self

    def preprocess(self):
        self.predicted_stores = pd.DataFrame(
            columns=["STORE", "STORE_NAME", "LONGITUDE", "LATITUDE", "UNITS"]
        )
        self.p = np.array(self.possible_stores[["LONGITUDE", "LATITUDE"]])
        self.s = np.array(self.sales[["LONGITUDE", "LATITUDE"]])
        self.i = np.array(self.initial_stores[["LONGITUDE", "LATITUDE"]])
        self.units = np.array(self.sales["UNITS"])

    def postprocess(self):
        # Add zipcode to predicted_stores
        self.predicted_stores = pd.merge(
            self.predicted_stores,
            self.sales[["LONGITUDE", "LATITUDE", "ZIPCODE"]],
            how="left",
            left_on=["LONGITUDE", "LATITUDE"],
            right_on=["LONGITUDE", "LATITUDE"],
        )
        return self.predicted_stores

    def to_csv(self, file_path="stores.csv"):
        final_df = self.predicted_stores[["ZIPCODE", "LONGITUDE", "LATITUDE", "UNITS"]]
        new_columns = {col: col.lower() for col in final_df.columns}
        final_df.rename(columns=new_columns).to_csv(file_path, index=False)
        return file_path

    def plot(self, file_path="stores.jpg"):
        plz_shape_df = gpd.read_file(
            "data/germany_map/plz-1stellig.shp", dtype={"plz": str}
        )
        fig, ax = plt.subplots()
        plz_shape_df.plot(ax=ax, color="lightblue", alpha=0.6)
        # Plot sales
        max_sale = self.sales["UNITS"].max()
        min_sale = self.sales["UNITS"].min()
        scale = 30
        for idx in self.sales.index:
            row = self.sales.iloc[idx]
            longitude = row["LONGITUDE"]
            latitude = row["LATITUDE"]
            unit = row["UNITS"]
            ax.plot(
                longitude,
                latitude,
                marker="o",
                markersize=scale * (unit - min_sale) / (max_sale - min_sale),
                c="grey",
                alpha=0.4,
            )
        # Plot stores
        for idx in self.initial_stores.index:
            row = self.initial_stores.iloc[idx]
            longitude = row["LONGITUDE"]
            latitude = row["LATITUDE"]
            ax.plot(longitude, latitude, marker="o", markersize=5, c="blue", alpha=0.8)

        for idx in self.predicted_stores.index:
            row = self.predicted_stores.iloc[idx]
            longitude = row["LONGITUDE"]
            latitude = row["LATITUDE"]
            ax.text(
                x=longitude,
                y=latitude + 0.09,
                s=row["STORE_NAME"].split("predicted_store_")[1],
                color="red",
                fontsize=5,
                ha="center",
            )
            ax.plot(longitude, latitude, marker="o", markersize=5, c="red", alpha=0.8)
        ax.set(title="Germany", aspect=1.3, facecolor="white")
        sales_legend = mlines.Line2D(
            [],
            [],
            color="grey",
            marker="o",
            linestyle="None",
            markersize=5,
            label="Sales",
        )
        initial_stores_legend = mlines.Line2D(
            [],
            [],
            color="blue",
            marker="o",
            linestyle="None",
            markersize=5,
            label="Actual stores",
        )
        predicted_stores_legend = mlines.Line2D(
            [],
            [],
            color="red",
            marker="o",
            linestyle="None",
            markersize=5,
            label="Predicted stores",
        )
        ax.legend(
            handles=[sales_legend, initial_stores_legend, predicted_stores_legend],
            loc="upper right",
        )

        plt.savefig(file_path, dpi=400)
        return file_path

    @property
    def current_stores(self):
        return pd.concat([self.initial_stores, self.predicted_stores], join="inner")


def main():
    print(">>Loading data...")
    sales = pd.read_excel("data/sales.xlsx")
    stores = pd.read_excel("data/stores.xlsx")
    print(">>Data loaded")

    predictor = StoresPredictor(
        initial_stores=stores,
        possible_stores=sales[["LONGITUDE", "LATITUDE"]],
        sales=sales,
        n=15,
        min_dist_stores=20,
        sales_radius=10,
    )

    print(">>Predicting new stores...")
    predictor.run()
    print(">>Prediction finished")
    print(">>Saving results to csv...")
    predictor.to_csv()
    print(">>File saved")
    print(">>Saving stores figure...")
    predictor.plot()
    print(">>Figure saved")


if __name__ == "__main__":
    main()
