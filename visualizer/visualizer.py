# visualizer.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature


class Visualizer:

    @staticmethod
    def plot_seismograms_and_map(raw_data_dicts, epicenter):
        """
        Слева — сейсмограммы всех станций
        Справа — карта с эпицентром
        """

        stations = list(raw_data_dicts.keys())
        n = len(stations)

        fig = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(n, 2, width_ratios=[2.2, 1])

        # ==================================================
        # СЛЕВА — СЕЙСМОГРАММЫ
        # ==================================================
        for i, station_id in enumerate(stations):
            ax = fig.add_subplot(gs[i, 0])

            data = raw_data_dicts[station_id]
            signal = data['raw_data']
            fs = data['sampling_rate']

            time_min = np.arange(len(signal)) / fs / 60.0

            ax.plot(time_min, signal, 'k', linewidth=0.7)
            ax.set_ylabel(station_id, fontsize=9)
            ax.grid(alpha=0.3)

            if i == n - 1:
                ax.set_xlabel("Время (мин)")
            else:
                ax.set_xticklabels([])

        # ==================================================
        # СПРАВА — КАРТА (CARTOPY)
        # ==================================================
        ax_map = fig.add_subplot(gs[:, 1], projection=ccrs.PlateCarree())

        # Границы карты по станциям
        lats = []
        lons = []
        for d in raw_data_dicts.values():
            lat, lon = d['coordinates']
            if lat is not None and lon is not None:
                lats.append(lat)
                lons.append(lon)

        epi_lat, epi_lon = epicenter
        lats.append(epi_lat)
        lons.append(epi_lon)

        margin = 1.0
        ax_map.set_extent([
            min(lons) - margin,
            max(lons) + margin,
            min(lats) - margin,
            max(lats) + margin
        ])

        # География
        ax_map.add_feature(cfeature.LAND, facecolor='lightgray')
        ax_map.add_feature(cfeature.COASTLINE)
        ax_map.add_feature(cfeature.BORDERS, linestyle=':')
        ax_map.add_feature(cfeature.LAKES, alpha=0.5)
        ax_map.add_feature(cfeature.RIVERS)

        # Эпицентр
        ax_map.plot(
            epi_lon, epi_lat,
            marker='*', color='red',
            markersize=15,
            transform=ccrs.PlateCarree(),
            label="Эпицентр"
        )

        # Станции
        for station_id, data in raw_data_dicts.items():
            lat, lon = data['coordinates']
            if lat is None or lon is None:
                continue

            ax_map.plot(
                lon, lat,
                marker='^', color='blue',
                markersize=8,
                transform=ccrs.PlateCarree()
            )

            ax_map.text(
                lon + 0.05, lat + 0.05,
                station_id,
                fontsize=9,
                transform=ccrs.PlateCarree()
            )

            # Линия станция–эпицентр
            ax_map.plot(
                [lon, epi_lon],
                [lat, epi_lat],
                'k--', alpha=0.5,
                transform=ccrs.PlateCarree()
            )

        ax_map.set_title("Карта станций и эпицентра", fontsize=12)
        ax_map.legend()

        plt.tight_layout()
        plt.show()
