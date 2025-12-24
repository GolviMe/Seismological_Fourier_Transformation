from obspy.clients.fdsn import Client
from obspy import UTCDateTime


class DataLoader:
    def __init__(self, provider="IRIS"):
        self.client = Client(provider)

        # === СПИСОК СТАНЦИЙ ===
        # Формат: (network, station)
        self.stations = [
            ("CH", "HASLI"),
            ("IU", "ANMO"),
            ("IU", "ULN"),
        ]

    def download_event_data(
        self,
        event_time,
        duration_before_min=5,
        duration_after_min=90
    ):
        """
        Загружает данные с нескольких станций
        и возвращает raw_data_dict для каждой станции
        """

        event_time = UTCDateTime(event_time)
        start = event_time - duration_before_min * 60
        end = event_time + duration_after_min * 60

        raw_data_dicts = {}

        for network, station in self.stations:
            station_id = f"{network}.{station}"
            print(f"Загружаем данные с {station_id} ({start} - {end})...")

            try:
                # --- ЗАГРУЗКА ВОЛНОФОРМ ---
                st = self.client.get_waveforms(
                    network=network,
                    station=station,
                    location="*",
                    channel="BHZ",
                    starttime=start,
                    endtime=end
                )

                tr = st[0]

                # --- ЗАГРУЗКА ИНВЕНТАРЯ ---
                inv = self.client.get_stations(
                    network=tr.stats.network,
                    station=tr.stats.station,
                    location=tr.stats.location,
                    channel=tr.stats.channel,
                    level="response",
                    starttime=start - 10,
                    endtime=end + 10
                )

                # --- ФОРМИРОВАНИЕ raw_data_dict (СТРОГО КАК В main.py) ---
                raw_data_dict = {
                    'raw_data': tr.data,
                    'sampling_rate': tr.stats.sampling_rate,
                    'coordinates': (
                        getattr(tr.stats, "sac", {}).get("stla", None),
                        getattr(tr.stats, "sac", {}).get("stlo", None),
                    ),
                    'inventory': inv,
                    'network': tr.stats.network,
                    'station': tr.stats.station,
                    'location': tr.stats.location,
                    'channel': tr.stats.channel
                }

                raw_data_dicts[station_id] = raw_data_dict
                print(f"✓ {station_id} загружена")

            except Exception as e:
                print(f"✗ Ошибка загрузки {station_id}: {e}")

        return raw_data_dicts
