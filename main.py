from signal_processor.signal_processor import SignalProcessor
from data_loader.data_loader import DataLoader
from visualizer.visualizer import Visualizer
from obspy import UTCDateTime


def test_with_swiss_station():

    # 1. Загрузка данных
    loader = DataLoader()
    event_time = UTCDateTime("2025-12-21T01:29:00")

    raw_data_dicts = loader.download_event_data(
        event_time=event_time.datetime,
        duration_before_min=5,
        duration_after_min=90
    )

    if not raw_data_dicts:
        print("Не удалось загрузить данные!")
        return

    # 2. Обработка всех станций
    processor = SignalProcessor()

    for station_id, raw_data_dict in raw_data_dicts.items():
        station_data, analysis = processor.process_station(
            raw_data_dict,
            raw_data_dict["station"]
        )
        raw_data_dicts[station_id]["processed_data"] = station_data
        raw_data_dicts[station_id]["analysis"] = analysis

    # 3. Визуализация
    epicenter = (46.2, 7.8)  # пример координат
    Visualizer.plot_seismograms_and_map(raw_data_dicts, epicenter)


if __name__ == "__main__":
    test_with_swiss_station()


"""
from signal_processor.signal_processor import SignalProcessor
from obspy.clients.fdsn import Client
from data_loader.data_loader import DataLoader
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np


def test_with_swiss_station():

    # 1. Загрузка данных (через DataLoader)
    loader = DataLoader()
    event_time = UTCDateTime("2025-12-21T01:29:00")

    raw_data_dicts = loader.download_event_data(
        event_time=event_time.datetime,
        duration_before_min=5,
        duration_after_min=90
    )

    if not raw_data_dicts:
        print("Не удалось загрузить данные!")
        return

    # Берем первую станцию (аналог CH.HASLI)
    station_id = list(raw_data_dicts.keys())[2]
    raw_data_dict = raw_data_dicts[station_id]

    print(f"\nРаботаем со станцией: {station_id}")

    # 2. Интеграция в SignalProcessor
    #safe_config = {'pre_filt': (0.01, 0.05, 8.0, 9.0), 'max_frequency': 9.0}
    processor = SignalProcessor()

    # 3. Запуск обработки
    # Метод сам вызовет калибровку, БПФ-анализ и фильтрацию
    station_data, analysis = processor.process_station(
        raw_data_dict,
        raw_data_dict["station"]
    )

    # 4. Визуализация результатов
    print(f"\nАнализ завершен. Оптимальный фильтр: {analysis.optimal_band} Гц")

    data_filtered = station_data.z_data 
    # Извлекаем исходный сигнал из словаря
    data_raw = raw_data_dict['raw_data'] 

    # 2. Создание временной оси в минутах
    # Количество точек / частота дискретизации / 60 секунд
    sampling_rate = station_data.sampling_rate
    time_min = np.arange(len(data_raw)) / sampling_rate / 60.0

    # 3. Ваша визуализация
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    plt.subplots_adjust(hspace=0.3)

    # Верхний график: Исходные данные (Counts)
    ax1.plot(time_min, data_raw, 'k', linewidth=0.8)
    ax1.set_title(f"1. Исходная сейсмограмма (отсчеты)", fontsize=12)
    ax1.set_ylabel("Амплитуда (Counts)")
    ax1.grid(alpha=0.3)

    # Нижний график: Отфильтрованные данные (Скорость)
    # После SignalProcessor данные уже в физических величинах (м/с)
    ax2.plot(time_min, data_filtered, 'b', linewidth=1)
    ax2.set_title(
        f"2. Обработанный сигнал (Фильтр: {analysis.optimal_band} Гц)",
        fontsize=12
    )
    ax2.set_xlabel("Время (минуты)")
    ax2.set_ylabel("Скорость (м/с)")
    ax2.grid(alpha=0.3)

    plt.show()


if __name__ == "__main__":
    test_with_swiss_station()

"""