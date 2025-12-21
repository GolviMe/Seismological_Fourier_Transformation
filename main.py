from signal_processor.signal_processor import SignalProcessor
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np

def test_with_swiss_station():
    # 1. Твой рабочий кусок кода для загрузки
    client = Client("IRIS")
    event_time = UTCDateTime("2025-12-21T01:29:00")

    # 2. Берем окно: 5 минут до события и 90 минут после, 
    # чтобы увидеть приход всех типов волн
    start = event_time - 300 
    end = event_time + 5400
    
    print(f"Загружаем последние данные с CH.HASLI ({start} - {end})...")
    
    try:
        # Загружаем данные
        st = client.get_waveforms("CH", "HASLI", "*", "BHZ", start, end)
        # Загружаем инвентарь (нужен для калибровки remove_response в твоем классе)
        inv = client.get_stations(network="CH", station="HASLI", level="response", 
                                 starttime=start, endtime=end)
        
        tr = st[0]
        
        inv = client.get_stations(
            network=tr.stats.network,
            station=tr.stats.station,
            location=tr.stats.location,  # именно так, как в данных!
            channel=tr.stats.channel,
            level="response",
            starttime=start - 10,  # немного раньше для надежности
            endtime=end + 10
        )
        
        raw_data_dict = {
            'raw_data': tr.data,
            'sampling_rate': tr.stats.sampling_rate,
            'coordinates': (46.7, 8.3),
            'inventory': inv,
            'network': tr.stats.network,   # Берем из tr.stats, а не из строки
            'station': tr.stats.station,   # Добавь это поле, если оно нужно
            'location': tr.stats.location, # Передаем реальный код локации (часто это '')
            'channel': tr.stats.channel
        }
        
    except Exception as e:
        print(f"Ошибка при загрузке: {e}")
        return

    # 2. Интеграция в SignalProcessor
    #safe_config = {'pre_filt': (0.01, 0.05, 8.0, 9.0), 'max_frequency': 9.0}
    processor = SignalProcessor()

    # 3. Запуск обработки
    # Метод сам вызовет калибровку, БПФ-анализ и фильтрацию
    station_data, analysis = processor.process_station(raw_data_dict, raw_data_dict["station"])

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
    ax2.set_title(f"2. Обработанный сигнал (Фильтр: {analysis.optimal_band} Гц)", fontsize=12)
    ax2.set_xlabel("Время (минуты)")
    ax2.set_ylabel("Скорость (м/с)")
    ax2.grid(alpha=0.3)

    plt.show()

if __name__ == "__main__":
    test_with_swiss_station()