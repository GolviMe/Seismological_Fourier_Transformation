"""
interfaces.py - Контракты между модулями проекта
Все члены команды ДОЛЖНЫ соблюдать эти интерфейсы.
Изменения в интерфейсах согласовываются ВСЕЙ командой.
"""

from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from datetime import datetime

# ============================================================================
# ОСНОВНЫЕ ТИПЫ ДАННЫХ
# ============================================================================

@dataclass
class StationData:
    """Данные одной сейсмической станции после обработки"""
    station_id: str                    # Код станции, например "KIV"
    z_data: np.ndarray                # Вертикальная компонента (основная)
    n_data: Optional[np.ndarray]      # Северная компонента (опционально)
    e_data: Optional[np.ndarray]      # Восточная компонента (опционально)
    sampling_rate: float              # Частота дискретизации (Гц)
    coordinates: Tuple[float, float]  # (широта, долгота)
    elevation: Optional[float]        # Высота над уровнем моря (м)
    processed: bool = True            # Флаг, что данные обработаны
    
@dataclass
class WaveArrival:
    """Время прихода волны на станции"""
    station_id: str
    p_time: Optional[float]          # Время P-волны в секундах от начала записи
    s_time: Optional[float]          # Время S-волны в секундах от начала записи
    p_confidence: float = 0.0        # Уверенность в определении P (0-1)
    s_confidence: float = 0.0        # Уверенность в определении S (0-1)
    method: str = ""                 # Метод детектирования (например "STA/LTA")
    
@dataclass
class EarthquakeLocation:
    """Результат локации землетрясения"""
    epicenter: Tuple[float, float]    # (широта, долгота)
    depth: Optional[float]            # Глубина (км)
    magnitude: Optional[float]        # Магнитуда
    confidence_radius: float          # Радиус погрешности (км)
    origin_time: Optional[datetime]   # Время в очаге
    stations_used: List[str]          # Список станций, использованных для локации
    residuals: List[float]            # Невязки по станциям
    
@dataclass
class SpectralAnalysis:
    """Результаты спектрального анализа"""
    frequencies: np.ndarray          # Массив частот (Гц)
    amplitude_raw: np.ndarray        # Амплитудный спектр исходных данных
    amplitude_filtered: np.ndarray   # Амплитудный спектр после фильтрации
    noise_spectrum: np.ndarray       # Спектр шумового участка
    signal_spectrum: np.ndarray      # Спектр сигнального участка
    optimal_band: Tuple[float, float]  # Оптимальная полоса фильтра (fmin, fmax)

# ============================================================================
# ИНТЕРФЕЙСЫ (контракты)
# ============================================================================

class IDataLoader:
    """
    ИНТЕРФЕЙС 1: Загрузчик данных (реализует Человек 2)
    Отвечает за получение сырых данных с сейсмографов
    """
    
    def download_raw_data(self,
                         stations: List[str],
                         start_time: datetime,
                         duration_minutes: int = 30,
                         network: str = "*") -> Dict[str, Dict[str, Any]]:
        """
        Загружает сырые данные с серверов
        
        Args:
            stations: Список кодов станций ['KIV', 'ARU', ...]
            start_time: Начальное время UTC
            duration_minutes: Длительность записи в минутах
            network: Сеть станций (по умолчанию любая)
            
        Returns:
            Словарь, где ключ - код станции, значение - словарь с данными:
            {
                'raw_data': np.ndarray,      # Сырые отсчеты
                'sampling_rate': float,
                'channels': List[str],       # ['BHZ', 'BHN', 'BHE']
                'coordinates': (lat, lon),
                'metadata': Dict[str, Any]   # Дополнительные метаданные
            }
            
        Raises:
            DataDownloadError: Если данные не загружены
            StationNotFoundError: Если станция не найдена
        """
        raise NotImplementedError("Должен быть реализован в data_loader.py")
    

class ISignalProcessor:
    """
    ИНТЕРФЕЙС 2: Обработчик сигналов (реализует Человек 2)
    Отвечает за обработку данных: калибровка, Фурье-анализ, фильтрация
    """
    
    def process_station(self,
                       raw_data: Dict[str, Any],
                       station_id: str) -> Tuple[StationData, SpectralAnalysis]:
        """
        Обрабатывает данные одной станции
        
        Args:
            raw_data: Сырые данные от IDataLoader.download_raw_data()
            station_id: Код станции
            
        Returns:
            Кортеж из:
            1. StationData - обработанные данные
            2. SpectralAnalysis - результаты спектрального анализа
            
        Шаги обработки (внутренние, не видны другим):
        1. Калибровка (удаление инструментального отклика)
        2. БПФ-анализ для определения оптимального фильтра
        3. Применение bandpass-фильтра
        4. Возврат готовых данных
        """
        raise NotImplementedError("Должен быть реализован в signal_processor.py")
    
    def get_optimal_filter_band(self,
                               station_id: str,
                               raw_data: np.ndarray,
                               sampling_rate: float) -> Tuple[float, float]:
        """
        Определяет оптимальную полосу частот для фильтрации
        
        Args:
            station_id: Код станции (для кэширования)
            raw_data: Сырые данные после калибровки
            sampling_rate: Частота дискретизации
            
        Returns:
            (fmin, fmax) - оптимальные частоты фильтра
            
        Примечание:
        Использует БПФ для анализа спектра и определения,
        на каких частотах сигнал землетрясения сильнее шума
        """
        raise NotImplementedError("Должен быть реализован в signal_processor.py")
    

class IWaveDetector:
    """
    ИНТЕРФЕЙС 3: Детектор волн (реализует Человек 3)
    Отвечает за определение времён прихода P и S волн
    """
    
    def detect_arrivals(self,
                       station_data: StationData) -> WaveArrival:
        """
        Определяет время прихода P и S волн
        
        Args:
            station_data: Обработанные данные станции
            
        Returns:
            WaveArrival с определенными временами
            
        Алгоритм (внутренний):
        1. Применение STA/LTA или других методов
        2. Поиск P-волны (первое резкое изменение)
        3. Поиск S-волны (максимальная энергия после P)
        4. Расчет уверенности детектирования
        """
        raise NotImplementedError("Должен быть реализован в wave_detector.py")
    
    def verify_arrivals(self,
                       arrivals: WaveArrival,
                       station_data: StationData) -> bool:
        """
        Проверяет, что определенные времена реалистичны
        
        Args:
            arrivals: Определенные времена волн
            station_data: Исходные данные
            
        Returns:
            True если времена физически реалистичны
        
        Проверяет:
        - S время > P время
        - S-P в разумных пределах (10-1000 сек)
        - Форма сигнала соответствует ожидаемой
        """
        raise NotImplementedError("Должен быть реализован в wave_detector.py")
    

class IEpicenterLocator:
    """
    ИНТЕРФЕЙС 4: Локатор эпицентра (реализует Человек 3)
    Отвечает за вычисление координат землетрясения
    """
    
    def locate_epicenter(self,
                        arrivals: List[WaveArrival],
                        stations_info: Dict[str, StationData]) -> EarthquakeLocation:
        """
        Вычисляет эпицентр землетрясения
        
        Args:
            arrivals: Времена прихода волн для каждой станции
            stations_info: Информация о станциях (координаты)
            
        Returns:
            EarthquakeLocation - результат локации
            
        Алгоритм (внутренний):
        1. Вычисление расстояний от станций
        2. Триангуляция (метод наименьших квадратов)
        3. Расчет погрешности
        4. Оценка магнитуды (опционально)
        """
        raise NotImplementedError("Должен быть реализован в epicenter_locator.py")
    
    def calculate_distance(self,
                          p_time: float,
                          s_time: float,
                          velocity_model: str = "default") -> float:
        """
        Вычисляет расстояние до эпицентра по разнице S-P
        
        Args:
            p_time: Время P-волны
            s_time: Время S-волны
            velocity_model: Модель скоростей ("default", "crust", "global")
            
        Returns:
            Расстояние в километрах
        """
        raise NotImplementedError("Должен быть реализован в epicenter_locator.py")
    

class IVisualizer:
    """
    ИНТЕРФЕЙС 5: Визуализатор (реализует Человек 1)
    Отвечает за создание графиков и карт
    """
    
    def create_seismogram_plot(self,
                              station_data: StationData,
                              arrivals: Optional[WaveArrival] = None,
                              title: str = "") -> plt.Figure:
        """
        Создает график сейсмограммы
        
        Args:
            station_data: Данные станции
            arrivals: Определенные времена волн (опционально)
            title: Заголовок графика
            
        Returns:
            matplotlib Figure с графиком
        """
        raise NotImplementedError("Должен быть реализован в visualizer.py")
    
    def create_spectrum_plot(self,
                           spectral_analysis: SpectralAnalysis,
                           title: str = "") -> plt.Figure:
        """
        Создает график спектрального анализа
        
        Args:
            spectral_analysis: Результаты БПФ-анализа
            title: Заголовок графика
            
        Returns:
            matplotlib Figure с графиками спектров
        """
        raise NotImplementedError("Должен быть реализован в visualizer.py")
    
    def create_map_with_epicenter(self,
                                 location: EarthquakeLocation,
                                 stations: List[StationData],
                                 arrivals: List[WaveArrival]) -> plt.Figure:
        """
        Создает карту с эпицентром и станциями
        
        Args:
            location: Результат локации
            stations: Информация о станциях
            arrivals: Времена прихода волн
            
        Returns:
            matplotlib Figure с картой
        """
        raise NotImplementedError("Должен быть реализован в visualizer.py")
    

class IErrorHandler:
    """
    ИНТЕРФЕЙС 6: Обработчик ошибок (реализует Человек 1)
    Отвечает за обработку исключительных ситуаций
    """
    
    class DataDownloadError(Exception):
        """Ошибка загрузки данных"""
        pass
    
    class ProcessingError(Exception):
        """Ошибка обработки данных"""
        pass
    
    class DetectionError(Exception):
        """Ошибка детектирования волн"""
        pass
    
    class LocationError(Exception):
        """Ошибка локации эпицентра"""
        pass
    
    def handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """
        Обрабатывает ошибку и возвращает информацию для пользователя
        
        Args:
            error: Пойманное исключение
            context: Контекст, в котором произошла ошибка
            
        Returns:
            Словарь с информацией об ошибке:
            {
                'message': str,
                'recovery_suggestion': str,
                'critical': bool
            }
        """
        raise NotImplementedError("Должен быть реализован в error_handler.py")

# ============================================================================
# ФАБРИКИ (создатели объектов)
# ============================================================================

def create_data_loader() -> IDataLoader:
    """
    Фабрика для создания загрузчика данных
    Использование: data_loader = create_data_loader()
    """
    from data_loader import DataLoader  # Импорт здесь, чтобы избежать циклических зависимостей
    return DataLoader()

def create_signal_processor() -> ISignalProcessor:
    """Фабрика для создания обработчика сигналов"""
    from signal_processor import SignalProcessor
    return SignalProcessor()

def create_wave_detector() -> IWaveDetector:
    """Фабрика для создания детектора волн"""
    from wave_detector import WaveDetector
    return WaveDetector()

def create_epicenter_locator() -> IEpicenterLocator:
    """Фабрика для создания локатора эпицентра"""
    from epicenter_locator import EpicenterLocator
    return EpicenterLocator()

def create_visualizer() -> IVisualizer:
    """Фабрика для создания визуализатора"""
    from visualizer import Visualizer
    return Visualizer()

def create_error_handler() -> IErrorHandler:
    """Фабрика для создания обработчика ошибок"""
    from error_handler import ErrorHandler
    return ErrorHandler()

# ============================================================================
# КОНФИГУРАЦИЯ ПРОЕКТА
# ============================================================================

class ProjectConfig:
    """
    Конфигурация проекта - настройки по умолчанию
    Могут быть переопределены в main.py
    """
    
    # Станции по умолчанию для Евразии
    DEFAULT_STATIONS = [
        "KIV",    # Киев, Украина
        "ARU",    # Арти, Россия
        "IL31",   # Израиль
        "BGCA",   # Болгария
        "TNS",    # Тунис
    ]
    
    # Частоты фильтров по умолчанию (если автоподбор не работает)
    DEFAULT_FILTER_BAND = (1.0, 10.0)  # Гц
    
    # Параметры детектирования
    STA_WINDOW = 1.0    # секунд
    LTA_WINDOW = 30.0   # секунд
    DETECTION_THRESHOLD = 3.0
    
    # Скорости волн (км/с)
    P_WAVE_VELOCITY = 6.0
    S_WAVE_VELOCITY = 3.5
    
    # Сервера данных
    DATA_SERVERS = ["IRIS", "EMSC", "GEONET"]
    
    # Пути для сохранения
    OUTPUT_DIR = "output"
    FIGURES_DIR = "output/figures"
    DATA_CACHE_DIR = "cache"