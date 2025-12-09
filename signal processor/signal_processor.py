import os
from typing import Dict, List, Tuple, Optional, Any
from ..interfaces import ISignalProcessor, StationData, SpectralAnalysis

class SignalProcessor(ISignalProcessor):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация процессора сигналов
        
        Args:
            config: Конфигурация обработки (может быть пустой)
        """
        self.config = config or self._get_default_config()
        self.cache_dir = self.config.get('cache_dir', 'cache')
        self.optimal_filters_cache = {}  # Кэш оптимальных фильтров по станциям
        
        # Создаем директории для кэша
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, 'spectra'), exist_ok=True)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Возвращает конфигурацию по умолчанию"""
        return {
            'cache_dir': 'cache',
            'pre_filt': (0.005, 0.01, 10.0, 20.0),  # Частоты для калибровки
            'water_level': 60,  # Уровень воды для деконволюции
            'min_snr_improvement': 2.0,  # Минимальное улучшение SNR
            'default_filter_band': (1.0, 10.0),  # По умолчанию, если автоподбор не работает
            'fft_window': 'hann',  # Окно для БПФ
            'max_frequency': 20.0,  # Максимальная частота для анализа (Гц)
        }
    
    def process_station(self,
                       raw_data: Dict[str, Any],
                       station_id: str) -> Tuple[StationData, SpectralAnalysis]:
        """
        ОСНОВНОЙ МЕТОД: Полная обработка данных станции
        Реализация интерфейса ISignalProcessor.process_station()
        """
        print(f"[SignalProcessor] Обработка станции {station_id}...")
        
        # 1. Проверка входных данных
        self._validate_raw_data(raw_data)
        
        # 2. Калибровка (удаление инструментального отклика)
        calibrated_data = self._calibrate_data(raw_data, station_id)
        
        # 3. БПФ-анализ для определения оптимального фильтра
        fmin, fmax, spectral_analysis = self.get_optimal_filter_band(
            station_id, calibrated_data, raw_data['sampling_rate']
        )
        
        # 4. Применение bandpass-фильтра через БПФ
        filtered_data = self._apply_bandpass_filter_fft(
            calibrated_data, raw_data['sampling_rate'], fmin, fmax
        )
        
        # 5. Подготовка результата в формате StationData
        station_data = self._create_station_data(
            station_id=station_id,
            filtered_data=filtered_data,
            sampling_rate=raw_data['sampling_rate'],
            coordinates=raw_data['coordinates'],
            raw_data=raw_data
        )
        
        # 6. Сохранение в кэш (для повторного использования)
        self._save_to_cache(station_id, spectral_analysis, fmin, fmax)
        
        print(f"[SignalProcessor] Станция {station_id} обработана. "
              f"Фильтр: {fmin:.2f}-{fmax:.2f} Гц")
        
        return station_data, spectral_analysis