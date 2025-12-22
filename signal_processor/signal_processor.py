import numpy as np
from obspy import UTCDateTime, Trace
from obspy.core.trace import Stats
from typing import Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
import json
import os
from dataclasses import asdict
from interfaces import ISignalProcessor, StationData, SpectralAnalysis

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
            'pre_filt': (0.005, 0.01, 10.0, 40.0),  # Частоты для калибровки
            'water_level': 60,  # Уровень воды для деконволюции
            'min_snr_improvement': 2.0,  # Минимальное улучшение SNR
            'default_filter_band': (1.0, 10.0),  # По умолчанию, если автоподбор не работает
            'fft_window': 'hann',  # Окно для БПФ
            'max_frequency': 40.0,  # Максимальная частота для анализа (Гц)
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
    
    def get_optimal_filter_band(self,
                               station_id: str,
                               raw_data: np.ndarray,
                               sampling_rate: float) -> Tuple[float, float, SpectralAnalysis]:
        """
        ОПРЕДЕЛЕНИЕ ОПТИМАЛЬНОГО ФИЛЬТРА через БПФ-анализ
        Реализация интерфейса ISignalProcessor.get_optimal_filter_band()
        """
        print(f"[SignalProcessor] БПФ-анализ для станции {station_id}...")
        
        # Проверка кэша
        cache_key = f"{station_id}_{len(raw_data)}_{sampling_rate}"
        if cache_key in self.optimal_filters_cache:
            print(f"[SignalProcessor] Используем кэш для {station_id}")
            cached = self.optimal_filters_cache[cache_key]
            return cached['fmin'], cached['fmax'], cached['spectral_analysis']
        
        # 1. Разделение на сегменты: шум и возможный сигнал
        noise_segment, signal_segment = self._extract_noise_and_signal_segments(
            raw_data, sampling_rate
        )
        
        # 2. Вычисление БПФ для шума и сигнала
        fft_noise, freqs = self._compute_fft_with_window(noise_segment, sampling_rate)
        fft_signal, _ = self._compute_fft_with_window(signal_segment, sampling_rate)
        
        # 3. Расчет амплитудных спектров
        amp_noise = np.abs(fft_noise)
        amp_signal = np.abs(fft_signal)
        
        # 4. Вычисление отношения сигнал/шум (SNR)
        snr_ratio = self._calculate_snr_ratio(amp_signal, amp_noise)
        
        # 5. Определение оптимальной полосы частот
        fmin, fmax = self._find_optimal_band_from_snr(snr_ratio, freqs)
        
        # 6. Создание объекта SpectralAnalysis
        spectral_analysis = SpectralAnalysis(
            frequencies=freqs,
            amplitude_raw=amp_signal,
            amplitude_filtered=None,  # Заполнится позже
            noise_spectrum=amp_noise,
            signal_spectrum=amp_signal,
            optimal_band=(fmin, fmax)
        )
        
        # 7. Кэширование результатов
        self.optimal_filters_cache[cache_key] = {
            'fmin': fmin,
            'fmax': fmax,
            'spectral_analysis': spectral_analysis
        }
        
        return fmin, fmax, spectral_analysis
    
    # ============================================================================
    # ВНУТРЕННИЕ МЕТОДЫ ОБРАБОТКИ (Человек 2 знает, другие - нет)
    # ============================================================================
    
    def _validate_raw_data(self, raw_data: Dict[str, Any]) -> None:
        """Проверка корректности сырых данных"""
        required_keys = ['raw_data', 'sampling_rate', 'coordinates', 'inventory']
        for key in required_keys:
            if key not in raw_data:
                raise ValueError(f"Отсутствует обязательное поле: {key}")
        
        if len(raw_data['raw_data']) == 0:
            raise ValueError("Пустой массив данных")
    
    def _calibrate_data(self, raw_data: Dict[str, Any], station_id: str) -> np.ndarray:
        print(f"  [Калибровка] Станция {station_id}")
        
        tr = Trace(data=raw_data['raw_data'].astype(np.float32))
        tr.stats.sampling_rate = raw_data['sampling_rate']
        tr.stats.network = raw_data.get('network', 'XX')
        tr.stats.station = station_id
        tr.stats.location = raw_data.get('location', '')
        tr.stats.channel = raw_data.get('channel', 'BHZ')

        # Используем response, который ты привязала в main.py
        if 'inventory' in raw_data and raw_data['inventory']:
            # Пытаемся взять response напрямую, чтобы избежать ValueError
            try:
                tr.stats.response = raw_data['inventory'][0][0][0].response
            except:
                pass
        
        tr.detrend(type='linear')
        tr.taper(max_percentage=0.05, type='hann')
        
        # ГЛАВНОЕ ИСПРАВЛЕНИЕ: Автоматический расчет безопасных частот
        # Мы берем минимум между твоим конфигом и пределом Найквиста
        nyquist = tr.stats.sampling_rate / 2
        f_top = min(self.config['pre_filt'][2], nyquist - 1.0) 
        f_extreme = min(self.config['pre_filt'][3], nyquist - 0.1)
        
        safe_pre_filt = (
            self.config['pre_filt'][0], 
            self.config['pre_filt'][1], 
            f_top, 
            f_extreme
        )

        # Теперь калибровка не упадет!
        tr.remove_response(
            inventory=None, # Мы уже привязали response выше
            output="VEL",
            pre_filt=safe_pre_filt,
            water_level=self.config.get('water_level', 60)
        )
            
        return tr.data
    
    def _extract_noise_and_signal_segments(self,
                                          data: np.ndarray,
                                          sampling_rate: float,
                                          segment_duration: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Извлечение ОДИНАКОВЫХ по длине сегментов
        """
        # Количество точек в одном сегменте
        n_samples = int(segment_duration * sampling_rate)
        
        # Шум: берем самое начало записи
        noise_segment = data[:n_samples]
        
        # Сигнал: берем кусок такой же длины из середины или конца
        # Чтобы не выйти за границы массива, берем n_samples с конца
        signal_segment = data[-n_samples:]
        
        return noise_segment, signal_segment
    
    def _compute_fft_with_window(self,
                                data: np.ndarray,
                                sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычисление БПФ с применением оконной функции
        """

        window = np.hanning(len(data))
        windowed_data = data * window
        
        # Вычисление БПФ
        n = len(windowed_data)
        fft_data = np.fft.rfft(windowed_data)
        freqs = np.fft.rfftfreq(n, 1/sampling_rate)
        
        # Нормализация по энергии окна
        fft_data = fft_data / np.sum(window)
        
        return fft_data, freqs
    
    def _calculate_snr_ratio(self,
                            signal_spectrum: np.ndarray,
                            noise_spectrum: np.ndarray,
                            epsilon: float = 1e-10) -> np.ndarray:
        """
        Расчет отношения сигнал/шум (SNR) по частотам
        """
        # Избегаем деления на ноль
        noise_spectrum_safe = noise_spectrum.copy()
        noise_spectrum_safe[noise_spectrum_safe < epsilon] = epsilon
        
        snr = signal_spectrum / noise_spectrum_safe
        
        # Сглаживание SNR (скользящее среднее)
        window_size = max(1, len(snr) // 100)
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            snr_smoothed = np.convolve(snr, kernel, mode='same')
        else:
            snr_smoothed = snr
        
        return snr_smoothed
    
    def _find_optimal_band_from_snr(self,
                                   snr_ratio: np.ndarray,
                                   frequencies: np.ndarray,
                                   min_snr: float = 2.0) -> Tuple[float, float]:
        """
        Определение оптимальной полосы частот на основе SNR
        """
        # Ограничиваем анализ разумным диапазоном частот
        max_freq = self.config['max_frequency']
        valid_mask = frequencies <= max_freq
        freqs_valid = frequencies[valid_mask]
        snr_valid = snr_ratio[:len(freqs_valid)]
        
        if len(snr_valid) == 0:
            print("[SignalProcessor] Не удалось определить оптимальный фильтр, используем по умолчанию")
            return self.config['default_filter_band']
        
        # Находим частоты, где SNR превышает порог
        above_threshold = snr_valid > min_snr
        
        if not np.any(above_threshold):
            # Если ни одна частота не превышает порог, берем частоту с максимальным SNR
            best_idx = np.argmax(snr_valid)
            best_freq = freqs_valid[best_idx]
            fmin = max(0.1, best_freq * 0.5)
            fmax = min(max_freq, best_freq * 2.0)
        else:
            # Берем диапазон, где SNR выше порога
            threshold_indices = np.where(above_threshold)[0]
            fmin = freqs_valid[threshold_indices[0]]
            fmax = freqs_valid[threshold_indices[-1]]
            
            # Расширяем диапазон на 20% для уверенности
            bandwidth = fmax - fmin
            fmin = max(0.1, fmin - 0.2 * bandwidth)
            fmax = min(max_freq, fmax + 0.2 * bandwidth)
        
        # Обеспечиваем минимальную ширину полосы
        if fmax - fmin < 1.0:
            center = (fmin + fmax) / 2
            fmin = max(0.1, center - 0.5)
            fmax = min(max_freq, center + 0.5)

        if fmax < 3.0:
            fmax = 4.0
        
        return float(fmin), float(fmax)
    
    def _apply_bandpass_filter_fft(self,
                                  data: np.ndarray,
                                  sampling_rate: float,
                                  fmin: float,
                                  fmax: float) -> np.ndarray:
        """
        Применение bandpass-фильтра через БПФ (более точный метод)
        """
        print(f"  [Фильтрация] Применяем фильтр {fmin:.2f}-{fmax:.2f} Гц")
        
        # 1. БПФ исходного сигнала
        n = len(data)
        fft_data = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(n, 1/sampling_rate)
        
        # 2. Создание частотной маски с плавными краями
        mask = self._create_filter_mask(freqs, fmin, fmax)
        
        # 3. Применение маски
        fft_filtered = fft_data * mask
        
        # 4. Обратное БПФ
        filtered_data = np.fft.irfft(fft_filtered, n)
        
        # 5. Детрендинг результата (убираем артефакты на краях)
        filtered_data = filtered_data - np.mean(filtered_data)
        
        return filtered_data
    
    def _create_filter_mask(self,
                           frequencies: np.ndarray,
                           fmin: float,
                           fmax: float,
                           transition_width: float = 0.5) -> np.ndarray:
        """
        Создание маски фильтра с плавными переходами
        """
        mask = np.zeros_like(frequencies, dtype=np.float32)
        
        # Определяем переходные зоны
        lower_transition_start = max(0, fmin - transition_width)
        lower_transition_end = fmin + transition_width
        
        upper_transition_start = fmax - transition_width
        upper_transition_end = fmax + transition_width
        
        for i in range(len(frequencies)):
            freq = frequencies[i]
            if freq < lower_transition_start:
                mask[i] = 0.0
            elif freq < lower_transition_end:
                # Косинусное окно для плавного подъема
                x = (freq - lower_transition_start) / (2 * transition_width)
                mask[i] = 0.5 - 0.5 * np.cos(np.pi * x)
            elif freq <= upper_transition_start:
                mask[i] = 1.0
            elif freq < upper_transition_end:
                # Косинусное окно для плавного спада
                x = (freq - upper_transition_start) / (2 * transition_width)
                mask[i] = 0.5 + 0.5 * np.cos(np.pi * x)
            else:
                mask[i] = 0.0
        
        return mask
    
    def _create_station_data(self,
                            station_id: str,
                            filtered_data: np.ndarray,
                            sampling_rate: float,
                            coordinates: Tuple[float, float],
                            raw_data: Dict[str, Any]) -> StationData:
        """
        Создание объекта StationData в согласованном формате
        """
        return StationData(
            station_id=station_id,
            z_data=filtered_data,
            n_data=raw_data.get('n_data'),
            e_data=raw_data.get('e_data'),
            sampling_rate=sampling_rate,
            coordinates=coordinates,
            elevation=raw_data.get('elevation'),
            processed=True
        )
    
    def _save_to_cache(self,
                      station_id: str,
                      spectral_analysis: SpectralAnalysis,
                      fmin: float,
                      fmax: float) -> None:
        """
        Сохранение результатов обработки в кэш
        """
        cache_file = os.path.join(self.cache_dir, f"{station_id}_analysis.json")
        
        # Преобразуем SpectralAnalysis в словарь
        cache_data = {
            'station_id': station_id,
            'fmin': fmin,
            'fmax': fmax,
            'optimal_band': spectral_analysis.optimal_band,
            'timestamp': str(UTCDateTime())
        }
        
        # Сохраняем основные частоты спектра (первые 100 точек для экономии места)
        save_freqs = spectral_analysis.frequencies[:100]
        save_noise = spectral_analysis.noise_spectrum[:100]
        save_signal = spectral_analysis.signal_spectrum[:100]
        
        cache_data['frequencies'] = save_freqs.tolist()
        cache_data['noise_spectrum'] = save_noise.tolist()
        cache_data['signal_spectrum'] = save_signal.tolist()
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)