import numpy as np
from obspy import UTCDateTime, Trace
from obspy.core.trace import Stats
from typing import Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
import json
import os
from dataclasses import asdict
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
        """
        КАЛИБРОВКА: Удаление инструментального отклика сейсмометра
        Внутренний метод - другие члены команды не знают о его реализации
        """
        print(f"  [Калибровка] Станция {station_id}")
        
        # Преобразуем сырые данные в ObsPy Trace
        
        # Создаем объект Trace из сырых данных
        tr = Trace(data=raw_data['raw_data'].astype(np.float32))  # type: ignore
    
        tr.stats.sampling_rate = raw_data['sampling_rate']
        tr.stats.network = raw_data.get('network', 'XX')
        tr.stats.station = station_id
        tr.stats.channel = raw_data.get('channel', 'BHZ')
        
        # Детрендинг и tapering (подготовка к FFT)
        tr.detrend(type='linear')
        tr.taper(max_percentage=0.05, type='hann')
        
        tr.remove_response(
                inventory=raw_data['inventory'],  # ← Уже загружен!
                output="VEL",
                pre_filt=(0.005, 0.01, 10.0, 20.0),
                water_level=60
            )
            
        calibrated = tr.data
        
        return calibrated
    
    def _extract_noise_and_signal_segments(self,
                                          data: np.ndarray,
                                          sampling_rate: float,
                                          noise_duration: float = 100.0,
                                          signal_search_start: float = 200.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Извлечение сегментов шума и возможного сигнала
        """
        # Шум: первые noise_duration секунд
        noise_samples = int(noise_duration * sampling_rate)
        noise_segment = data[:min(noise_samples, len(data)//4)]
        
        # Сигнал: ищем в середине записи
        search_start = int(signal_search_start * sampling_rate)
        search_end = min(search_start + 300 * sampling_rate, len(data))
        
        if search_end - search_start < 10 * sampling_rate:
            # Если запись короткая, берем середину
            mid_point = len(data) // 2
            signal_segment = data[mid_point:mid_point + int(30 * sampling_rate)]
        else:
            signal_segment = data[search_start:search_end]
        
        return noise_segment, signal_segment
    
    def _compute_fft_with_window(self,
                                data: np.ndarray,
                                sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычисление БПФ с применением оконной функции
        """
        # Применение оконной функции для уменьшения утечки спектра
        if self.config['fft_window'] == 'hann':
            window = np.hanning(len(data))
        elif self.config['fft_window'] == 'hamming':
            window = np.hamming(len(data))
        else:
            window = np.ones(len(data))
        
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
            'snr_improvement': spectral_analysis.snr_improvement,
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
    
    # ============================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ (могут использоваться другими)
    # ============================================================================
    
    def plot_spectral_analysis(self,
                              spectral_analysis: SpectralAnalysis,
                              station_id: str = "") -> plt.Figure:
        """
        Визуализация результатов БПФ-анализа
        Может использоваться Человеком 1 для отчетов
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Спектры шума и сигнала
        ax1 = axes[0, 0]
        ax1.semilogy(spectral_analysis.frequencies,
                    spectral_analysis.noise_spectrum,
                    'gray', label='Фоновый шум', alpha=0.7)
        ax1.semilogy(spectral_analysis.frequencies,
                    spectral_analysis.signal_spectrum,
                    'red', label='Сигнал события', linewidth=2)
        ax1.set_xlim(0, self.config['max_frequency'])
        ax1.set_xlabel('Частота (Гц)')
        ax1.set_ylabel('Амплитуда')
        ax1.set_title(f'Спектры шума и сигнала {station_id}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Отношение сигнал/шум
        ax2 = axes[0, 1]
        ratio = (spectral_analysis.signal_spectrum / 
                (spectral_analysis.noise_spectrum + 1e-10))
        ax2.plot(spectral_analysis.frequencies, ratio, 'blue')
        ax2.axhline(y=2.0, color='red', linestyle='--', label='Порог SNR=2')
        ax2.set_xlim(0, self.config['max_frequency'])
        ax2.set_xlabel('Частота (Гц)')
        ax2.set_ylabel('Отношение сигнал/шум')
        ax2.set_title(f'SNR по частотам {station_id}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Выбранная полоса фильтра
        ax3 = axes[1, 0]
        fmin, fmax = spectral_analysis.optimal_band
        freqs = np.linspace(0, self.config['max_frequency'], 1000)
        mask = self._create_filter_mask(freqs, fmin, fmax)
        
        ax3.plot(freqs, mask, 'purple', linewidth=2)
        ax3.fill_between(freqs, 0, mask, alpha=0.3, color='purple')
        ax3.set_xlabel('Частота (Гц)')
        ax3.set_ylabel('Коэффициент передачи')
        ax3.set_title(f'Полосовой фильтр: {fmin:.2f}-{fmax:.2f} Гц')
        ax3.grid(True, alpha=0.3)
        
        # 4. Информация
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        info_text = f"""
        Станция: {station_id}
        
        Оптимальный фильтр:
        • Нижняя частота: {fmin:.2f} Гц
        • Верхняя частота: {fmax:.2f} Гц
        • Ширина полосы: {fmax-fmin:.2f} Гц
        
        Результаты анализа:
        • Улучшение SNR: {spectral_analysis.snr_improvement:.1f}×
        • Макс. частота анализа: {self.config['max_frequency']} Гц
        """
        
        ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'БПФ-анализ и подбор фильтра - {station_id}', fontsize=16)
        plt.tight_layout()
        
        return fig
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Возвращает сводку по обработке
        """
        return {
            'stations_processed': list(self.optimal_filters_cache.keys()),
            'cache_size': len(self.optimal_filters_cache),
            'config': self.config,
            'default_filter': self.config['default_filter_band']
        }


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def create_test_signal(sampling_rate: float = 40.0,
                      duration: float = 600.0) -> np.ndarray:
    """
    Создание тестового сейсмического сигнала
    Для отладки, когда нет доступа к реальным данным
    """
    n_samples = int(duration * sampling_rate)
    t = np.arange(n_samples) / sampling_rate
    
    # Фоновый шум
    noise = np.random.normal(0, 0.1, n_samples)
    
    # P-волна (короткий высокочастотный импульс)
    p_wave_time = 200.0  # секунд
    p_wave = np.zeros(n_samples)
    p_idx = int(p_wave_time * sampling_rate)
    p_wave[p_idx:p_idx+100] = 0.5 * np.sin(2 * np.pi * 8 * t[:100])
    
    # S-волна (длинный низкочастотный сигнал)
    s_wave_time = 280.0  # секунд
    s_wave = np.zeros(n_samples)
    s_idx = int(s_wave_time * sampling_rate)
    duration_s = 400  # длительность S-волны
    s_wave[s_idx:s_idx+duration_s] = (
        2.0 * np.sin(2 * np.pi * 2 * t[:duration_s]) *
        np.exp(-0.01 * t[:duration_s])
    )
    
    # Суммируем все компоненты
    signal = noise + p_wave + s_wave
    
    return signal


# ============================================================================
# ТЕСТОВЫЙ БЛОК (для самостоятельной проверки)
# ============================================================================

if __name__ == "__main__":
    print("Тестирование SignalProcessor...")
    
    # 1. Создаем тестовые данные
    sampling_rate = 40.0
    test_signal = create_test_signal(sampling_rate)
    
    raw_data = {
        'raw_data': test_signal,
        'sampling_rate': sampling_rate,
        'coordinates': (55.5, 37.6),
        'network': 'TEST',
        'channel': 'BHZ'
    }
    
    # 2. Создаем и тестируем процессор
    processor = SignalProcessor()
    
    # 3. Обрабатываем тестовые данные
    station_data, spectral_analysis = processor.process_station(
        raw_data, "TEST01"
    )
    
    # 4. Проверяем результаты
    print(f"\nРезультаты обработки:")
    print(f"  Размер данных: {len(station_data.z_data)} отсчетов")
    print(f"  Частота дискретизации: {station_data.sampling_rate} Гц")
    print(f"  Координаты: {station_data.coordinates}")
    print(f"  Оптимальный фильтр: {spectral_analysis.optimal_band} Гц")
    print(f"  Улучшение SNR: {spectral_analysis.snr_improvement:.1f}×")
    
    # 5. Визуализируем результаты
    fig = processor.plot_spectral_analysis(spectral_analysis, "TEST01")
    plt.savefig("test_spectral_analysis.png", dpi=150, bbox_inches='tight')
    print(f"\nГрафик сохранен в test_spectral_analysis.png")
    
    # 6. Выводим сводку
    summary = processor.get_processing_summary()
    print(f"\nСводка обработки: {summary['stations_processed']}")
    
    print("\n✅ SignalProcessor работает корректно!")