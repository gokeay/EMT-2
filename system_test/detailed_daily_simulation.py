"""
🔍 EMT RL Project - Detailed Daily Simulation
System Test: 1000 saatlik model karar analizi
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Path setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.ppo_agent import PPOAgent
from src.environment.energy_environment import EnergyEnvironment

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealDataHandler:
    """
    System test için gerçek veri yükleme sınıfı - Mock veri yok!
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize Real Data Handler
        
        Args:
            data_dir: Data dizini
        """
        self.data_dir = data_dir
        self.combined_data = None
        logger.info(f"🔧 RealDataHandler başlatıldı - Data dir: {data_dir}")
    
    def load_real_data(self) -> bool:
        """Gerçek CSV dosyalarından veri yükle"""
        try:
            logger.info("📂 Gerçek CSV dosyaları yükleniyor...")
            
            # CSV dosya yolları
            load_file = os.path.join(self.data_dir, "/content/EMT-2/data/synthetic_load_itu.csv")
            solar_file = os.path.join(self.data_dir, "/content/EMT-2/data/sim_solar_gen_result.csv")
            wind_file = os.path.join(self.data_dir, "/content/EMT-2/data/sim_wind_gen_result.csv")
            
            # Dosya varlık kontrolü
            for file_path in [load_file, solar_file, wind_file]:
                if not os.path.exists(file_path):
                    logger.error(f"❌ Dosya bulunamadı: {file_path}")
                    return False
            
            # Load data
            logger.info("📊 Load verisi yükleniyor...")
            load_data = pd.read_csv(load_file)
            load_data['datetime'] = pd.to_datetime(load_data['datetime'])
            logger.info(f"✅ Load: {len(load_data)} satır")
            
            # Solar data
            logger.info("☀️ Solar verisi yükleniyor...")
            solar_data = pd.read_csv(solar_file)
            solar_data['datetime'] = pd.to_datetime(solar_data['datetime'])
            logger.info(f"✅ Solar: {len(solar_data)} satır")
            
            # Wind data
            logger.info("💨 Wind verisi yükleniyor...")
            wind_data = pd.read_csv(wind_file)
            wind_data['datetime'] = pd.to_datetime(wind_data['datetime'])
            logger.info(f"✅ Wind: {len(wind_data)} satır")
            
            # Verileri birleştir
            logger.info("🔗 Veriler birleştiriliyor...")
            combined = load_data.merge(
                solar_data, on='datetime', how='inner'
            ).merge(
                wind_data, on='datetime', how='inner'
            )
            
            # Ek sütunlar
            combined['renewable_total_kW'] = (
                combined['solar_power_kW'] + combined['wind_power_kW']
            )
            combined['hour'] = combined['datetime'].dt.hour
            
            # Price categories (saat bazlı)
            def get_price_category(hour):
                if 22 <= hour or hour < 8:  # Gece (22:00-08:00)
                    return 'low', 1, 0, 0
                elif 8 <= hour < 18:  # Gündüz (08:00-18:00)
                    return 'medium', 0, 1, 0
                else:  # Peak (18:00-22:00)
                    return 'high', 0, 0, 1
            
            price_info = combined['hour'].apply(get_price_category)
            combined['price_category'] = [p[0] for p in price_info]
            combined['price_low'] = [p[1] for p in price_info]
            combined['price_medium'] = [p[2] for p in price_info]
            combined['price_high'] = [p[3] for p in price_info]
            
            self.combined_data = combined
            
            logger.info(f"✅ Gerçek veri yüklendi: {len(combined)} satır")
            logger.info(f"📅 Tarih aralığı: {combined['datetime'].min()} - {combined['datetime'].max()}")
            logger.info(f"📊 Load ortalama: {combined['load_kw'].mean():.1f} kW")
            logger.info(f"☀️ Solar ortalama: {combined['solar_power_kW'].mean():.1f} kW")
            logger.info(f"💨 Wind ortalama: {combined['wind_power_kW'].mean():.1f} kW")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Gerçek veri yükleme hatası: {e}")
            return False
    
    def get_episode_data(self, episode_length: int = 24) -> pd.DataFrame:
        """Episode için veri döndür"""
        if self.combined_data is None:
            raise ValueError("❌ Veri henüz yüklenmedi!")
        
        if len(self.combined_data) < episode_length:
            logger.warning(f"⚠️ İstenen episode uzunluğu ({episode_length}) mevcut veriden ({len(self.combined_data)}) büyük!")
            return self.combined_data.copy()
        
        # Veri setinin başından başla
        episode_data = self.combined_data.iloc[0:episode_length].copy()
        episode_data = episode_data.reset_index(drop=True)
        
        logger.info(f"📊 Episode verisi hazırlandı: {len(episode_data)} satır")
        logger.info(f"🕐 Başlangıç saati: {episode_data['datetime'].iloc[0]}")
        
        return episode_data
    
    def load_all_data(self) -> bool:
        """Environment uyumluluğu için - gerçek veri zaten yüklü"""
        return self.combined_data is not None


class DetailedDailySimulation:
    """
    1000 saatlik detaylı model simulasyon ve analiz sınıfı
    """
    
    def __init__(self, model_path: str, config_path: str = "configs/config.yaml"):
        """
        Initialize Detailed Daily Simulation
        
        Args:
            model_path: Trained model dosya yolu
            config_path: Config dosya yolu
        """
        self.model_path = model_path
        self.config_path = config_path
        self.output_dir = "system_test/results/"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Components
        self.agent = None
        self.environment = None
        self.data_handler = None
        
        # Simulation results storage
        self.simulation_data = []
        
        logger.info(f"🔍 DetailedDailySimulation başlatıldı - Model: {model_path}")
    
    def setup_simulation(self) -> bool:
        """Simulation için gerekli componentleri hazırla - GERÇEK VERİ İLE"""
        try:
            logger.info("🔧 Simulation setup başlıyor...")
            
            # 1. Real Data Handler oluştur
            self.data_handler = RealDataHandler()
            if not self.data_handler.load_real_data():
                logger.error("❌ Gerçek veri yüklenemedi!")
                return False
            
            # 2. Environment oluştur
            logger.info("🏗️ Environment oluşturuluyor...")
            self.environment = EnergyEnvironment()
            
            # Environment'a gerçek veri ver
            episode_data = self.data_handler.get_episode_data(8760)  # 1 yıllık veri
            self.environment.data_handler = self.data_handler
            self.environment.current_data = episode_data
            
            logger.info("✅ Environment hazır - Gerçek veri ile")
            
            # 3. Agent oluştur ve model yükle
            logger.info("🤖 PPO Agent oluşturuluyor...")
            self.agent = PPOAgent(environment=self.environment)
            self.agent.create_model()
            
            if os.path.exists(self.model_path):
                self.agent.load_model(self.model_path)
                logger.info(f"✅ Model yüklendi: {self.model_path}")
            else:
                logger.error(f"❌ Model dosyası bulunamadı: {self.model_path}")
                return False
            
            logger.info("✅ Simulation setup tamamlandı - GERÇEK VERİ İLE!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Simulation setup hatası: {e}")
            return False
    
    def run_1000hour_simulation(self, deterministic: bool = True) -> pd.DataFrame:
        """
        1000 saatlik detaylı simulasyon çalıştır
        
        Args:
            deterministic: Deterministic model prediction kullan
            
        Returns:
            pd.DataFrame: Detaylı simulation sonuçları
        """
        logger.info("🚀 1000 saatlik detaylı simulasyon başlıyor...")
        
        try:
            # Fresh 1000 saatlik data al
            episode_data = self.data_handler.get_episode_data(1000)
            
            # Environment'a bu veriyi ver
            self.environment.current_data = episode_data
            self.environment.current_step = 0
            self.environment.episode_length = 8760
            
            # Environment reset
            obs, info = self.environment.reset()
            
            # İlk durum bilgilerini kaydet
            initial_soc = obs[3]  # Battery SOC
            logger.info(f"📊 Başlangıç SOC: {initial_soc:.1%}")
            
            # Gerçek verinin ilk satırını logla
            first_row = episode_data.iloc[0]
            logger.info(f"📅 Başlangıç: {first_row['datetime']}")
            logger.info(f"📊 Load: {first_row['load_kw']:.1f} kW")
            logger.info(f"☀️ Solar: {first_row['solar_power_kW']:.1f} kW")
            logger.info(f"💨 Wind: {first_row['wind_power_kW']:.1f} kW")
            
            # 1000 saatlik simulasyon
            for hour in range(1000):
                #logger.info(f"⏰ Saat {hour+1}/1000 simüle ediliyor...")
                
                # Current step data
                current_row = episode_data.iloc[hour]
                
                # Model prediction - AI kararı
                action, prediction_info = self.agent.predict(obs, deterministic=deterministic)
                
                # Environment step - aksiyonu uygula
                next_obs, reward, terminated, truncated, step_info = self.environment.step(action)
                
                # Reward breakdown analizi
                reward_breakdown = self._analyze_reward_breakdown(
                    obs, action, reward, step_info
                )
                
                # Comprehensive step data
                step_data = {
                    # Time info
                    'Hour': hour + 1,
                    'Simulation_Time': f"{hour:02d}:00",
                    'Real_DateTime': str(current_row['datetime']),
                    
                    # State Space (Observation) - GERÇEK VERİ
                    'Load_kW': float(current_row['load_kw']),
                    'Solar_Generation_kW': float(current_row['solar_power_kW']),
                    'Wind_Generation_kW': float(current_row['wind_power_kW']),
                    'Battery_SOC_%': float(obs[3]) * 100,
                    'Price_Low': float(obs[4]),
                    'Price_Medium': float(obs[5]),
                    'Price_High': float(obs[6]),
                    
                    # Derived state info
                    'Total_Renewable_kW': float(current_row['solar_power_kW'] + current_row['wind_power_kW']),
                    'Current_Price_Category': current_row['price_category'],
                    'Renewable_Coverage_%': min(100, (float(current_row['solar_power_kW'] + current_row['wind_power_kW']) / float(current_row['load_kw'])) * 100) if current_row['load_kw'] > 0 else 0,
                    
                    # Action Space (AI Decisions)
                    'Grid_Energy_kW': float(action[0]),
                    'Battery_Power_kW': float(action[1]),
                    'Battery_Action': 'Charge' if action[1] > 0 else 'Discharge' if action[1] < 0 else 'Idle',
                    
                    # Energy Balance
                    'Energy_Demand_kW': float(current_row['load_kw']),
                    'Energy_Supply_Total_kW': float(current_row['solar_power_kW'] + current_row['wind_power_kW'] + action[0]),
                    'Energy_Balance_kW': float(current_row['solar_power_kW'] + current_row['wind_power_kW'] + action[0] - current_row['load_kw']),
                    
                    # Reward Breakdown
                    'Renewable_Bonus': reward_breakdown['renewable_bonus'],
                    'SOC_Penalty': reward_breakdown['soc_penalty'],
                    'Grid_Cost': reward_breakdown['grid_cost'],
                    'Total_Reward': float(reward),
                    
                    # Post-step info
                    'Next_SOC_%': float(next_obs[3]) * 100,
                    'SOC_Change_%': (float(next_obs[3]) - float(obs[3])) * 100,
                    
                    # Decision efficiency metrics
                    'Renewable_Utilization_%': min(100, (float(current_row['solar_power_kW'] + current_row['wind_power_kW']) / max(1, float(current_row['load_kw']))) * 100),
                    'Grid_Dependency_%': (float(action[0]) / max(1, float(current_row['load_kw']))) * 100,
                    'Battery_Efficiency': abs(float(action[1])) / 2000 * 100,  # Relative to max power
                    
                    # Episode info
                    'Episode_Step': hour + 1,
                    'Terminated': terminated,
                    'Truncated': truncated,
                    'Data_Source': 'REAL_CSV'
                }
                
                # Add to simulation data
                self.simulation_data.append(step_data)
                
                # Update observation for next step
                obs = next_obs
                
                # Check termination
                if terminated or truncated:
                    logger.warning(f"⚠️ Episode terminated at hour {hour+1}")
                    break
            
            # Convert to DataFrame
            df_results = pd.DataFrame(self.simulation_data)
            
            logger.info("✅ 1000 saatlik simulasyon tamamlandı!")
            logger.info("📊 VERİ KAYNAĞI: GERÇEK CSV DOSYALARI")
            return df_results
            
        except Exception as e:
            logger.error(f"❌ Simulasyon hatası: {e}")
            raise
    
    def _analyze_reward_breakdown(self, obs: np.ndarray, action: np.ndarray, 
                                total_reward: float, step_info: Dict) -> Dict:
        """Reward breakdown detaylı analizi"""
        
        # Environment'dan reward component'lerini almaya çalış
        try:
            # Manual reward calculation (environment logic'ini taklit et)
            load = float(obs[0])
            solar = float(obs[1]) 
            wind = float(obs[2])
            soc = float(obs[3])
            grid_energy = float(action[0])
            
            renewable_total = solar + wind
            renewable_usage = min(renewable_total, load)
            
            # Reward components (environment'dan alınması gereken değerler)
            renewable_bonus = renewable_usage * 50  # Renewable bonus weight
            
            # SOC penalty calculation
            soc_penalty = 0.0
            if soc < 0.2:  # Below min SOC
                soc_penalty = (0.2 - soc) * 100 * 100  # Heavy penalty
            elif soc > 0.9:  # Above max SOC
                soc_penalty = (soc - 0.9) * 100 * 100  # Heavy penalty
            
            # Grid cost calculation
            # Price determination based on hour (simplified)
            hour = len(self.simulation_data)
            if 22 <= hour or hour < 8:  # Night
                price = 0.12
            elif 8 <= hour < 18:  # Day
                price = 0.20
            else:  # Peak (18-22)
                price = 0.31
            
            grid_cost = grid_energy * price * 10  # Grid cost weight
            
            return {
                'renewable_bonus': renewable_bonus,
                'soc_penalty': -soc_penalty,
                'grid_cost': -grid_cost,
                'calculated_total': renewable_bonus - soc_penalty - grid_cost,
                'actual_total': total_reward
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Reward breakdown hesaplama hatası: {e}")
            return {
                'renewable_bonus': 0.0,
                'soc_penalty': 0.0,
                'grid_cost': 0.0,
                'calculated_total': 0.0,
                'actual_total': total_reward
            }
    
    def save_results(self, df_results: pd.DataFrame) -> Dict[str, str]:
        """Simulation sonuçlarını kaydet"""
        logger.info("💾 Simulation sonuçları kaydediliyor...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = {}
        
        try:
            # 1. Detailed CSV
            csv_path = os.path.join(self.output_dir, f"detailed_simulation_{timestamp}.csv")
            df_results.to_csv(csv_path, index=False, float_format='%.2f')
            saved_files['detailed_csv'] = csv_path
            
            # 2. Summary table (formatted)
            summary_path = os.path.join(self.output_dir, f"simulation_summary_{timestamp}.txt")
            self._create_formatted_summary(df_results, summary_path)
            saved_files['summary_txt'] = summary_path
            
            # 3. Decision analysis
            analysis_path = os.path.join(self.output_dir, f"decision_analysis_{timestamp}.txt")
            self._create_decision_analysis(df_results, analysis_path)
            saved_files['analysis_txt'] = analysis_path
            
            # 4. Hourly breakdown table
            hourly_path = os.path.join(self.output_dir, f"hourly_breakdown_{timestamp}.txt")
            self._create_hourly_table(df_results, hourly_path)
            saved_files['hourly_table'] = hourly_path
            
            logger.info(f"✅ Sonuçlar kaydedildi - {len(saved_files)} dosya")
            return saved_files
            
        except Exception as e:
            logger.error(f"❌ Sonuç kaydetme hatası: {e}")
            return {}
    
    def _create_formatted_summary(self, df: pd.DataFrame, file_path: str):
        """Formatted summary raporu oluştur"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("🔍 EMT RL PROJECT - 1000 HOUR DETAILED SIMULATION SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"📅 Simulation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"🤖 Model: {os.path.basename(self.model_path)}\n")
            f.write(f"⏰ Duration: 1000 hours ({len(df)} steps)\n\n")
            
            # Overall statistics
            f.write("📊 OVERALL STATISTICS\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total Energy Consumption: {df['Load_kW'].sum():,.0f} kWh\n")
            f.write(f"Total Solar Generation: {df['Solar_Generation_kW'].sum():,.0f} kWh\n")
            f.write(f"Total Wind Generation: {df['Wind_Generation_kW'].sum():,.0f} kWh\n")
            f.write(f"Total Renewable: {df['Total_Renewable_kW'].sum():,.0f} kWh\n")
            f.write(f"Total Grid Usage: {df['Grid_Energy_kW'].sum():,.0f} kWh\n")
            f.write(f"Average SOC: {df['Battery_SOC_%'].mean():.1f}%\n")
            f.write(f"SOC Range: {df['Battery_SOC_%'].min():.1f}% - {df['Battery_SOC_%'].max():.1f}%\n")
            f.write(f"Total Reward: {df['Total_Reward'].sum():,.0f}\n")
            f.write(f"Average Renewable Coverage: {df['Renewable_Coverage_%'].mean():.1f}%\n\n")
            
            # Energy efficiency
            renewable_total = df['Total_Renewable_kW'].sum()
            load_total = df['Load_kW'].sum()
            grid_total = df['Grid_Energy_kW'].sum()
            renewable_efficiency = (renewable_total / load_total) * 100 if load_total > 0 else 0
            grid_dependency = (grid_total / load_total) * 100 if load_total > 0 else 0
            
            f.write("⚡ ENERGY EFFICIENCY METRICS\n")
            f.write("-" * 50 + "\n")
            f.write(f"Renewable Efficiency: {renewable_efficiency:.1f}%\n")
            f.write(f"Grid Dependency: {grid_dependency:.1f}%\n")
            f.write(f"Battery Utilization: {df['Battery_Efficiency'].mean():.1f}%\n\n")
            
            # Price distribution
            price_counts = df['Current_Price_Category'].value_counts()
            f.write("💰 PRICE PERIOD DISTRIBUTION\n")
            f.write("-" * 50 + "\n")
            for price, count in price_counts.items():
                f.write(f"{price} Price Hours: {count}\n")
            f.write("\n")
            
            # Battery behavior
            charge_hours = len(df[df['Battery_Action'] == 'Charge'])
            discharge_hours = len(df[df['Battery_Action'] == 'Discharge'])
            idle_hours = len(df[df['Battery_Action'] == 'Idle'])
            
            f.write("🔋 BATTERY BEHAVIOR\n")
            f.write("-" * 50 + "\n")
            f.write(f"Charging Hours: {charge_hours}\n")
            f.write(f"Discharging Hours: {discharge_hours}\n")
            f.write(f"Idle Hours: {idle_hours}\n")
            f.write(f"Net Battery Change: {df['SOC_Change_%'].sum():.2f}%\n\n")
    
    def _create_decision_analysis(self, df: pd.DataFrame, file_path: str):
        """Decision analysis raporu oluştur"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("🧠 EMT RL PROJECT - DECISION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("💡 AI DECISION PATTERNS\n")
            f.write("-" * 50 + "\n")
            
            # Smart decisions analysis
            smart_decisions = 0
            total_decisions = len(df)
            
            for _, row in df.iterrows():
                # Check if AI made smart decisions
                if row['Current_Price_Category'] == 'High' and row['Battery_Action'] == 'Discharge':
                    smart_decisions += 1
                elif row['Current_Price_Category'] == 'Low' and row['Battery_Action'] == 'Charge':
                    smart_decisions += 1
                elif row['Renewable_Coverage_%'] > 100 and row['Battery_Action'] == 'Charge':
                    smart_decisions += 1
            
            decision_intelligence = (smart_decisions / total_decisions) * 100
            f.write(f"Decision Intelligence Score: {decision_intelligence:.1f}%\n")
            f.write(f"Smart Decisions: {smart_decisions}/{total_decisions}\n\n")
            
            # Peak shaving analysis
            peak_hours = df[df['Current_Price_Category'] == 'High']
            if len(peak_hours) > 0:
                avg_grid_peak = peak_hours['Grid_Energy_kW'].mean()
                avg_battery_peak = peak_hours['Battery_Power_kW'].mean()
                f.write("⚡ PEAK HOUR BEHAVIOR\n")
                f.write("-" * 30 + "\n")
                f.write(f"Average Grid Usage in Peak: {avg_grid_peak:.1f} kW\n")
                f.write(f"Average Battery Power in Peak: {avg_battery_peak:.1f} kW\n")
                f.write(f"Peak Shaving Strategy: {'Active' if avg_battery_peak < -50 else 'Minimal'}\n\n")
            
            # Renewable optimization
            high_renewable_hours = df[df['Renewable_Coverage_%'] > 100]
            if len(high_renewable_hours) > 0:
                excess_renewable = high_renewable_hours['Total_Renewable_kW'].sum() - high_renewable_hours['Load_kW'].sum()
                battery_utilization = high_renewable_hours[high_renewable_hours['Battery_Action'] == 'Charge']
                
                f.write("🌱 RENEWABLE OPTIMIZATION\n")
                f.write("-" * 30 + "\n")
                f.write(f"Hours with Excess Renewable: {len(high_renewable_hours)}\n")
                f.write(f"Total Excess Renewable: {excess_renewable:.1f} kWh\n")
                f.write(f"Battery Charging in Excess Hours: {len(battery_utilization)}\n")
                f.write(f"Renewable Utilization Strategy: {'Optimized' if len(battery_utilization) > len(high_renewable_hours)*0.5 else 'Suboptimal'}\n\n")
    
    def _create_hourly_table(self, df: pd.DataFrame, file_path: str):
        """Saatlik detay tablosu oluştur"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("📊 EMT RL PROJECT - HOURLY DECISION BREAKDOWN\n")
            f.write("=" * 120 + "\n\n")
            
            # Table header
            header = (
                f"{'Hour':<4} {'Time':<5} {'Load':<8} {'Solar':<8} {'Wind':<8} {'SOC%':<6} "
                f"{'Price':<6} {'Grid':<8} {'Batt':<8} {'Action':<9} {'RenBonus':<9} "
                f"{'SOCPen':<8} {'GridCost':<9} {'Reward':<8}\n"
            )
            f.write(header)
            f.write("-" * 120 + "\n")
            
            # Table rows
            for _, row in df.iterrows():
                row_str = (
                    f"{int(row['Hour']):<4} "
                    f"{row['Simulation_Time']:<5} "
                    f"{row['Load_kW']:<8.0f} "
                    f"{row['Solar_Generation_kW']:<8.0f} "
                    f"{row['Wind_Generation_kW']:<8.0f} "
                    f"{row['Battery_SOC_%']:<6.1f} "
                    f"{row['Current_Price_Category']:<6} "
                    f"{row['Grid_Energy_kW']:<8.0f} "
                    f"{row['Battery_Power_kW']:<8.0f} "
                    f"{row['Battery_Action']:<9} "
                    f"{row['Renewable_Bonus']:<9.0f} "
                    f"{row['SOC_Penalty']:<8.0f} "
                    f"{row['Grid_Cost']:<9.0f} "
                    f"{row['Total_Reward']:<8.0f}\n"
                )
                f.write(row_str)
            
            f.write("\n" + "=" * 120 + "\n")
            f.write("LEGEND:\n")
            f.write("- Load/Solar/Wind/Grid/Batt: kW values\n") 
            f.write("- SOC%: Battery State of Charge percentage\n")
            f.write("- Price: Low/Medium/High price category\n")
            f.write("- Action: Battery Charge/Discharge/Idle\n")
            f.write("- RenBonus: Renewable energy bonus points\n")
            f.write("- SOCPen: SOC violation penalty points\n")
            f.write("- GridCost: Grid energy cost points\n")
            f.write("- Reward: Total reward = RenBonus - SOCPen - GridCost\n")
    
    def generate_insights(self, df: pd.DataFrame) -> str:
        """Simulation insights oluştur"""
        insights = []
        
        # Energy efficiency insights
        renewable_efficiency = (df['Total_Renewable_kW'].sum() / df['Load_kW'].sum()) * 100
        if renewable_efficiency > 90:
            insights.append(f"🌱 Excellent renewable utilization: {renewable_efficiency:.1f}%")
        elif renewable_efficiency > 70:
            insights.append(f"🌱 Good renewable utilization: {renewable_efficiency:.1f}%")
        else:
            insights.append(f"🌱 Improvement needed in renewable utilization: {renewable_efficiency:.1f}%")
        
        # SOC management insights
        soc_violations = len(df[(df['Battery_SOC_%'] < 20) | (df['Battery_SOC_%'] > 90)])
        if soc_violations == 0:
            insights.append("🔋 Perfect SOC management - no violations")
        else:
            insights.append(f"🔋 SOC violations detected: {soc_violations} hours")
        
        # Smart charging insights
        peak_hours = df[df['Current_Price_Category'] == 'High']
        peak_discharge = len(peak_hours[peak_hours['Battery_Action'] == 'Discharge'])
        if peak_discharge > len(peak_hours) * 0.7:
            insights.append("⚡ Smart peak shaving strategy active")
        
        return "\n".join(insights)


def main():
    """Ana detailed simulation fonksiyonu"""
    print("🔍 EMT RL Project - Detailed 1000-Hour Simulation")
    print("=" * 80)
    
    # Model path kontrolü
    model_paths = [
        "/content/models/PPO_0004_400k_20250611_1212/model.zip"
    ]
    
    selected_model = None
    for model_path in model_paths:
        if os.path.exists(model_path):
            selected_model = model_path
            break
    
    if not selected_model:
        print("❌ Eğitilmiş model bulunamadı!")
        return
    
    print(f"🤖 Model: {selected_model}")
    print("🎯 Simulation: 1000 saatlik detaylı karar analizi")
    
    # Detailed simulation oluştur
    simulator = DetailedDailySimulation(selected_model)
    
    # Setup
    if not simulator.setup_simulation():
        print("❌ Simulation setup başarısız!")
        return
    
    print("✅ Setup tamamlandı!")
    
    # 1000 saatlik simulation çalıştır
    print("\n🚀 1000 saatlik simulation başlıyor...")
    results_df = simulator.run_1000hour_simulation(deterministic=True)
    
    # Sonuçları kaydet
    print("\n💾 Sonuçlar kaydediliyor...")
    saved_files = simulator.save_results(results_df)
    
    # Quick insights
    print("\n💡 Quick Insights:")
    insights = simulator.generate_insights(results_df)
    print(insights)
    
    print(f"\n🎉 Simulation tamamlandı!")
    print("📁 Sonuçlar 'system_test/results/' dizininde:")
    for file_type, file_path in saved_files.items():
        print(f"   📄 {file_type}: {os.path.basename(file_path)}")
    
    # Print first few rows as preview
    print("\n📊 İlk 5 saatlik özet:")
    print("=" * 80)
    preview_cols = ['Hour', 'Load_kW', 'Solar_Generation_kW', 'Wind_Generation_kW', 
                   'Battery_SOC_%', 'Grid_Energy_kW', 'Battery_Power_kW', 'Total_Reward']
    print(results_df[preview_cols].head().to_string(index=False))


if __name__ == "__main__":
    main() 