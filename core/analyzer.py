"""
=============================================================================
📊 محرك التحليل المتقدم مع إصلاحات رياضية
=============================================================================
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import chain
from typing import List, Dict, Tuple, Optional, Set
from scipy.stats import poisson, norm
import warnings
warnings.filterwarnings('ignore')

from config.settings import Config
from utils.logger import logger

class AdvancedAnalyzer:
    """محرك تحليل متقدم مع إصلاحات رياضية"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.total_draws = len(df)
        self.all_nums = list(chain.from_iterable(df['numbers']))
        self.freq = Counter(self.all_nums)
        
        # إصلاح Poisson: حساب Lambda الصحيح
        self._initialize_poisson_correction()
        
        # تصنيف الأرقام
        self._classify_numbers()
        
        # آخر سحب
        self.last_draw = set(df.iloc[-1]['numbers']) if not df.empty else set()
        
        # بناء النماذج المتقدمة
        self._build_advanced_markov()
        self._analyze_poisson_precise()
        self._analyze_patterns()
        
        # تسجيل التحليل
        logger.logger.info("🔍 تهيئة محرك التحليل المتقدم", extra={
            'total_draws': self.total_draws,
            'unique_numbers': len(self.freq),
            'classification': {
                'hot_count': len(self.hot),
                'cold_count': len(self.cold),
                'neutral_count': len(self.neutral)
            }
        })
    
    def _initialize_poisson_correction(self):
        """إصلاح حساب Lambda لتحليل Poisson"""
        # ❌ الخطأ الأصلي: expected_per_draw = 6/32 = 0.1875 (نسبة)
        # ❌ ثم: lambda_param = 0.1875 * total_draws (خطأ)
        
        # ✅ الإصلاح: حساب صحيح للتوقع الرياضي
        # لكل سحب: احتمال ظهور رقم معين = 6/32 = 0.1875
        # التوقع الرياضي لعدد مرات ظهور الرقم في كل السحوبات:
        
        # الطريقة الصحيحة:
        # 1. احتمال ظهور الرقم في سحب واحد = 6/32
        # 2. عدد المحاولات = total_draws
        # 3. توقع عدد مرات الظهور = (6/32) * total_draws
        
        # لكن Poisson يحتاج إلى rate (متوسط الأحداث في فترة زمنية)
        # في حالتنا: rate = (6/32) * total_draws (لكل الرقم)
        
        self.poisson_rate = (Config.DEFAULT_TICKET_SIZE / Config.MAX_NUMBER) * self.total_draws
        
        logger.logger.info("📐 إصلاح حساب Poisson", extra={
            'old_calculation': 'poisson.pmf(k, k) [خطأ]',
            'new_calculation': f'poisson.pmf(k, {self.poisson_rate:.4f})',
            'poisson_rate': round(self.poisson_rate, 4),
            'ticket_size': Config.DEFAULT_TICKET_SIZE,
            'max_number': Config.MAX_NUMBER,
            'total_draws': self.total_draws
        })
    
    def _classify_numbers(self):
        """تصنيف الأرقام إلى ساخن/بارد/محايد"""
        # استخدام الانحراف المعياري للتصنيف
        frequencies = list(self.freq.values())
        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies)
        
        self.hot = set()
        self.cold = set()
        self.neutral = set()
        
        for num in range(1, 33):
            freq = self.freq.get(num, 0)
            z_score = (freq - mean_freq) / std_freq if std_freq > 0 else 0
            
            if z_score > 1.0:  # أكثر من انحراف معياري واحد
                self.hot.add(num)
            elif z_score < -1.0:  # أقل من انحراف معياري واحد
                self.cold.add(num)
            else:
                self.neutral.add(num)
    
    def _build_advanced_markov(self):
        """بناء سلسلة ماركوف من الدرجة الثانية مع تحسين الذاكرة"""
        # Markov Order 1 مع تحسين الذاكرة
        self.markov_1 = {}
        
        # Markov Order 2 مع تحسين الذاكرة
        self.markov_2 = {}
        
        # عدادات للاحتفاظ فقط بالانتقالات المتكررة
        markov_1_counts = defaultdict(Counter)
        markov_2_counts = defaultdict(Counter)
        
        for i in range(len(self.df) - 1):
            current = self.df.iloc[i]['numbers']
            next_draw = self.df.iloc[i + 1]['numbers']
            
            # Order 1
            for num in current:
                markov_1_counts[num].update(next_draw)
            
            # Order 2 (pairs)
            for j in range(len(current) - 1):
                pair = (current[j], current[j + 1])
                markov_2_counts[pair].update(next_draw)
        
        # تصفية الانتقالات النادرة
        for num, counter in markov_1_counts.items():
            total = sum(counter.values())
            if total >= Config.MARKOV_MIN_OCCURRENCES:
                self.markov_1[num] = {
                    num: count/total 
                    for num, count in counter.most_common(10)  # حفظ أفضل 10 فقط
                }
        
        for pair, counter in markov_2_counts.items():
            total = sum(counter.values())
            if total >= Config.MARKOV_MIN_OCCURRENCES:
                self.markov_2[pair] = {
                    num: count/total 
                    for num, count in counter.most_common(8)  # حفظ أفضل 8 فقط
                }
        
        logger.logger.info("🔗 بناء سلسلة ماركوف المتقدمة", extra={
            'markov_1_states': len(self.markov_1),
            'markov_2_states': len(self.markov_2),
            'min_occurrences': Config.MARKOV_MIN_OCCURRENCES
        })
    
    def _analyze_poisson_precise(self):
        """تحليل بواسون دقيق مع Z-Score صحيح"""
        self.poisson_data = []
        
        for num in range(Config.MIN_NUMBER, Config.MAX_NUMBER + 1):
            actual_count = self.freq.get(num, 0)
            
            # ✅ استخدام Lambda المصحح
            poisson_prob = poisson.pmf(actual_count, self.poisson_rate)
            
            # حساب Z-Score صحيح
            # للـ Poisson: μ = λ, σ = √λ
            std_dev = np.sqrt(self.poisson_rate)
            z_score = (actual_count - self.poisson_rate) / std_dev if std_dev > 0 else 0
            
            # حساب قيمة P (الدلالة الإحصائية)
            p_value = 2 * (1 - norm.cdf(abs(z_score))) if std_dev > 0 else 1
            
            # الشذوذ = مدى بعد الرقم عن المتوقع
            anomaly_score = abs(z_score)
            
            # حساب آخر ظهور
            appearances = [i for i, nums in enumerate(self.df['numbers']) if num in nums]
            last_seen = self.total_draws - 1 - appearances[-1] if appearances else self.total_draws
            
            # متوسط الفجوة
            if len(appearances) > 1:
                gaps = np.diff(appearances)
                avg_gap = np.mean(gaps)
                gap_std = np.std(gaps)
            else:
                avg_gap = self.total_draws
                gap_std = 0
            
            self.poisson_data.append({
                'number': num,
                'frequency': actual_count,
                'expected': round(self.poisson_rate, 2),
                'last_seen': last_seen,
                'avg_gap': round(avg_gap, 2),
                'gap_std': round(gap_std, 2),
                'z_score': round(z_score, 3),
                'p_value': round(p_value, 4),
                'anomaly_score': round(anomaly_score, 3),
                'is_significant': p_value < 0.05,
                'status': 'hot' if num in self.hot else 'cold' if num in self.cold else 'neutral',
                'classification': self._classify_anomaly(z_score, p_value)
            })
            
            # تسجيل الشذوذات الكبيرة
            if abs(z_score) > 2.5:
                logger.log_anomaly(num, z_score, self.poisson_rate, actual_count)
    
    def _classify_anomaly(self, z_score: float, p_value: float) -> str:
        """تصنيف الشذوذ الإحصائي"""
        if abs(z_score) > 3 and p_value < 0.01:
            return "extreme_anomaly"
        elif abs(z_score) > 2.5 and p_value < 0.05:
            return "significant_anomaly"
        elif abs(z_score) > 2:
            return "moderate_anomaly"
        elif abs(z_score) > 1.5:
            return "mild_anomaly"
        else:
            return "normal"
    
    def _analyze_patterns(self):
        """تحليل الأنماط المتقدمة"""
        self.patterns = {
            'consecutive_freq': defaultdict(int),
            'shadow_freq': defaultdict(int),
            'sum_distribution': [],
            'odd_even_ratio': [],
            'decade_distribution': defaultdict(int),  # توزيع العشرات
            'prime_distribution': defaultdict(int),   # توزيع الأعداد الأولية
            'cluster_analysis': self._analyze_clusters()
        }
        
        for nums in self.df['numbers']:
            # المتتاليات
            consec = sum(1 for i in range(len(nums) - 1) if nums[i + 1] - nums[i] == 1)
            self.patterns['consecutive_freq'][consec] += 1
            
            # الظلال (أرقام بنفس خانة الآحاد)
            shadows = sum(1 for c in Counter([n % 10 for n in nums]).values() if c > 1)
            self.patterns['shadow_freq'][shadows] += 1
            
            # المجموع
            self.patterns['sum_distribution'].append(sum(nums))
            
            # نسبة الفردي/الزوجي
            odd_count = sum(1 for n in nums if n % 2)
            self.patterns['odd_even_ratio'].append(odd_count)
            
            # توزيع العشرات
            for num in nums:
                decade = (num - 1) // 10  # 0-2: 1-10, 11-20, 21-30
                self.patterns['decade_distribution'][decade] += 1
            
            # الأعداد الأولية
            prime_count = sum(1 for n in nums if self._is_prime(n))
            self.patterns['prime_distribution'][prime_count] += 1
    
    def _analyze_clusters(self) -> Dict:
        """تحليل العناقيد والتجمعات"""
        clusters = []
        for nums in self.df['numbers']:
            # البحث عن تجمعات (مجموعات أرقام متقاربة)
            sorted_nums = sorted(nums)
            current_cluster = [sorted_nums[0]]
            
            for i in range(1, len(sorted_nums)):
                if sorted_nums[i] - sorted_nums[i-1] <= 3:  # فرق 3 أو أقل
                    current_cluster.append(sorted_nums[i])
                else:
                    if len(current_cluster) >= 3:
                        clusters.append(current_cluster)
                    current_cluster = [sorted_nums[i]]
            
            if len(current_cluster) >= 3:
                clusters.append(current_cluster)
        
        # تحليل العناقيد
        cluster_stats = {
            'total_clusters': len(clusters),
            'avg_cluster_size': np.mean([len(c) for c in clusters]) if clusters else 0,
            'max_cluster_size': max([len(c) for c in clusters]) if clusters else 0,
            'common_clusters': Counter([tuple(c) for c in clusters]).most_common(5)
        }
        
        return cluster_stats
    
    def _is_prime(self, n: int) -> bool:
        """التحقق إذا كان الرقم أولياً"""
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    def get_markov_prediction(self, last_numbers: List[int], top_n: int = 8) -> List[Tuple[int, float]]:
        """توقع الأرقام بناءً على Markov مع أوزان ذكية"""
        candidates = Counter()
        
        # استخدام Order 1 مع أوزان
        for num in last_numbers:
            if num in self.markov_1:
                for next_num, prob in self.markov_1[num].items():
                    candidates[next_num] += prob * 1.0  # وزن Order 1
        
        # استخدام Order 2 (أوزان أعلى)
        for i in range(len(last_numbers) - 1):
            pair = (last_numbers[i], last_numbers[i + 1])
            if pair in self.markov_2:
                for next_num, prob in self.markov_2[pair].items():
                    candidates[next_num] += prob * 2.0  # وزن مضاعف لـ Order 2
        
        # إضافة عامل الشعبية (التكرار التاريخي)
        for num, freq in self.freq.most_common(20):
            if num not in last_numbers:
                popularity_factor = freq / max(self.freq.values())
                candidates[num] += popularity_factor * 0.5  # وزن متوسط
        
        # تحويل إلى احتمالات
        total = sum(candidates.values())
        if total == 0:
            return []
        
        probabilities = [(num, count / total) for num, count in candidates.most_common(top_n)]
        
        # تسجيل التوقعات
        logger.log_prediction('markov_chain', 0.0, probabilities[0][1] if probabilities else 0, 
                            ['markov_1', 'markov_2', 'popularity'])
        
        return probabilities
    
    def get_ticket_analysis(self, ticket: List[int]) -> Dict:
        """تحليل شامل لتذكرة معينة"""
        nums = sorted(ticket)
        
        analysis = {
            'basic': {
                'sum': sum(nums),
                'odd': sum(1 for n in nums if n % 2),
                'even': sum(1 for n in nums if n % 2 == 0),
                'consecutive': sum(1 for i in range(len(nums) - 1) if nums[i + 1] - nums[i] == 1),
                'shadows': sum(1 for c in Counter([n % 10 for n in nums]).values() if c > 1),
                'range_width': nums[-1] - nums[0],
                'avg_spacing': np.mean([nums[i+1] - nums[i] for i in range(len(nums)-1)]) if len(nums) > 1 else 0
            },
            'classification': {
                'hot_count': len(set(nums) & self.hot),
                'cold_count': len(set(nums) & self.cold),
                'neutral_count': len(set(nums) & self.neutral),
                'last_match': len(set(nums) & self.last_draw)
            },
            'statistical': {
                'avg_frequency': round(np.mean([self.freq[n] for n in nums]), 2),
                'freq_std': round(np.std([self.freq[n] for n in nums]), 2),
                'diversity_score': round(len(set([self.freq[n] for n in nums])) / len(nums), 3),
                'balance_score': self._calculate_balance_score(nums)
            },
            'advanced': {
                'prime_count': sum(1 for n in nums if self._is_prime(n)),
                'decade_distribution': self._get_decade_distribution(nums),
                'cluster_score': self._calculate_cluster_score(nums),
                'pattern_complexity': self._calculate_pattern_complexity(nums)
            }
        }
        
        # حساب درجة الجودة الشاملة
        analysis['quality_score'] = self._calculate_quality_score(analysis)
        
        return analysis
    
    def _calculate_balance_score(self, nums: List[int]) -> float:
        """حساب درجة التوازن في التذكرة"""
        if len(nums) < 2:
            return 1.0
        
        # توزيع العشرات
        decades = [(n - 1) // 10 for n in nums]
        decade_balance = len(set(decades)) / 3  # 3 عقود
        
        # توزيع النصفين (1-16, 17-32)
        first_half = sum(1 for n in nums if n <= 16)
        second_half = len(nums) - first_half
        half_balance = 1 - abs(first_half - second_half) / len(nums)
        
        # توزيع الأرباع
        quarters = [0, 0, 0, 0]
        for n in nums:
            quarter = (n - 1) // 8
            quarters[quarter] += 1
        quarter_balance = len([q for q in quarters if q > 0]) / 4
        
        return round((decade_balance + half_balance + quarter_balance) / 3, 3)
    
    def _get_decade_distribution(self, nums: List[int]) -> Dict:
        """توزيع الأرقام على العقود"""
        distribution = {0: 0, 1: 0, 2: 0, 3: 0}  # 1-10, 11-20, 21-30, 31-32
        for n in nums:
            decade = (n - 1) // 10
            distribution[min(decade, 3)] += 1
        return distribution
    
    def _calculate_cluster_score(self, nums: List[int]) -> float:
        """حساب درجة التجمع"""
        if len(nums) < 3:
            return 0.0
        
        sorted_nums = sorted(nums)
        clusters = []
        current_cluster = [sorted_nums[0]]
        
        for i in range(1, len(sorted_nums)):
            if sorted_nums[i] - sorted_nums[i-1] <= 3:
                current_cluster.append(sorted_nums[i])
            else:
                if len(current_cluster) >= 2:
                    clusters.append(current_cluster)
                current_cluster = [sorted_nums[i]]
        
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)
        
        if not clusters:
            return 0.0
        
        # درجة تعتمد على عدد وحجم العناقيد
        total_clustered = sum(len(c) for c in clusters)
        cluster_score = total_clustered / len(nums)
        
        # عقوبة للعناقيد الكبيرة جداً
        max_cluster_size = max(len(c) for c in clusters) if clusters else 0
        if max_cluster_size > 4:
            cluster_score *= 0.7
        
        return round(cluster_score, 3)
    
    def _calculate_pattern_complexity(self, nums: List[int]) -> float:
        """حساب تعقيد النمط"""
        features = [
            len(set(nums)),  # التفرد
            len(set([n % 10 for n in nums])),  # تنوع الآحاد
            sum(1 for i in range(len(nums)-1) if nums[i+1] - nums[i] == 1),  # المتتاليات
            len([n for n in nums if self._is_prime(n)]),  # الأعداد الأولية
            nums[-1] - nums[0]  # عرض النطاق
        ]
        
        # تطبيع الميزات
        normalized = [f / len(nums) for f in features]
        
        # حساب الإنتروبيا (مقياس التعقيد)
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in normalized)
        
        return round(entropy / np.log2(len(features)), 3)
    
    def _calculate_quality_score(self, analysis: Dict) -> float:
        """حساب درجة الجودة الشاملة للتذكرة"""
        weights = {
            'balance': 0.25,
            'diversity': 0.20,
            'pattern': 0.15,
            'statistical': 0.20,
            'historical': 0.20
        }
        
        scores = {
            'balance': analysis['statistical']['balance_score'],
            'diversity': analysis['statistical']['diversity_score'],
            'pattern': analysis['advanced']['pattern_complexity'],
            'statistical': min(1.0, analysis['statistical']['avg_frequency'] / 10),  # تطبيع
            'historical': analysis['classification']['hot_count'] / len(analysis['basic'])
        }
        
        quality_score = sum(scores[key] * weights[key] for key in weights)
        return round(quality_score * 10, 2)  # تحويل لمقياس 0-10