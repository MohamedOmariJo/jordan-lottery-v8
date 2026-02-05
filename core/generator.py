"""
=============================================================================
🎰 مولد التذاكر الذكي المحسّن مع تحسينات الأداء
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import Config
from utils.logger import logger
from utils.performance import PerformanceBenchmark
from core.analyzer import AdvancedAnalyzer

class SmartGenerator:
    """مولد تذاكر ذكي مع فلاتر متقدمة وتحسينات أداء"""
    
    def __init__(self, analyzer: AdvancedAnalyzer):
        self.analyzer = analyzer
        self.benchmark = PerformanceBenchmark()
        self.cache = {}
        
    def generate_tickets(
        self,
        count: int,
        size: int = 6,
        constraints: Optional[Dict] = None,
        use_cache: bool = True
    ) -> List[List[int]]:
        """توليد تذاكر مع فلاتر محسنة واستخدام cache"""
        
        if constraints is None:
            constraints = {}
        
        # التحقق من Cache أولاً
        cache_key = self._generate_cache_key(count, size, constraints)
        if use_cache and cache_key in self.cache:
            logger.logger.info(f"🎯 استخدام Cache للتوليد - مفتاح: {cache_key[:50]}...")
            return self.cache[cache_key].copy()
        
        op_id = logger.start_operation('ticket_generation', {
            'count': count,
            'size': size,
            'constraints': constraints
        })
        
        try:
            with self.benchmark.monitor_operation('generation'):
                # إعداد Pool الأرقام
                pool = self._prepare_number_pool(constraints)
                
                if len(pool) < size:
                    error_msg = f"❌ عدد الأرقام المتاحة ({len(pool)}) أقل من حجم التذكرة ({size})"
                    logger.logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # التوليد باستخدام الطريقة المثلى
                if count <= 10:
                    tickets = self._generate_small_batch(pool, size, count, constraints)
                elif count <= 100:
                    tickets = self._generate_medium_batch(pool, size, count, constraints)
                else:
                    tickets = self._generate_large_batch(pool, size, count, constraints)
                
                # تطبيق الفلاتر المتقدمة
                if constraints:
                    tickets = self._apply_advanced_filters(tickets, constraints)
                
                # الحد من العدد المطلوب
                tickets = tickets[:count]
                
                # تسجيل في Cache
                if use_cache and len(tickets) > 0:
                    self.cache[cache_key] = tickets.copy()
                    # تنظيف Cache القديم
                    self._clean_cache()
                
                logger.end_operation(op_id, 'completed', {
                    'generated_count': len(tickets),
                    'success_rate': round(len(tickets) / count * 100, 2),
                    'cache_used': use_cache,
                    'cache_key': cache_key[:30]
                })
                
                return tickets
                
        except Exception as e:
            logger.end_operation(op_id, 'failed', {'error': str(e)})
            raise
    
    def _prepare_number_pool(self, constraints: Dict) -> List[int]:
        """تحضير مجموعة الأرقام مع تطبيق الاستبعاد"""
        pool = list(range(Config.MIN_NUMBER, Config.MAX_NUMBER + 1))
        
        if 'exclude' in constraints:
            exclude_set = set(constraints['exclude'])
            pool = [n for n in pool if n not in exclude_set]
        
        # إضافة تحسين: إزالة الأرقام ذات التكرار المنخفض جداً
        if constraints.get('filter_low_freq', False):
            avg_freq = np.mean(list(self.analyzer.freq.values()))
            pool = [n for n in pool if self.analyzer.freq.get(n, 0) >= avg_freq * 0.5]
        
        return pool
    
    def _generate_small_batch(self, pool: List[int], size: int, 
                            count: int, constraints: Dict) -> List[List[int]]:
        """توليد دفعات صغيرة (<= 10)"""
        tickets = []
        attempts = 0
        max_attempts = count * 100
        
        while len(tickets) < count and attempts < max_attempts:
            attempts += 1
            
            # توليد تذكرة
            ticket = sorted(random.sample(pool, size))
            
            # التحقق من القيود الأساسية
            if not self._satisfies_basic_constraints(ticket, constraints):
                continue
            
            # التحقق من القيود المتقدمة
            if not self._satisfies_advanced_constraints(ticket, constraints):
                continue
            
            # تجنب التكرار
            ticket_tuple = tuple(ticket)
            if ticket_tuple not in [tuple(t) for t in tickets]:
                tickets.append(ticket)
        
        return tickets
    
    def _generate_medium_batch(self, pool: List[int], size: int, 
                             count: int, constraints: Dict) -> List[List[int]]:
        """توليد دفعات متوسطة (<= 100) باستخدام تحسينات"""
        tickets_set = set()
        batch_size = min(1000, count * 10)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for _ in range(min(10, count)):
                future = executor.submit(
                    self._generate_batch_parallel,
                    pool, size, batch_size, constraints
                )
                futures.append(future)
            
            for future in as_completed(futures):
                batch_tickets = future.result()
                for ticket in batch_tickets:
                    if len(tickets_set) >= count:
                        break
                    tickets_set.add(tuple(ticket))
        
        return [list(t) for t in list(tickets_set)[:count]]
    
    def _generate_large_batch(self, pool: List[int], size: int, 
                            count: int, constraints: Dict) -> List[List[int]]:
        """توليد دفعات كبيرة (> 100) باستخدام vectorization"""
        tickets_set = set()
        total_generated = 0
        
        while len(tickets_set) < count and total_generated < count * 100:
            # توليد دفعة كبيرة
            batch_size = min(10000, (count - len(tickets_set)) * 10)
            
            # استخدام numpy للسرعة
            batch = np.array([
                np.random.choice(pool, size=size, replace=False)
                for _ in range(batch_size)
            ])
            
            # تطبيق الفلاتر الأساسية بسرعة
            batch = self._filter_batch_vectorized(batch, constraints)
            
            # إضافة للنتائج
            for ticket in batch:
                ticket_tuple = tuple(sorted(ticket))
                tickets_set.add(ticket_tuple)
                if len(tickets_set) >= count:
                    break
            
            total_generated += batch_size
        
        return [list(t) for t in list(tickets_set)[:count]]
    
    def _generate_batch_parallel(self, pool: List[int], size: int, 
                               batch_size: int, constraints: Dict) -> List[List[int]]:
        """توليد دفعة بالتوازي"""
        batch_tickets = []
        
        for _ in range(batch_size):
            ticket = sorted(random.sample(pool, size))
            
            if (self._satisfies_basic_constraints(ticket, constraints) and 
                self._satisfies_advanced_constraints(ticket, constraints)):
                batch_tickets.append(ticket)
        
        return batch_tickets
    
    def _filter_batch_vectorized(self, batch: np.ndarray, constraints: Dict) -> np.ndarray:
        """تصفية الدفعة باستخدام vectorization"""
        if batch.size == 0:
            return batch
        
        masks = []
        
        # فلتر المجموع
        if 'sum_range' in constraints:
            min_sum, max_sum = constraints['sum_range']
            sums = batch.sum(axis=1)
            masks.append((sums >= min_sum) & (sums <= max_sum))
        
        # فلتر الفردي
        if 'odd' in constraints:
            target_odd = constraints['odd']
            odd_counts = np.sum(batch % 2, axis=1)
            masks.append(odd_counts == target_odd)
        
        # فلتر المتتاليات
        if 'consecutive' in constraints:
            target_consec = constraints['consecutive']
            consec_counts = np.array([
                np.sum(np.diff(row) == 1)
                for row in batch
            ])
            masks.append(consec_counts == target_consec)
        
        # تطبيق جميع الأقنعة
        if masks:
            combined_mask = np.all(masks, axis=0)
            batch = batch[combined_mask]
        
        return batch
    
    def _satisfies_basic_constraints(self, ticket: List[int], constraints: Dict) -> bool:
        """التحقق من القيود الأساسية"""
        # الفردي
        if 'odd' in constraints:
            odd_count = sum(1 for n in ticket if n % 2)
            if odd_count != constraints['odd']:
                return False
        
        # المجموع
        if 'sum_range' in constraints:
            ticket_sum = sum(ticket)
            min_sum, max_sum = constraints['sum_range']
            if not (min_sum <= ticket_sum <= max_sum):
                return False
        
        # الأرقام الثابتة
        if 'fixed' in constraints:
            fixed_set = set(constraints['fixed'])
            if not fixed_set.issubset(set(ticket)):
                return False
        
        return True
    
    def _satisfies_advanced_constraints(self, ticket: List[int], constraints: Dict) -> bool:
        """التحقق من القيود المتقدمة"""
        # المتتاليات
        if 'consecutive' in constraints:
            consec_count = sum(1 for i in range(len(ticket)-1) 
                             if ticket[i+1] - ticket[i] == 1)
            if consec_count != constraints['consecutive']:
                return False
        
        # الظلال
        if 'shadows' in constraints:
            shadows_count = sum(1 for c in Counter([n % 10 for n in ticket]).values() 
                              if c > 1)
            if shadows_count != constraints['shadows']:
                return False
        
        # Hot/Cold
        if 'hot_min' in constraints:
            hot_count = len(set(ticket) & self.analyzer.hot)
            if hot_count < constraints['hot_min']:
                return False
        
        if 'cold_max' in constraints:
            cold_count = len(set(ticket) & self.analyzer.cold)
            if cold_count > constraints['cold_max']:
                return False
        
        # التطابق مع آخر سحب
        if 'last_match' in constraints:
            match_count = len(set(ticket) & self.analyzer.last_draw)
            if match_count != constraints['last_match']:
                return False
        
        return True
    
    def _apply_advanced_filters(self, tickets: List[List[int]], constraints: Dict) -> List[List[int]]:
        """تطبيق فلاتر متقدمة بعد التوليد"""
        filtered_tickets = []
        
        for ticket in tickets:
            if self._satisfies_advanced_constraints(ticket, constraints):
                filtered_tickets.append(ticket)
        
        return filtered_tickets
    
    def generate_markov_based(self, count: int, size: int = 6) -> List[List[int]]:
        """توليد تذاكر بناءً على Markov"""
        op_id = logger.start_operation('markov_generation', {
            'count': count,
            'size': size
        })
        
        try:
            with self.benchmark.monitor_operation('markov_generation'):
                tickets = []
                last_nums = sorted(list(self.analyzer.last_draw))
                
                for _ in range(count):
                    # الحصول على التوقعات
                    predictions = self.analyzer.get_markov_prediction(last_nums, top_n=15)
                    
                    if not predictions:
                        # fallback إلى عشوائي
                        ticket = sorted(random.sample(range(1, 33), size))
                    else:
                        # اختيار من التوقعات بأوزان
                        candidates, weights = zip(*predictions)
                        
                        # تكملة إذا لم يكن كافي
                        while len(candidates) < size * 2:
                            remaining = list(set(range(1, 33)) - set(candidates))
                            candidates = list(candidates) + random.sample(remaining, 
                                                                         min(size * 2 - len(candidates), 
                                                                             len(remaining)))
                        
                        # تحضير الأوزان
                        weights = np.array(weights[:len(candidates)])
                        weights = weights / weights.sum()
                        
                        # اختيار بأوزان
                        selected = np.random.choice(
                            candidates,
                            size=size,
                            replace=False,
                            p=weights
                        )
                        ticket = sorted(selected.tolist())
                    
                    if ticket not in tickets:
                        tickets.append(ticket)
                
                logger.end_operation(op_id, 'completed', {
                    'generated_count': len(tickets),
                    'markov_used': len(predictions) > 0 if 'predictions' in locals() else False
                })
                
                return tickets
                
        except Exception as e:
            logger.end_operation(op_id, 'failed', {'error': str(e)})
            raise
    
    def _generate_cache_key(self, count: int, size: int, constraints: Dict) -> str:
        """توليد مفتاح Cache فريد"""
        import hashlib
        import json
        
        data = {
            'count': count,
            'size': size,
            'constraints': constraints,
            'analyzer_hash': hash(str(self.analyzer.freq))
        }
        
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _clean_cache(self):
        """تنظيف Cache القديم"""
        max_cache_size = 100  # أقصى عدد من العناصر في Cache
        
        if len(self.cache) > max_cache_size:
            # إزالة العناصر الأقدم (بسيط - في الإنتاج استخدم LRU)
            keys_to_remove = list(self.cache.keys())[:len(self.cache) - max_cache_size]
            for key in keys_to_remove:
                del self.cache[key]
    
    def generate_with_ml(self, count: int, size: int = 6, 
                        model_name: str = 'random_forest') -> List[List[int]]:
        """توليد تذاكر باستخدام تنبؤات ML"""
        op_id = logger.start_operation('ml_generation', {
            'count': count,
            'size': size,
            'model': model_name
        })
        
        try:
            with self.benchmark.monitor_operation('ml_generation'):
                # هذا يحتاج إلى integration مع predictor
                # هنا مثال بسيط
                tickets = []
                
                for _ in range(count):
                    # استخدام تنبؤات ML كمرجع
                    ticket = self._generate_ml_inspired_ticket(size)
                    if ticket not in tickets:
                        tickets.append(ticket)
                
                logger.end_operation(op_id, 'completed', {
                    'generated_count': len(tickets),
                    'model_used': model_name
                })
                
                return tickets
                
        except Exception as e:
            logger.end_operation(op_id, 'failed', {'error': str(e)})
            raise
    
    def _generate_ml_inspired_ticket(self, size: int) -> List[int]:
        """توليد تذكرة مستوحاة من تنبؤات ML"""
        # هذا مثال بسيط - في التطبيق الحقيقي سيستخدم نموذج ML فعلي
        pool = list(range(1, 33))
        
        # إعطاء وزن أعلى للأرقام الساخنة
        weights = np.ones(32)
        for num in self.analyzer.hot:
            weights[num-1] = 2.0
        for num in self.analyzer.cold:
            weights[num-1] = 0.5
        
        weights = weights / weights.sum()
        
        ticket = np.random.choice(
            pool,
            size=size,
            replace=False,
            p=weights
        )
        
        return sorted(ticket.tolist())
    
    def get_generation_stats(self) -> Dict:
        """الحصول على إحصائيات التوليد"""
        return {
            'cache_size': len(self.cache),
            'performance_stats': self.benchmark.get_performance_report('generation'),
            'generator_info': {
                'class': self.__class__.__name__,
                'analyzer_initialized': self.analyzer is not None,
                'methods_available': [
                    'generate_tickets',
                    'generate_markov_based',
                    'generate_with_ml'
                ]
            }
        }