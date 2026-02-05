"""
=============================================================================
🧠 نظام Machine Learning متقدم للتنبؤ
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import Counter
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

from config.settings import Config
from utils.logger import logger
from utils.performance import PerformanceBenchmark

class LotteryPredictor:
    """نظام تنبؤ متقدم باستخدام Machine Learning"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.is_trained = False
        self.benchmark = PerformanceBenchmark()
        
        # تهيئة النماذج
        self._initialize_models()
    
    def _initialize_models(self):
        """تهيئة نماذج ML مختلفة"""
        # 1. Random Forest (للتصنيف العام)
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # 2. Gradient Boosting (للتنبؤ الدقيق)
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # Scalers
        for model_name in self.models:
            self.scalers[model_name] = StandardScaler()
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """إعداد features متقدمة من البيانات"""
        operation_id = logger.start_operation('feature_preparation', {
            'total_draws': len(df),
            'models_count': len(self.models)
        })
        
        features_list = []
        labels_list = []
        
        try:
            for i in range(len(df) - 2):  # نحتاج سحبين للمستقبل
                current = df.iloc[i]['numbers']
                next_draw = df.iloc[i + 1]['numbers']
                future_draw = df.iloc[i + 2]['numbers']
                
                # Basic features
                basic_features = [
                    *current,  # الأرقام الحالية
                    sum(current),  # المجموع
                    sum(1 for n in current if n % 2),  # عدد الفردي
                    sum(1 for i in range(len(current)-1) if current[i+1] - current[i] == 1),  # المتتاليات
                    current[-1] - current[0],  # عرض النطاق
                    np.mean(current),  # المتوسط
                    np.std(current)  # الانحراف المعياري
                ]
                
                # Statistical features
                freq_counter = Counter(list(chain.from_iterable(df.iloc[:i+1]['numbers'])))
                statistical_features = [
                    np.mean([freq_counter.get(n, 0) for n in current]),  # متوسط التكرار
                    np.std([freq_counter.get(n, 0) for n in current]),  # انحراف التكرار
                    len(set(current) & set(next_draw)),  # تطابق مع السحب التالي
                ]
                
                # Pattern features
                pattern_features = [
                    len(set([n % 10 for n in current])),  # تنوع الآحاد
                    sum(1 for n in current if self._is_prime(n)),  # الأعداد الأولية
                    self._calculate_balance(current)  # درجة التوازن
                ]
                
                # Combine all features
                feature_vector = basic_features + statistical_features + pattern_features
                features_list.append(feature_vector)
                
                # Label: هل يظهر الرقم في السحب المستقبلي؟
                # نحن نتوقع احتمالية ظهور كل رقم
                for num in range(1, 33):
                    label = 1 if num in future_draw else 0
                    labels_list.append(label)
            
            features_array = np.array(features_list)
            labels_array = np.array(labels_list)
            
            logger.end_operation(operation_id, 'completed', {
                'features_shape': features_array.shape,
                'labels_shape': labels_array.shape,
                'feature_count': features_array.shape[1]
            })
            
            return features_array, labels_array
            
        except Exception as e:
            logger.end_operation(operation_id, 'failed', {'error': str(e)})
            raise
    
    def train(self, df: pd.DataFrame, model_name: str = 'random_forest'):
        """تدريب النموذج المحدد"""
        operation_id = logger.start_operation('model_training', {
            'model': model_name,
            'data_size': len(df)
        })
        
        try:
            self.benchmark.start_monitoring(f'train_{model_name}')
            
            # تحضير البيانات
            X, y = self.prepare_features(df)
            
            if X.shape[0] < 10:
                raise ValueError(f"بيانات غير كافية للتدريب: {X.shape[0]} عينة فقط")
            
            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # تطبيع البيانات
            X_train_scaled = self.scalers[model_name].fit_transform(X_train)
            X_test_scaled = self.scalers[model_name].transform(X_test)
            
            # التدريب
            model = self.models[model_name]
            model.fit(X_train_scaled, y_train)
            
            # التقييم
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            
            # حساب أهمية الميزات
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = model.feature_importances_
            
            # حفظ النموذج
            self._save_model(model_name)
            
            metrics = self.benchmark.stop_monitoring(f'train_{model_name}')
            
            logger.end_operation(operation_id, 'completed', {
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'cv_mean': round(cv_scores.mean(), 4),
                'cv_std': round(cv_scores.std(), 4),
                'training_samples': X_train.shape[0],
                'testing_samples': X_test.shape[0],
                **metrics
            })
            
            logger.log_prediction(
                model_name=model_name,
                accuracy=accuracy,
                confidence=precision,
                features_used=[f'feature_{i}' for i in range(X.shape[1])]
            )
            
            self.is_trained = True
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'cv_scores': cv_scores.tolist(),
                'feature_importance': self.feature_importance.get(model_name, []).tolist()
            }
            
        except Exception as e:
            logger.end_operation(operation_id, 'failed', {'error': str(e)})
            raise
    
    def predict(self, current_numbers: List[int], df: pd.DataFrame, 
                top_n: int = 10, model_name: str = 'random_forest') -> List[Tuple[int, float]]:
        """التنبؤ بالأرقام التالية"""
        if not self.is_trained or model_name not in self.models:
            raise ValueError(f"النموذج {model_name} غير مدرب")
        
        operation_id = logger.start_operation('prediction', {
            'model': model_name,
            'current_numbers': current_numbers,
            'top_n': top_n
        })
        
        try:
            self.benchmark.start_monitoring(f'predict_{model_name}')
            
            # تحضير features للرقم الحالي
            feature_vector = self._prepare_single_features(current_numbers, df)
            
            # تطبيع
            scaled_features = self.scalers[model_name].transform([feature_vector])
            
            # التنبؤ لكل رقم ممكن
            predictions = []
            model = self.models[model_name]
            
            for num in range(1, 33):
                if num in current_numbers:
                    continue
                
                # إنشاء نسخة من الميزات مع الرقم المستهدف
                test_features = scaled_features.copy()
                
                # التنبؤ
                prob = model.predict_proba(test_features)[0][1]  # احتمالية الظهور
                predictions.append((num, prob))
            
            # ترتيب حسب الاحتمالية
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_predictions = predictions[:top_n]
            
            metrics = self.benchmark.stop_monitoring(f'predict_{model_name}')
            
            logger.end_operation(operation_id, 'completed', {
                'top_predictions': top_predictions,
                'highest_probability': top_predictions[0][1] if top_predictions else 0,
                **metrics
            })
            
            return top_predictions
            
        except Exception as e:
            logger.end_operation(operation_id, 'failed', {'error': str(e)})
            raise
    
    def _prepare_single_features(self, numbers: List[int], df: pd.DataFrame) -> np.ndarray:
        """تحضير features لسحب واحد"""
        # Basic features
        basic_features = [
            *sorted(numbers),
            sum(numbers),
            sum(1 for n in numbers if n % 2),
            sum(1 for i in range(len(numbers)-1) if numbers[i+1] - numbers[i] == 1),
            numbers[-1] - numbers[0],
            np.mean(numbers),
            np.std(numbers)
        ]
        
        # Statistical features (من البيانات التاريخية)
        freq_counter = Counter(list(chain.from_iterable(df['numbers'])))
        statistical_features = [
            np.mean([freq_counter.get(n, 0) for n in numbers]),
            np.std([freq_counter.get(n, 0) for n in numbers]),
            0  # لا يوجد سحب تالي للمقارنة
        ]
        
        # Pattern features
        pattern_features = [
            len(set([n % 10 for n in numbers])),
            sum(1 for n in numbers if self._is_prime(n)),
            self._calculate_balance(numbers)
        ]
        
        return np.array(basic_features + statistical_features + pattern_features)
    
    def _is_prime(self, n: int) -> bool:
        """التحقق إذا كان الرقم أولياً"""
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    def _calculate_balance(self, numbers: List[int]) -> float:
        """حساب درجة التوازن"""
        if len(numbers) < 2:
            return 1.0
        
        # توزيع النصفين
        first_half = sum(1 for n in numbers if n <= 16)
        second_half = len(numbers) - first_half
        balance = 1 - abs(first_half - second_half) / len(numbers)
        
        return balance
    
    def _save_model(self, model_name: str):
        """حفظ النموذج للاستخدام المستقبلي"""
        import os
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        
        model_path = os.path.join(Config.MODELS_DIR, f'{model_name}.pkl')
        scaler_path = os.path.join(Config.MODELS_DIR, f'{model_name}_scaler.pkl')
        
        joblib.dump(self.models[model_name], model_path)
        joblib.dump(self.scalers[model_name], scaler_path)
    
    def load_model(self, model_name: str):
        """تحميل نموذج محفوظ"""
        import os
        
        model_path = os.path.join(Config.MODELS_DIR, f'{model_name}.pkl')
        scaler_path = os.path.join(Config.MODELS_DIR, f'{model_name}_scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.models[model_name] = joblib.load(model_path)
            self.scalers[model_name] = joblib.load(scaler_path)
            self.is_trained = True
            logger.logger.info(f"✅ تم تحميل النموذج {model_name}")
        else:
            raise FileNotFoundError(f"ملفات النموذج {model_name} غير موجودة")
    
    def ensemble_predict(self, current_numbers: List[int], df: pd.DataFrame, 
                        top_n: int = 10) -> List[Tuple[int, float]]:
        """تنبؤ باستخدام Ensemble من عدة نماذج"""
        all_predictions = []
        
        for model_name in self.models:
            try:
                predictions = self.predict(current_numbers, df, top_n=20, model_name=model_name)
                all_predictions.append(predictions)
            except Exception as e:
                logger.logger.warning(f"فشل التنبؤ بالنموذج {model_name}: {e}")
                continue
        
        if not all_predictions:
            return []
        
        # دمج التوقعات (متوسط الاحتمالات)
        combined_scores = Counter()
        
        for predictions in all_predictions:
            for num, prob in predictions:
                combined_scores[num] += prob
        
        # تحويل إلى متوسط
        for num in combined_scores:
            combined_scores[num] /= len(all_predictions)
        
        # ترتيب وترشيح
        final_predictions = [(num, score) for num, score in combined_scores.most_common(top_n)]
        
        return final_predictions

class RecommendationEngine:
    """نظام توصيات ذكي يعتمد على تعلم تفضيلات المستخدم"""
    
    def __init__(self):
        self.user_profiles = {}
        self.collaborative_matrix = None
        
    def learn_preferences(self, user_id: str, selected_tickets: List[List[int]], 
                         rejected_tickets: List[List[int]] = None):
        """تعلم تفضيلات المستخدم"""
        profile = {
            'selected_patterns': self._extract_patterns(selected_tickets),
            'preferred_numbers': self._get_common_numbers(selected_tickets),
            'avoided_numbers': self._get_common_numbers(rejected_tickets) if rejected_tickets else set(),
            'sum_preference': self._get_sum_preference(selected_tickets),
            'odd_even_preference': self._get_odd_even_preference(selected_tickets),
            'learning_strength': min(1.0, len(selected_tickets) / 10)  # قوة التعلم
        }
        
        self.user_profiles[user_id] = profile
        
        logger.logger.info(f"🎯 تعلم تفضيلات المستخدم {user_id}", extra={
            'selected_tickets': len(selected_tickets),
            'preferred_numbers_count': len(profile['preferred_numbers']),
            'learning_strength': profile['learning_strength']
        })
    
    def recommend(self, user_id: str, base_tickets: List[List[int]], 
                 count: int = 5) -> List[List[int]]:
        """توليد توصيات مخصصة"""
        if user_id not in self.user_profiles:
            return base_tickets[:count]
        
        profile = self.user_profiles[user_id]
        recommendations = []
        
        for base_ticket in base_tickets[:10]:  # استخدام أول 10 تذاكر كقاعدة
            customized = self._customize_ticket(base_ticket, profile)
            if customized and customized not in recommendations:
                recommendations.append(customized)
                if len(recommendations) >= count:
                    break
        
        return recommendations
    
    def _extract_patterns(self, tickets: List[List[int]]) -> Dict:
        """استخراج الأنماط من التذاكر"""
        if not tickets:
            return {}
        
        patterns = {
            'consecutive_range': [],
            'shadow_range': [],
            'sum_range': [],
            'odd_range': []
        }
        
        for ticket in tickets:
            patterns['consecutive_range'].append(
                sum(1 for i in range(len(ticket)-1) if ticket[i+1] - ticket[i] == 1)
            )
            patterns['shadow_range'].append(
                sum(1 for c in Counter([n % 10 for n in ticket]).values() if c > 1)
            )
            patterns['sum_range'].append(sum(ticket))
            patterns['odd_range'].append(sum(1 for n in ticket if n % 2))
        
        # حساب المتوسط والنطاق
        for key in patterns:
            if patterns[key]:
                patterns[key] = {
                    'min': min(patterns[key]),
                    'max': max(patterns[key]),
                    'avg': np.mean(patterns[key])
                }
            else:
                patterns[key] = {'min': 0, 'max': 0, 'avg': 0}
        
        return patterns
    
    def _get_common_numbers(self, tickets: List[List[int]]) -> Set[int]:
        """الحصول على الأرقام المشتركة"""
        if not tickets:
            return set()
        
        counter = Counter()
        for ticket in tickets:
            counter.update(ticket)
        
        # الأرقام التي تظهر في أكثر من 30% من التذاكر
        threshold = len(tickets) * 0.3
        return {num for num, count in counter.items() if count >= threshold}
    
    def _get_sum_preference(self, tickets: List[List[int]]) -> Dict:
        """تحديد تفضيل المجموع"""
        if not tickets:
            return {'min': 20, 'max': 200, 'avg': 100}
        
        sums = [sum(t) for t in tickets]
        return {
            'min': min(sums),
            'max': max(sums),
            'avg': np.mean(sums),
            'std': np.std(sums)
        }
    
    def _get_odd_even_preference(self, tickets: List[List[int]]) -> Dict:
        """تحديد تفضيل الفردي/الزوجي"""
        if not tickets:
            return {'min_odd': 0, 'max_odd': 6, 'avg_odd': 3}
        
        odd_counts = [sum(1 for n in t if n % 2) for t in tickets]
        return {
            'min_odd': min(odd_counts),
            'max_odd': max(odd_counts),
            'avg_odd': np.mean(odd_counts),
            'preferred_odd': int(np.round(np.mean(odd_counts)))
        }
    
    def _customize_ticket(self, base_ticket: List[int], profile: Dict) -> List[int]:
        """تخصيص التذكرة بناءً على التفضيلات"""
        # البدء بنسخة من التذكرة الأساسية
        ticket = base_ticket.copy()
        
        # تطبيق تفضيلات الأرقام
        preferred = profile['preferred_numbers']
        avoided = profile['avoided_numbers']
        
        # استبدال الأرقام المرفوضة بالمفضلة إن أمكن
        for i in range(len(ticket)):
            if ticket[i] in avoided and preferred:
                # اختيار رقم مفضل غير موجود في التذكرة
                for pref_num in preferred:
                    if pref_num not in ticket:
                        ticket[i] = pref_num
                        break
        
        # ضبط عدد الأرقام الفردية
        target_odd = profile['odd_even_preference']['preferred_odd']
        current_odd = sum(1 for n in ticket if n % 2)
        
        if current_odd > target_odd:
            # تحويل بعض الفردي إلى زوجي
            odd_indices = [i for i, n in enumerate(ticket) if n % 2]
            changes_needed = current_odd - target_odd
            
            for i in odd_indices[:changes_needed]:
                # تحويل إلى أقرب رقم زوجي
                ticket[i] = ticket[i] + 1 if ticket[i] < 32 else ticket[i] - 1
        
        elif current_odd < target_odd:
            # تحويل بعض الزوجي إلى فردي
            even_indices = [i for i, n in enumerate(ticket) if n % 2 == 0]
            changes_needed = target_odd - current_odd
            
            for i in even_indices[:changes_needed]:
                # تحويل إلى أقرب رقم فردي
                ticket[i] = ticket[i] + 1 if ticket[i] < 32 else ticket[i] - 1
        
        # ضبط المجموع
        sum_pref = profile['sum_preference']
        current_sum = sum(ticket)
        target_sum = int(sum_pref['avg'])
        
        if abs(current_sum - target_sum) > 10:
            # تعديل بعض الأرقام للاقتراب من المجموع المستهدف
            diff = target_sum - current_sum
            adjustment_per_num = diff // len(ticket)
            
            if abs(adjustment_per_num) > 0:
                for i in range(len(ticket)):
                    new_val = ticket[i] + adjustment_per_num
                    if 1 <= new_val <= 32:
                        ticket[i] = new_val
        
        return sorted(ticket)