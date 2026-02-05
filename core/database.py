"""
=============================================================================
💾 نظام إدارة قاعدة البيانات المتقدم
=============================================================================
"""

import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import hashlib

from config.settings import Config
from utils.logger import logger

# تعريف قاعدة النماذج
Base = declarative_base()

class DrawRecord(Base):
    """سجل السحب في قاعدة البيانات"""
    __tablename__ = 'draws'
    
    id = Column(Integer, primary_key=True)
    draw_id = Column(Integer, unique=True, nullable=False, index=True)
    numbers = Column(JSON, nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    sum_total = Column(Integer)
    odd_count = Column(Integer)
    even_count = Column(Integer)
    consecutive_pairs = Column(Integer)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    metadata = Column(JSON)  # لحفظ تحليلات إضافية
    
    def __repr__(self):
        return f"<DrawRecord(draw_id={self.draw_id}, date={self.date})>"

class TicketRecord(Base):
    """سجل التذاكر المحفوظة"""
    __tablename__ = 'tickets'
    
    id = Column(Integer, primary_key=True)
    ticket_hash = Column(String(64), unique=True, nullable=False, index=True)
    numbers = Column(JSON, nullable=False)
    generated_at = Column(DateTime, default=datetime.now)
    strategy = Column(String(50))  # نمط التوليد
    constraints = Column(JSON)  # القيود المستخدمة
    analysis = Column(JSON)  # تحليل التذكرة
    user_id = Column(String(100), index=True)
    tags = Column(JSON)  # وسوم للتذاكر
    
    def __repr__(self):
        return f"<TicketRecord(hash={self.ticket_hash[:8]}...)>"

class PredictionRecord(Base):
    """سجل تنبؤات ML"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False, index=True)
    input_data = Column(JSON, nullable=False)
    predictions = Column(JSON, nullable=False)
    accuracy = Column(Float)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.now, index=True)
    
    def __repr__(self):
        return f"<PredictionRecord(model={self.model_name})>"

class UserPreference(Base):
    """تفضيلات المستخدم"""
    __tablename__ = 'user_preferences'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), unique=True, nullable=False, index=True)
    preferred_numbers = Column(JSON)
    avoided_numbers = Column(JSON)
    sum_preferences = Column(JSON)
    odd_even_preferences = Column(JSON)
    pattern_preferences = Column(JSON)
    learning_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    def __repr__(self):
        return f"<UserPreference(user_id={self.user_id})>"

class DatabaseManager:
    """مدير قاعدة البيانات المتقدم"""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or Config.DATABASE_URL
        self.engine = None
        self.SessionLocal = None
        self._setup_database()
    
    def _setup_database(self):
        """إعداد قاعدة البيانات وتهيئة الجداول"""
        op_id = logger.start_operation('database_setup', {
            'database_url': self.db_url
        })
        
        try:
            # إنشاء محرك قاعدة البيانات
            self.engine = create_engine(
                self.db_url,
                echo=False,  # تفعيل في وضع التطوير فقط
                pool_size=Config.get_database_config()['pool_size'],
                max_overflow=Config.get_database_config()['max_overflow'],
                pool_timeout=Config.get_database_config()['pool_timeout'],
                pool_recycle=Config.get_database_config()['pool_recycle']
            )
            
            # إنشاء الجداول
            Base.metadata.create_all(bind=self.engine)
            
            # إنشاء Session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.end_operation(op_id, 'completed', {
                'tables_created': list(Base.metadata.tables.keys()),
                'database_ready': True
            })
            
        except Exception as e:
            logger.end_operation(op_id, 'failed', {'error': str(e)})
            raise
    
    def get_session(self) -> Session:
        """الحصول على جلسة قاعدة بيانات"""
        return self.SessionLocal()
    
    def add_draw_with_analysis(self, numbers: List[int], draw_date: datetime, 
                              draw_id: Optional[int] = None, metadata: Dict = None):
        """إضافة سحب مع تحليله تلقائياً"""
        op_id = logger.start_operation('add_draw', {
            'draw_id': draw_id,
            'numbers_count': len(numbers)
        })
        
        session = self.get_session()
        
        try:
            # التحقق من عدم التكرار
            if draw_id:
                existing = session.query(DrawRecord).filter_by(draw_id=draw_id).first()
                if existing:
                    logger.logger.warning(f"⚠️ السحب #{draw_id} موجود مسبقاً")
                    session.close()
                    logger.end_operation(op_id, 'skipped', {'reason': 'already_exists'})
                    return
            
            # إنشاء السجل
            draw_record = DrawRecord(
                draw_id=draw_id or self._get_next_draw_id(session),
                numbers=numbers,
                date=draw_date,
                sum_total=sum(numbers),
                odd_count=sum(1 for n in numbers if n % 2),
                even_count=sum(1 for n in numbers if n % 2 == 0),
                consecutive_pairs=self._count_consecutive_pairs(numbers),
                metadata=metadata or {}
            )
            
            session.add(draw_record)
            session.commit()
            
            logger.end_operation(op_id, 'completed', {
                'draw_id': draw_record.draw_id,
                'record_added': True
            })
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.end_operation(op_id, 'failed', {'error': str(e)})
            raise
        finally:
            session.close()
    
    def add_ticket(self, numbers: List[int], strategy: str = None, 
                  constraints: Dict = None, analysis: Dict = None, 
                  user_id: str = None, tags: List[str] = None):
        """إضافة تذكرة إلى قاعدة البيانات"""
        op_id = logger.start_operation('add_ticket', {
            'strategy': strategy,
            'user_id': user_id
        })
        
        session = self.get_session()
        
        try:
            # إنشاء hash فريد للتذكرة
            ticket_hash = self._create_ticket_hash(numbers)
            
            # التحقق من عدم التكرار
            existing = session.query(TicketRecord).filter_by(ticket_hash=ticket_hash).first()
            if existing:
                logger.logger.info(f"🎫 التذكرة موجودة مسبقاً (hash: {ticket_hash[:8]}...)")
                session.close()
                logger.end_operation(op_id, 'skipped', {'reason': 'already_exists'})
                return existing
            
            # إنشاء السجل
            ticket_record = TicketRecord(
                ticket_hash=ticket_hash,
                numbers=sorted(numbers),
                generated_at=datetime.now(),
                strategy=strategy,
                constraints=constraints or {},
                analysis=analysis or {},
                user_id=user_id,
                tags=tags or []
            )
            
            session.add(ticket_record)
            session.commit()
            
            logger.end_operation(op_id, 'completed', {
                'ticket_hash': ticket_hash,
                'record_added': True
            })
            
            return ticket_record
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.end_operation(op_id, 'failed', {'error': str(e)})
            raise
        finally:
            session.close()
    
    def add_prediction(self, model_name: str, input_data: Dict, 
                      predictions: List, accuracy: float = None, 
                      confidence: float = None):
        """إضافة تنبؤ ML إلى قاعدة البيانات"""
        op_id = logger.start_operation('add_prediction', {
            'model': model_name,
            'predictions_count': len(predictions)
        })
        
        session = self.get_session()
        
        try:
            prediction_record = PredictionRecord(
                model_name=model_name,
                input_data=input_data,
                predictions=predictions,
                accuracy=accuracy,
                confidence=confidence,
                created_at=datetime.now()
            )
            
            session.add(prediction_record)
            session.commit()
            
            logger.end_operation(op_id, 'completed', {
                'prediction_id': prediction_record.id,
                'record_added': True
            })
            
            return prediction_record
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.end_operation(op_id, 'failed', {'error': str(e)})
            raise
        finally:
            session.close()
    
    def update_user_preferences(self, user_id: str, preferences: Dict):
        """تحديث تفضيلات المستخدم"""
        op_id = logger.start_operation('update_user_preferences', {
            'user_id': user_id
        })
        
        session = self.get_session()
        
        try:
            # البحث عن المستخدم
            user_pref = session.query(UserPreference).filter_by(user_id=user_id).first()
            
            if user_pref:
                # تحديث التفضيلات
                for key, value in preferences.items():
                    if hasattr(user_pref, key):
                        setattr(user_pref, key, value)
                user_pref.updated_at = datetime.now()
            else:
                # إنشاء جديد
                user_pref = UserPreference(
                    user_id=user_id,
                    **preferences,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                session.add(user_pref)
            
            session.commit()
            
            logger.end_operation(op_id, 'completed', {
                'user_id': user_id,
                'preferences_updated': True
            })
            
            return user_pref
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.end_operation(op_id, 'failed', {'error': str(e)})
            raise
        finally:
            session.close()
    
    def get_user_tickets(self, user_id: str, limit: int = 100) -> List[TicketRecord]:
        """الحصول على تذاكر المستخدم"""
        session = self.get_session()
        
        try:
            tickets = session.query(TicketRecord)\
                .filter_by(user_id=user_id)\
                .order_by(TicketRecord.generated_at.desc())\
                .limit(limit)\
                .all()
            
            return tickets
            
        finally:
            session.close()
    
    def get_draws_by_date_range(self, start_date: datetime, 
                               end_date: datetime) -> List[DrawRecord]:
        """الحصول على السحوبات ضمن نطاق تاريخي"""
        session = self.get_session()
        
        try:
            draws = session.query(DrawRecord)\
                .filter(DrawRecord.date >= start_date)\
                .filter(DrawRecord.date <= end_date)\
                .order_by(DrawRecord.date)\
                .all()
            
            return draws
            
        finally:
            session.close()
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات التحليل"""
        session = self.get_session()
        
        try:
            stats = {
                'total_draws': session.query(DrawRecord).count(),
                'total_tickets': session.query(TicketRecord).count(),
                'total_predictions': session.query(PredictionRecord).count(),
                'total_users': session.query(UserPreference).count(),
                'recent_activity': {}
            }
            
            # النشاط الحديث
            today = datetime.now().date()
            week_ago = datetime.now().date() - timedelta(days=7)
            
            stats['recent_activity']['draws_today'] = session.query(DrawRecord)\
                .filter(DrawRecord.date >= today).count()
            
            stats['recent_activity']['tickets_week'] = session.query(TicketRecord)\
                .filter(TicketRecord.generated_at >= week_ago).count()
            
            return stats
            
        finally:
            session.close()
    
    def export_data(self, table_name: str, format: str = 'json') -> str:
        """تصدير بيانات الجدول"""
        op_id = logger.start_operation('export_data', {
            'table': table_name,
            'format': format
        })
        
        session = self.get_session()
        
        try:
            # تحديد الجدول
            tables = {
                'draws': DrawRecord,
                'tickets': TicketRecord,
                'predictions': PredictionRecord,
                'user_preferences': UserPreference
            }
            
            if table_name not in tables:
                raise ValueError(f"جدول غير معروف: {table_name}")
            
            Model = tables[table_name]
            records = session.query(Model).all()
            
            # تحويل إلى قاموس
            data = []
            for record in records:
                record_dict = {}
                for column in Model.__table__.columns:
                    value = getattr(record, column.name)
                    # معالجة التواريخ
                    if isinstance(value, datetime):
                        value = value.isoformat()
                    record_dict[column.name] = value
                data.append(record_dict)
            
            # التصدير بالتنسيق المطلوب
            if format == 'json':
                output = json.dumps(data, ensure_ascii=False, indent=2)
            elif format == 'csv':
                import csv
                import io
                
                output_buffer = io.StringIO()
                if data:
                    writer = csv.DictWriter(output_buffer, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
                    output = output_buffer.getvalue()
                else:
                    output = ''
            else:
                raise ValueError(f"تنسيق غير معروف: {format}")
            
            logger.end_operation(op_id, 'completed', {
                'records_exported': len(data),
                'format': format
            })
            
            return output
            
        except Exception as e:
            logger.end_operation(op_id, 'failed', {'error': str(e)})
            raise
        finally:
            session.close()
    
    def backup_database(self, backup_path: str = None):
        """إنشاء نسخة احتياطية من قاعدة البيانات"""
        import shutil
        import os
        
        op_id = logger.start_operation('database_backup', {})
        
        try:
            if not backup_path:
                backup_dir = os.path.join(Config.EXPORT_DIR, 'backups')
                os.makedirs(backup_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = os.path.join(backup_dir, f'db_backup_{timestamp}.db')
            
            # إغلاق جميع الاتصالات
            if self.engine:
                self.engine.dispose()
            
            # نسخ ملف قاعدة البيانات
            if self.db_url.startswith('sqlite:///'):
                db_file = self.db_url.replace('sqlite:///', '')
                if os.path.exists(db_file):
                    shutil.copy2(db_file, backup_path)
                    logger.end_operation(op_id, 'completed', {
                        'backup_path': backup_path,
                        'backup_size': os.path.getsize(backup_path)
                    })
                else:
                    raise FileNotFoundError(f"ملف قاعدة البيانات غير موجود: {db_file}")
            else:
                raise NotImplementedError("النسخ الاحتياطي غير متوفر لهذا النوع من قواعد البيانات")
            
        except Exception as e:
            logger.end_operation(op_id, 'failed', {'error': str(e)})
            raise
    
    def _get_next_draw_id(self, session: Session) -> int:
        """الحصول على رقم السحب التالي"""
        last_draw = session.query(DrawRecord)\
            .order_by(DrawRecord.draw_id.desc())\
            .first()
        
        if last_draw:
            return last_draw.draw_id + 1
        else:
            return 1
    
    def _count_consecutive_pairs(self, numbers: List[int]) -> int:
        """عد الأزواج المتتالية"""
        sorted_nums = sorted(numbers)
        return sum(1 for i in range(len(sorted_nums)-1) 
                  if sorted_nums[i+1] - sorted_nums[i] == 1)
    
    def _create_ticket_hash(self, numbers: List[int]) -> str:
        """إنشاء hash فريد للتذكرة"""
        numbers_str = ','.join(map(str, sorted(numbers)))
        return hashlib.sha256(numbers_str.encode()).hexdigest()
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """تنظيف البيانات القديمة"""
        op_id = logger.start_operation('data_cleanup', {
            'days_to_keep': days_to_keep
        })
        
        session = self.get_session()
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # حذف التذاكر القديمة
            old_tickets = session.query(TicketRecord)\
                .filter(TicketRecord.generated_at < cutoff_date)\
                .delete()
            
            # حذف التنبؤات القديمة
            old_predictions = session.query(PredictionRecord)\
                .filter(PredictionRecord.created_at < cutoff_date)\
                .delete()
            
            session.commit()
            
            logger.end_operation(op_id, 'completed', {
                'old_tickets_deleted': old_tickets,
                'old_predictions_deleted': old_predictions,
                'cutoff_date': cutoff_date.isoformat()
            })
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.end_operation(op_id, 'failed', {'error': str(e)})
            raise
        finally:
            session.close()