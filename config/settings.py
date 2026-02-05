"""
=============================================================================
🎯 إعدادات التطبيق المحسنة v8.0
=============================================================================
"""

import os
from typing import Dict, Any
from datetime import timedelta
import secrets

class Config:
    """إعدادات مركزية متقدمة للتطبيق"""
    
    # إصدار التطبيق
    APP_VERSION = "8.0.0 PRO"
    APP_NAME = "Jordan Lottery AI Pro"
    
    # الأمان
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
    SESSION_TIMEOUT = timedelta(hours=2)
    
    # البيانات
    GITHUB_URL = "https://raw.githubusercontent.com/MohamedOmariJo/omari/main/250.xlsx"
    BACKUP_FILE = "data/history.xlsx"
    DATABASE_URL = "sqlite:///data/lottery_v8.db"
    
    # نطاق الأرقام
    MIN_NUMBER = 1
    MAX_NUMBER = 32
    DEFAULT_TICKET_SIZE = 6
    
    # التوليد
    MAX_TICKETS_PER_GENERATION = 100
    MAX_GENERATION_ATTEMPTS = 1000
    BATCH_SIZE = 10000
    
    # الذاكرة المؤقتة
    CACHE_TTL = 3600  # ساعة واحدة
    MODEL_CACHE_TTL = 86400  # يوم واحد
    
    # ML وإحصاءات
    MONTE_CARLO_SIMULATIONS = 50000
    MARKOV_MIN_OCCURRENCES = 3
    POISSON_CONFIDENCE_LEVEL = 0.95
    
    # التنسيقات
    DATE_FORMAT = "%Y-%m-%d"
    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # المسارات
    LOGS_DIR = "logs"
    MODELS_DIR = "data/models"
    EXPORT_DIR = "exports"
    
    # API Keys (يجب تعيينها في environment variables)
    TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")
    TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "")
    EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "")
    
    # إعدادات الأداء
    ENABLE_PROFILING = False
    MAX_MEMORY_USAGE_MB = 512
    MAX_CPU_PERCENT = 80
    
    @classmethod
    def get_database_config(cls) -> Dict[str, Any]:
        """الحصول على إعدادات قاعدة البيانات"""
        return {
            'url': cls.DATABASE_URL,
            'pool_size': 10,
            'max_overflow': 20,
            'pool_timeout': 30,
            'pool_recycle': 3600
        }
    
    @classmethod
    def get_logging_config(cls) -> Dict[str, Any]:
        """الحصول على إعدادات Logging"""
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'datefmt': cls.DATETIME_FORMAT
                },
                'simple': {
                    'format': '%(levelname)s: %(message)s'
                }
            },
            'handlers': {
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': os.path.join(cls.LOGS_DIR, 'app.log'),
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5,
                    'formatter': 'detailed',
                    'level': 'INFO'
                },
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'simple',
                    'level': 'WARNING'
                }
            },
            'loggers': {
                'lottery': {
                    'handlers': ['file', 'console'],
                    'level': 'INFO',
                    'propagate': True
                }
            }
        }