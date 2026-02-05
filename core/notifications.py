"""
=============================================================================
🔔 نظام الإشعارات المتعدد القنوات
=============================================================================
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import os
from enum import Enum

from config.settings import Config
from utils.logger import logger

class NotificationPriority(Enum):
    """أولويات الإشعارات"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    CRITICAL = "critical"

class NotificationChannel(Enum):
    """قنوات الإرسال"""
    IN_APP = "in_app"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    LOG = "log"

class Notification:
    """تمثيل للإشعار"""
    
    def __init__(self, title: str, message: str, 
                 priority: NotificationPriority = NotificationPriority.INFO,
                 channels: List[NotificationChannel] = None,
                 metadata: Dict = None):
        self.id = self._generate_id()
        self.title = title
        self.message = message
        self.priority = priority
        self.channels = channels or [NotificationChannel.IN_APP, NotificationChannel.LOG]
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.sent_at = None
        self.status = "pending"
        self.retry_count = 0
    
    def _generate_id(self) -> str:
        """توليد معرف فريد للإشعار"""
        import uuid
        return str(uuid.uuid4())
    
    def to_dict(self) -> Dict:
        """تحويل الإشعار إلى قاموس"""
        return {
            'id': self.id,
            'title': self.title,
            'message': self.message,
            'priority': self.priority.value,
            'channels': [c.value for c in self.channels],
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'sent_at': self.sent_at.isoformat() if self.sent_at else None,
            'status': self.status,
            'retry_count': self.retry_count
        }

class NotificationProvider:
    """مزود خدمة الإشعارات الأساسي"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_enabled = True
        self.stats = {
            'total_sent': 0,
            'total_failed': 0,
            'last_sent': None
        }
    
    def send(self, notification: Notification) -> bool:
        """إرسال الإشعار"""
        raise NotImplementedError
    
    def can_send(self, notification: Notification) -> bool:
        """التحقق من إمكانية الإرسال"""
        return self.is_enabled
    
    def update_stats(self, success: bool):
        """تحديث إحصائيات المزود"""
        self.stats['last_sent'] = datetime.now()
        if success:
            self.stats['total_sent'] += 1
        else:
            self.stats['total_failed'] += 1
    
    def get_stats(self) -> Dict:
        """الحصول على إحصائيات المزود"""
        return self.stats.copy()

class InAppProvider(NotificationProvider):
    """مزود الإشعارات داخل التطبيق"""
    
    def __init__(self):
        super().__init__('in_app')
        self.notifications_history = []
        self.max_history = 100
    
    def send(self, notification: Notification) -> bool:
        """إرسال إشعار داخل التطبيق"""
        try:
            # في التطبيق الحقيقي، سيتم عرض هذا في واجهة المستخدم
            # هنا نسجل فقط
            logger.logger.info(f"🔔 إشعار داخل التطبيق: {notification.title}")
            
            # حفظ في التاريخ
            self.notifications_history.append(notification)
            
            # الاحتفاظ فقط بأحدث الإشعارات
            if len(self.notifications_history) > self.max_history:
                self.notifications_history = self.notifications_history[-self.max_history:]
            
            notification.sent_at = datetime.now()
            notification.status = "sent"
            self.update_stats(True)
            
            return True
            
        except Exception as e:
            logger.logger.error(f"❌ فشل إرسال إشعار داخل التطبيق: {e}")
            notification.status = "failed"
            self.update_stats(False)
            return False
    
    def get_recent_notifications(self, limit: int = 20) -> List[Notification]:
        """الحصول على أحدث الإشعارات"""
        return self.notifications_history[-limit:]

class EmailProvider(NotificationProvider):
    """مزود الإشعارات عبر البريد الإلكتروني"""
    
    def __init__(self, smtp_server: str = None, smtp_port: int = 587,
                 username: str = None, password: str = None):
        super().__init__('email')
        
        self.smtp_server = smtp_server or os.getenv('EMAIL_SMTP_SERVER', '')
        self.smtp_port = smtp_port
        self.username = username or os.getenv('EMAIL_USER', '')
        self.password = password or os.getenv('EMAIL_PASSWORD', '')
        
        # تعطيل إذا لم تكن هناك إعدادات
        if not all([self.smtp_server, self.username, self.password]):
            self.is_enabled = False
            logger.logger.warning("⚠️ مزود البريد الإلكتروني معطل - إعدادات غير مكتملة")
    
    def send(self, notification: Notification) -> bool:
        """إرسال إشعار عبر البريد الإلكتروني"""
        if not self.is_enabled:
            return False
        
        try:
            # إنشاء الرسالة
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[Jordan Lottery] {notification.title}"
            msg['From'] = self.username
            msg['To'] = notification.metadata.get('recipient', self.username)
            
            # نص الرسالة
            text = f"""
            {notification.title}
            {'=' * len(notification.title)}
            
            {notification.message}
            
            تاريخ الإرسال: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            الأولوية: {notification.priority.value}
            """
            
            # HTML للمظهر الأفضل
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ 
                        background-color: {'#10b981' if notification.priority == NotificationPriority.SUCCESS else 
                                         '#f59e0b' if notification.priority == NotificationPriority.WARNING else
                                         '#ef4444' if notification.priority == NotificationPriority.ERROR else
                                         '#3b82f6'};
                        color: white;
                        padding: 15px;
                        border-radius: 8px;
                        margin-bottom: 20px;
                    }}
                    .content {{ padding: 20px; background-color: #f9fafb; border-radius: 8px; }}
                    .footer {{ margin-top: 20px; font-size: 12px; color: #6b7280; text-align: center; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h2>{notification.title}</h2>
                    </div>
                    <div class="content">
                        <p>{notification.message.replace(chr(10), '<br>')}</p>
                    </div>
                    <div class="footer">
                        <p>Jordan Lottery AI Pro - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # إضافة المحتوى
            part1 = MIMEText(text, 'plain')
            part2 = MIMEText(html, 'html')
            msg.attach(part1)
            msg.attach(part2)
            
            # الإرسال
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            notification.sent_at = datetime.now()
            notification.status = "sent"
            self.update_stats(True)
            
            logger.logger.info(f"📧 تم إرسال إشعار بالبريد: {notification.title}")
            return True
            
        except Exception as e:
            logger.logger.error(f"❌ فشل إرسال إشعار بالبريد: {e}")
            notification.status = "failed"
            self.update_stats(False)
            return False

class LogProvider(NotificationProvider):
    """مزود الإشعارات عبر السجلات"""
    
    def __init__(self):
        super().__init__('log')
    
    def send(self, notification: Notification) -> bool:
        """تسجيل الإشعار في السجلات"""
        try:
            # استخدام مستوى التسجيل المناسب
            log_level = {
                NotificationPriority.INFO: 'info',
                NotificationPriority.WARNING: 'warning',
                NotificationPriority.ERROR: 'error',
                NotificationPriority.SUCCESS: 'info',
                NotificationPriority.CRITICAL: 'critical'
            }.get(notification.priority, 'info')
            
            # التسجيل
            log_message = f"🔔 {notification.title}: {notification.message}"
            getattr(logger.logger, log_level)(log_message, extra={
                'notification_id': notification.id,
                'priority': notification.priority.value,
                'metadata': notification.metadata
            })
            
            notification.sent_at = datetime.now()
            notification.status = "sent"
            self.update_stats(True)
            
            return True
            
        except Exception as e:
            logger.logger.error(f"❌ فشل تسجيل الإشعار: {e}")
            notification.status = "failed"
            self.update_stats(False)
            return False

class NotificationSystem:
    """نظام الإشعارات الرئيسي"""
    
    def __init__(self):
        self.providers = {}
        self.notifications_queue = []
        self.notifications_history = []
        self.max_history = 1000
        self.retry_limit = 3
        
        # تهيئة المزودين
        self._initialize_providers()
    
    def _initialize_providers(self):
        """تهيئة جميع مزودي الإشعارات"""
        # مزود داخل التطبيق
        self.providers[NotificationChannel.IN_APP] = InAppProvider()
        
        # مزود البريد الإلكتروني
        email_provider = EmailProvider()
        if email_provider.is_enabled:
            self.providers[NotificationChannel.EMAIL] = email_provider
        
        # مزود السجلات
        self.providers[NotificationChannel.LOG] = LogProvider()
        
        logger.logger.info("🔔 نظام الإشعارات مهيأ", extra={
            'providers_count': len(self.providers),
            'providers': list(self.providers.keys())
        })
    
    def send(self, title: str, message: str, 
            priority: NotificationPriority = NotificationPriority.INFO,
            channels: List[NotificationChannel] = None,
            metadata: Dict = None) -> Dict[str, Any]:
        """إرسال إشعار"""
        op_id = logger.start_operation('send_notification', {
            'title': title,
            'priority': priority.value
        })
        
        try:
            # إنشاء الإشعار
            notification = Notification(
                title=title,
                message=message,
                priority=priority,
                channels=channels or [NotificationChannel.IN_APP, NotificationChannel.LOG],
                metadata=metadata or {}
            )
            
            # إضافة إلى قائمة الانتظار
            self.notifications_queue.append(notification)
            
            # معالجة الإشعار
            result = self._process_notification(notification)
            
            # حفظ في التاريخ
            self._add_to_history(notification)
            
            logger.end_operation(op_id, 'completed', {
                'notification_id': notification.id,
                'status': notification.status,
                'channels_used': result
            })
            
            return {
                'notification_id': notification.id,
                'status': notification.status,
                'channels': result,
                'created_at': notification.created_at
            }
            
        except Exception as e:
            logger.end_operation(op_id, 'failed', {'error': str(e)})
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _process_notification(self, notification: Notification) -> Dict[str, bool]:
        """معالجة إرسال الإشعار عبر جميع القنوات"""
        results = {}
        
        for channel in notification.channels:
            if channel in self.providers:
                provider = self.providers[channel]
                
                # التحقق من إمكانية الإرسال
                if not provider.can_send(notification):
                    results[channel.value] = False
                    continue
                
                # المحاولة مع إعادة المحاولة
                success = False
                for attempt in range(self.retry_limit):
                    try:
                        success = provider.send(notification)
                        if success:
                            break
                        
                        notification.retry_count += 1
                        logger.logger.warning(
                            f"⚠️ إعادة محاولة إرسال الإشعار {notification.id} "
                            f"عبر {channel.value} (المحاولة {attempt + 1})"
                        )
                        
                    except Exception as e:
                        logger.logger.error(
                            f"❌ خطأ في إرسال الإشعار عبر {channel.value}: {e}"
                        )
                
                results[channel.value] = success
                
                if not success:
                    notification.status = "partially_failed"
            
            else:
                results[channel.value] = False
                logger.logger.warning(f"⚠️ قناة إشعار غير معروفة: {channel.value}")
        
        # تحديث حالة الإشعار
        if all(results.values()):
            notification.status = "sent"
        elif any(results.values()):
            notification.status = "partially_sent"
        else:
            notification.status = "failed"
        
        return results
    
    def _add_to_history(self, notification: Notification):
        """إضافة الإشعار إلى التاريخ"""
        self.notifications_history.append(notification)
        
        # الاحتفاظ فقط بأحدث الإشعارات
        if len(self.notifications_history) > self.max_history:
            self.notifications_history = self.notifications_history[-self.max_history:]
    
    def get_notifications(self, limit: int = 50, 
                         priority: NotificationPriority = None,
                         status: str = None) -> List[Dict]:
        """الحصول على الإشعارات"""
        filtered = self.notifications_history.copy()
        
        if priority:
            filtered = [n for n in filtered if n.priority == priority]
        
        if status:
            filtered = [n for n in filtered if n.status == status]
        
        # ترتيب حسب التاريخ
        filtered.sort(key=lambda x: x.created_at, reverse=True)
        
        # الحد الأقصى
        filtered = filtered[:limit]
        
        return [n.to_dict() for n in filtered]
    
    def get_provider_stats(self) -> Dict[str, Dict]:
        """الحصول على إحصائيات جميع المزودين"""
        stats = {}
        
        for channel, provider in self.providers.items():
            stats[channel.value] = provider.get_stats()
        
        return stats
    
    def send_bulk(self, notifications: List[Dict]) -> List[Dict]:
        """إرسال إشعارات جماعية"""
        results = []
        
        for notification_data in notifications:
            result = self.send(
                title=notification_data.get('title', ''),
                message=notification_data.get('message', ''),
                priority=NotificationPriority(notification_data.get('priority', 'info')),
                channels=[NotificationChannel(c) for c in notification_data.get('channels', ['in_app'])],
                metadata=notification_data.get('metadata', {})
            )
            results.append(result)
        
        return results
    
    def schedule_notification(self, title: str, message: str, 
                            send_time: datetime,
                            priority: NotificationPriority = NotificationPriority.INFO,
                            channels: List[NotificationChannel] = None,
                            metadata: Dict = None) -> str:
        """جدولة إشعار للوقت المستقبلي"""
        # هذا يحتاج إلى نظام جدولة (مثل APScheduler)
        # هنا نسجل فقط
        notification_id = f"scheduled_{datetime.now().timestamp()}"
        
        logger.logger.info(f"📅 جدولة إشعار: {title} للوقت {send_time}", extra={
            'notification_id': notification_id,
            'send_time': send_time.isoformat(),
            'priority': priority.value
        })
        
        return notification_id
    
    def clear_notifications(self, older_than_days: int = 30):
        """مسح الإشعارات القديمة"""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        old_count = len(self.notifications_history)
        self.notifications_history = [
            n for n in self.notifications_history 
            if n.created_at > cutoff_date
        ]
        new_count = len(self.notifications_history)
        
        logger.logger.info(f"🧹 مسح الإشعارات القديمة", extra={
            'old_count': old_count,
            'new_count': new_count,
            'removed_count': old_count - new_count,
            'cutoff_date': cutoff_date.isoformat()
        })
    
    def export_notifications(self, format: str = 'json') -> str:
        """تصدير الإشعارات"""
        notifications_data = [n.to_dict() for n in self.notifications_history]
        
        if format == 'json':
            return json.dumps(notifications_data, ensure_ascii=False, indent=2)
        elif format == 'csv':
            import csv
            import io
            
            if not notifications_data:
                return ''
            
            output_buffer = io.StringIO()
            writer = csv.DictWriter(output_buffer, fieldnames=notifications_data[0].keys())
            writer.writeheader()
            writer.writerows(notifications_data)
            return output_buffer.getvalue()
        else:
            raise ValueError(f"تنسيق غير معروف: {format}")

# وظائف مساعدة للاستخدام السريع
def notify_info(title: str, message: str, metadata: Dict = None):
    """إرسال إشعار معلومات"""
    notification_system = NotificationSystem()
    return notification_system.send(
        title=title,
        message=message,
        priority=NotificationPriority.INFO,
        metadata=metadata
    )

def notify_success(title: str, message: str, metadata: Dict = None):
    """إرسال إشعار نجاح"""
    notification_system = NotificationSystem()
    return notification_system.send(
        title=title,
        message=message,
        priority=NotificationPriority.SUCCESS,
        metadata=metadata
    )

def notify_warning(title: str, message: str, metadata: Dict = None):
    """إرسال إشعار تحذير"""
    notification_system = NotificationSystem()
    return notification_system.send(
        title=title,
        message=message,
        priority=NotificationPriority.WARNING,
        metadata=metadata
    )

def notify_error(title: str, message: str, metadata: Dict = None):
    """إرسال إشعار خطأ"""
    notification_system = NotificationSystem()
    return notification_system.send(
        title=title,
        message=message,
        priority=NotificationPriority.ERROR,
        metadata=metadata
    )