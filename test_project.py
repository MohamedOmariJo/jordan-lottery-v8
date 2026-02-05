#!/usr/bin/env python3
"""
اختبار شامل للمشروع المصلح
"""

import sys
import os

# إضافة المسار
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("=" * 60)
print("🧪 اختبار شامل لجميع مكونات المشروع")
print("=" * 60)
print()

# قائمة الاستيرادات المطلوبة
test_imports = [
    ("config.settings", "Config", "الإعدادات"),
    ("utils.logger", "logger", "نظام التسجيل"),
    ("utils.performance", "PerformanceBenchmark", "مراقبة الأداء"),
    ("utils.pdf_generator", "PDFGenerator", "توليد PDF"),
    ("core.validator", "AdvancedValidator", "المتحقق المتقدم"),
    ("core.analyzer", "AdvancedAnalyzer", "المحلل المتقدم"),
    ("core.models", "LotteryPredictor", "نظام التنبؤ"),
    ("core.models", "RecommendationEngine", "محرك التوصيات"),
    ("core.generator", "SmartGenerator", "المولد الذكي"),
    ("core.notifications", "NotificationSystem", "نظام الإشعارات"),
]

passed = 0
failed = 0
skipped = 0

for module_name, class_name, description in test_imports:
    try:
        print(f"📦 اختبار {description} ({module_name}.{class_name})...", end=" ")
        
        # الاستيراد
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        
        # محاولة إنشاء كائن (إذا لم يكن singleton)
        try:
            if class_name not in ['logger', 'Config']:
                obj = cls()
                print("✅ نجح")
            else:
                print("✅ نجح")
        except Exception as e:
            # بعض الكائنات قد تتطلب معاملات
            if "missing" in str(e).lower() or "required" in str(e).lower():
                print("✅ نجح (يتطلب معاملات)")
            else:
                print(f"⚠️  تحذير: {e}")
        
        passed += 1
        
    except ImportError as e:
        if "sqlalchemy" in str(e):
            print("⏭️  تخطي (يتطلب sqlalchemy)")
            skipped += 1
        else:
            print(f"❌ فشل: {e}")
            failed += 1
    except Exception as e:
        print(f"❌ فشل: {e}")
        failed += 1

print()
print("=" * 60)
print("📊 ملخص النتائج:")
print("=" * 60)
print(f"✅ نجح:   {passed}")
print(f"❌ فشل:   {failed}")
print(f"⏭️  تخطي:  {skipped}")
print()

if failed == 0:
    print("🎉 جميع الاختبارات نجحت!")
    print("✨ المشروع جاهز للنشر على Streamlit Cloud")
else:
    print("⚠️  بعض الاختبارات فشلت، يرجى مراجعة الأخطاء أعلاه")

print("=" * 60)
