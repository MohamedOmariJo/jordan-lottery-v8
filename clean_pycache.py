#!/usr/bin/env python3
"""
سكريبت لحذف جميع مجلدات __pycache__ من المشروع
"""

import os
import shutil
from pathlib import Path

def remove_pycache_dirs(root_path='.'):
    """حذف جميع مجلدات __pycache__ في المشروع"""
    
    root = Path(root_path)
    pycache_dirs = list(root.rglob('__pycache__'))
    
    if not pycache_dirs:
        print("✅ لم يتم العثور على أي مجلدات __pycache__")
        return
    
    print(f"🔍 تم العثور على {len(pycache_dirs)} مجلد __pycache__\n")
    
    removed = 0
    for pycache_dir in pycache_dirs:
        try:
            # عرض المسار النسبي
            rel_path = pycache_dir.relative_to(root)
            print(f"🗑️  حذف: {rel_path}")
            
            # حذف المجلد
            shutil.rmtree(pycache_dir)
            removed += 1
            
        except Exception as e:
            print(f"❌ خطأ في حذف {pycache_dir}: {e}")
    
    print(f"\n✅ تم حذف {removed} من {len(pycache_dirs)} مجلد __pycache__")
    
    # حذف ملفات .pyc المتبقية
    pyc_files = list(root.rglob('*.pyc'))
    if pyc_files:
        print(f"\n🔍 تم العثور على {len(pyc_files)} ملف .pyc")
        for pyc_file in pyc_files:
            try:
                pyc_file.unlink()
                print(f"🗑️  حذف: {pyc_file.relative_to(root)}")
            except Exception as e:
                print(f"❌ خطأ: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🧹 تنظيف مجلدات __pycache__")
    print("=" * 60)
    print()
    
    # تشغيل التنظيف
    remove_pycache_dirs()
    
    print()
    print("=" * 60)
    print("✨ اكتمل التنظيف!")
    print("=" * 60)
