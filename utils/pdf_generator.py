"""
=============================================================================
📄 مولد PDF احترافي مع تصميمات متقدمة
=============================================================================
"""

from reportlab.lib.pagesizes import A4, LETTER
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import cm, inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from typing import List, Dict, Optional, Tuple
import io
from datetime import datetime
import random

from config.settings import Config
from utils.logger import logger

class PDFGenerator:
    """مولد PDF احترافي مع تصميمات متقدمة"""
    
    @staticmethod
    def create_ticket_pdf(tickets: List[List[int]], 
                         metadata: Optional[Dict] = None,
                         design: str = 'professional') -> io.BytesIO:
        """إنشاء ملف PDF للتذاكر بتصميم محسن"""
        op_id = logger.start_operation('pdf_generation', {
            'tickets_count': len(tickets),
            'design': design
        })
        
        try:
            buffer = io.BytesIO()
            
            if design == 'professional':
                pdf = PDFGenerator._create_professional_pdf(buffer, tickets, metadata)
            elif design == 'minimal':
                pdf = PDFGenerator._create_minimal_pdf(buffer, tickets, metadata)
            elif design == 'colorful':
                pdf = PDFGenerator._create_colorful_pdf(buffer, tickets, metadata)
            else:
                pdf = PDFGenerator._create_professional_pdf(buffer, tickets, metadata)
            
            logger.end_operation(op_id, 'completed', {
                'pdf_size': buffer.getbuffer().nbytes,
                'design_used': design
            })
            
            return pdf
            
        except Exception as e:
            logger.end_operation(op_id, 'failed', {'error': str(e)})
            raise
    
    @staticmethod
    def _create_professional_pdf(buffer: io.BytesIO, 
                                tickets: List[List[int]], 
                                metadata: Optional[Dict]) -> io.BytesIO:
        """إنشاء PDF احترافي"""
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        styles = getSampleStyleSheet()
        
        # أنماط مخصصة
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1e40af'),
            alignment=TA_CENTER,
            spaceAfter=30
        )
        
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#374151'),
            spaceAfter=15
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#4b5563')
        )
        
        # المحتوى
        story = []
        
        # العنوان
        story.append(Paragraph("🎰 Jordan Lottery AI Pro", title_style))
        story.append(Spacer(1, 0.5*cm))
        
        # معلومات الملف
        info_text = f"""
        <b>تاريخ التوليد:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>عدد التذاكر:</b> {len(tickets)}<br/>
        <b>الإستراتيجية:</b> {metadata.get('strategy', 'Smart Generation') if metadata else 'Smart Generation'}<br/>
        <b>الإصدار:</b> {Config.APP_VERSION}
        """
        
        story.append(Paragraph(info_text, normal_style))
        story.append(Spacer(1, 1*cm))
        
        # التذاكر
        for i, ticket in enumerate(tickets, 1):
            # عنوان التذكرة
            story.append(Paragraph(f"🎫 التذكرة #{i}", header_style))
            
            # جدول التذكرة
            ticket_data = []
            for num in ticket:
                # تحديد اللون بناءً على حالة الرقم
                bg_color = PDFGenerator._get_number_color(num)
                
                ticket_data.append([
                    Paragraph(
                        f"<font size='14' color='white'><b>{num}</b></font>",
                        ParagraphStyle(
                            'BallStyle',
                            parent=styles['Normal'],
                            alignment=TA_CENTER,
                            backColor=bg_color,
                            borderColor=colors.white,
                            borderWidth=1,
                            borderPadding=5,
                            borderRadius=10
                        )
                    )
                ])
            
            # إنشاء جدول
            table = Table(ticket_data, colWidths=[1.5*cm] * len(ticket))
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.transparent),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOX', (0, 0), (-1, -1), 1, colors.grey),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            
            story.append(table)
            story.append(Spacer(1, 0.3*cm))
            
            # معلومات التذكرة
            ticket_sum = sum(ticket)
            odd_count = sum(1 for n in ticket if n % 2)
            ticket_info = f"""
            <b>المجموع:</b> {ticket_sum} | 
            <b>الفردي/الزوجي:</b> {odd_count}/{len(ticket)-odd_count} | 
            <b>المتتاليات:</b> {sum(1 for i in range(len(ticket)-1) if ticket[i+1] - ticket[i] == 1)}
            """
            
            story.append(Paragraph(ticket_info, normal_style))
            story.append(Spacer(1, 0.8*cm))
            
            # صفحة جديدة كل 5 تذاكر
            if i % 5 == 0 and i < len(tickets):
                story.append(Paragraph(f"-- صفحة {i//5 + 1} --", 
                                     ParagraphStyle('PageBreak', alignment=TA_CENTER)))
                story.append(Spacer(1, 1*cm))
        
        # التذييل
        story.append(Spacer(1, 2*cm))
        footer_text = """
        <font size='8' color='#6b7280'>
        <i>تم إنشاء هذا الملف بواسطة Jordan Lottery AI Pro v8.0<br/>
        هذا الملف للأغراض الترفيهية والتعليمية فقط<br/>
        جميع الحقوق محفوظة © 2026</i>
        </font>
        """
        story.append(Paragraph(footer_text, ParagraphStyle('Footer', alignment=TA_CENTER)))
        
        # بناء PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer
    
    @staticmethod
    def _create_minimal_pdf(buffer: io.BytesIO, 
                           tickets: List[List[int]], 
                           metadata: Optional[Dict]) -> io.BytesIO:
        """إنشاء PDF بسيط وأنيق"""
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=1.5*cm,
            leftMargin=1.5*cm,
            topMargin=1.5*cm,
            bottomMargin=1.5*cm
        )
        
        styles = getSampleStyleSheet()
        
        # أنماط مخصصة
        title_style = ParagraphStyle(
            'MinimalTitle',
            parent=styles['Title'],
            fontSize=20,
            textColor=colors.black,
            alignment=TA_CENTER,
            spaceAfter=20
        )
        
        # المحتوى
        story = []
        
        # العنوان
        story.append(Paragraph("Jordan Lottery Tickets", title_style))
        story.append(Spacer(1, 0.5*cm))
        
        # معلومات
        info_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Tickets: {len(tickets)}"
        story.append(Paragraph(info_text, styles['Normal']))
        story.append(Spacer(1, 1*cm))
        
        # التذاكر
        for i, ticket in enumerate(tickets, 1):
            # رقم التذكرة
            story.append(Paragraph(f"Ticket #{i}", styles['Heading3']))
            
            # الأرقام
            numbers_text = " | ".join([f"<b>{num}</b>" for num in ticket])
            story.append(Paragraph(numbers_text, styles['Normal']))
            
            # معلومات سريعة
            ticket_sum = sum(ticket)
            odd_count = sum(1 for n in ticket if n % 2)
            quick_info = f"Sum: {ticket_sum} | Odd: {odd_count} | Even: {len(ticket)-odd_count}"
            story.append(Paragraph(quick_info, styles['Small']))
            
            story.append(Spacer(1, 0.5*cm))
            
            # خط فاصل
            if i < len(tickets):
                story.append(Paragraph("<hr width='100%'/>", styles['Normal']))
                story.append(Spacer(1, 0.3*cm))
        
        # بناء PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer
    
    @staticmethod
    def _create_colorful_pdf(buffer: io.BytesIO, 
                            tickets: List[List[int]], 
                            metadata: Optional[Dict]) -> io.BytesIO:
        """إنشاء PDF ملون وجذاب"""
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        
        # خلفية متدرجة
        c.setFillColorRGB(0.95, 0.95, 0.98)
        c.rect(0, 0, width, height, fill=1, stroke=0)
        
        # العنوان مع تأثير
        c.setFont("Helvetica-Bold", 28)
        c.setFillColorRGB(0.2, 0.4, 0.8)
        c.drawString(2*cm, height - 3*cm, "🎰 Jordan Lottery")
        
        c.setFont("Helvetica", 12)
        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.drawString(2*cm, height - 3.7*cm, f"AI Generated Tickets - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # معلومات
        c.setFont("Helvetica", 10)
        c.setFillColorRGB(0.3, 0.3, 0.3)
        c.drawString(2*cm, height - 4.4*cm, f"Total Tickets: {len(tickets)}")
        
        if metadata:
            c.drawString(2*cm, height - 5*cm, f"Strategy: {metadata.get('strategy', 'Smart')}")
        
        # خط فاصل
        c.setStrokeColorRGB(0.2, 0.4, 0.8)
        c.setLineWidth(2)
        c.line(2*cm, height - 5.5*cm, width - 2*cm, height - 5.5*cm)
        
        # إعدادات الرسم
        y = height - 7*cm
        tickets_per_page = 8
        
        for i, ticket in enumerate(tickets, 1):
            # صفحة جديدة إذا لزم الأمر
            if y < 5*cm:
                c.showPage()
                
                # خلفية الصفحة الجديدة
                c.setFillColorRGB(0.95, 0.95, 0.98)
                c.rect(0, 0, width, height, fill=1, stroke=0)
                
                y = height - 3*cm
                c.setFont("Helvetica-Bold", 16)
                c.setFillColorRGB(0.2, 0.4, 0.8)
                c.drawString(2*cm, y, f"🎰 Jordan Lottery - Page {i//tickets_per_page + 1}")
                y -= 1.5*cm
            
            # رسم بطاقة التذكرة
            PDFGenerator._draw_ticket_card(c, ticket, i, 2*cm, y - 2.5*cm, width - 4*cm, 2.5*cm)
            
            y -= 3.2*cm
        
        # التذييل
        c.setFont("Helvetica", 8)
        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.drawString(2*cm, 2*cm, "Generated by Jordan Lottery AI Pro v8.0 - For educational purposes only")
        c.drawString(width - 10*cm, 2*cm, f"Page 1 of {(len(tickets) + tickets_per_page - 1) // tickets_per_page}")
        
        c.save()
        buffer.seek(0)
        
        return buffer
    
    @staticmethod
    def _draw_ticket_card(c: canvas.Canvas, ticket: List[int], 
                         ticket_num: int, x: float, y: float, 
                         width: float, height: float):
        """رسم بطاقة تذكرة ملونة"""
        # خلفية البطاقة
        c.setFillColorRGB(1, 1, 1)
        c.setStrokeColorRGB(0.8, 0.8, 0.8)
        c.setLineWidth(1)
        c.roundRect(x, y, width, height, 10, fill=1, stroke=1)
        
        # ظل خفيف
        c.setFillColorRGB(0.95, 0.95, 0.95)
        c.roundRect(x + 2, y - 2, width, height, 10, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1)
        c.roundRect(x, y, width, height, 10, fill=1, stroke=1)
        
        # عنوان التذكرة
        c.setFont("Helvetica-Bold", 14)
        c.setFillColorRGB(0.2, 0.2, 0.2)
        c.drawString(x + 1*cm, y + height - 1.2*cm, f"🎫 Ticket #{ticket_num}")
        
        # رسم الكرات
        ball_x = x + 1*cm
        ball_y = y + height - 2.2*cm
        ball_radius = 0.6*cm
        
        for num in sorted(ticket):
            # لون الكرة
            ball_color = PDFGenerator._get_number_color_rgb(num)
            c.setFillColorRGB(*ball_color)
            
            # رسم الكرة
            c.circle(ball_x, ball_y, ball_radius, fill=1, stroke=0)
            
            # حدود الكرة
            c.setStrokeColorRGB(1, 1, 1)
            c.setLineWidth(2)
            c.circle(ball_x, ball_y, ball_radius, fill=0, stroke=1)
            
            # الرقم
            c.setFillColorRGB(1, 1, 1)
            c.setFont("Helvetica-Bold", 12)
            c.drawCentredString(ball_x, ball_y - 0.15*cm, str(num))
            
            ball_x += 1.5*cm
        
        # معلومات التذكرة
        c.setFont("Helvetica", 9)
        c.setFillColorRGB(0.5, 0.5, 0.5)
        
        ticket_sum = sum(ticket)
        odd_count = sum(1 for n in ticket if n % 2)
        consec_count = sum(1 for i in range(len(ticket)-1) if ticket[i+1] - ticket[i] == 1)
        
        info_text = f"Sum: {ticket_sum} | Odd/Even: {odd_count}/{len(ticket)-odd_count} | Consecutive: {consec_count}"
        c.drawString(x + 1*cm, y + 0.5*cm, info_text)
    
    @staticmethod
    def _get_number_color(num: int) -> colors.Color:
        """الحصول على لون الرقم بناءً على حالته"""
        # هذه دالة وهمية - في التطبيق الحقيقي ستحتاج إلى analyzer
        # لتحديد إذا كان الرقم ساخن/بارد/محايد
        colors_map = {
            'hot': colors.HexColor('#ef4444'),      # أحمر
            'cold': colors.HexColor('#3b82f6'),     # أزرق
            'neutral': colors.HexColor('#10b981')   # أخضر
        }
        
        # توزيع عشوائي للعرض (في التطبيق الحقيقي استخدم analyzer)
        if num % 3 == 0:
            return colors_map['hot']
        elif num % 3 == 1:
            return colors_map['cold']
        else:
            return colors_map['neutral']
    
    @staticmethod
    def _get_number_color_rgb(num: int) -> Tuple[float, float, float]:
        """الحصول على لون RGB للرقم"""
        # توزيع ألوان مختلف
        colors_rgb = [
            (0.87, 0.27, 0.27),  # أحمر
            (0.23, 0.51, 0.97),  # أزرق
            (0.06, 0.72, 0.51),  # أخضر
            (0.97, 0.61, 0.23),  # برتقالي
            (0.58, 0.29, 0.97),  # بنفسجي
        ]
        
        return colors_rgb[num % len(colors_rgb)]
    
    @staticmethod
    def create_statistics_pdf(analyzer_data: Dict, 
                             predictions: List[Dict] = None) -> io.BytesIO:
        """إنشاء PDF للإحصائيات والتحليلات"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        
        styles = getSampleStyleSheet()
        
        # المحتوى
        story = []
        
        # العنوان
        story.append(Paragraph("📊 Lottery Statistics Report", styles['Title']))
        story.append(Spacer(1, 1*cm))
        
        # معلومات أساسية
        story.append(Paragraph("Basic Statistics", styles['Heading2']))
        
        basic_stats = f"""
        Total Draws: {analyzer_data.get('total_draws', 0)}<br/>
        Date Range: {analyzer_data.get('date_range', 'N/A')}<br/>
        Most Frequent Number: {analyzer_data.get('most_frequent', 'N/A')}<br/>
        Least Frequent Number: {analyzer_data.get('least_frequent', 'N/A')}<br/>
        Average Sum: {analyzer_data.get('avg_sum', 0):.1f}
        """
        
        story.append(Paragraph(basic_stats, styles['Normal']))
        story.append(Spacer(1, 0.5*cm))
        
        # توزيع الأرقام
        if 'number_distribution' in analyzer_data:
            story.append(Paragraph("Number Distribution", styles['Heading2']))
            
            # إنشاء جدول للتوزيع
            dist_data = [['Number', 'Frequency', 'Percentage']]
            for num, freq in analyzer_data['number_distribution'].items():
                percentage = (freq / analyzer_data['total_draws']) * 100 if analyzer_data['total_draws'] > 0 else 0
                dist_data.append([str(num), str(freq), f"{percentage:.1f}%"])
            
            table = Table(dist_data, colWidths=[2*cm, 3*cm, 3*cm])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
            story.append(Spacer(1, 0.5*cm))
        
        # التنبؤات
        if predictions:
            story.append(Paragraph("AI Predictions", styles['Heading2']))
            
            pred_data = [['Rank', 'Number', 'Probability']]
            for i, pred in enumerate(predictions[:10], 1):
                pred_data.append([
                    str(i),
                    str(pred.get('number', '')),
                    f"{pred.get('probability', 0):.1%}"
                ])
            
            pred_table = Table(pred_data, colWidths=[2*cm, 3*cm, 4*cm])
            pred_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            
            story.append(pred_table)
        
        # التذييل
        story.append(Spacer(1, 2*cm))
        footer = Paragraph(
            f"Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')} by Jordan Lottery AI Pro",
            ParagraphStyle('Footer', fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
        )
        story.append(footer)
        
        # بناء PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer
    
    @staticmethod
    def create_portfolio_pdf(portfolio: List[Dict], 
                            user_info: Dict = None) -> io.BytesIO:
        """إنشاء PDF للمحفظة الشخصية"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        
        styles = getSampleStyleSheet()
        
        # المحتوى
        story = []
        
        # العنوان
        story.append(Paragraph("💼 Lottery Portfolio", styles['Title']))
        story.append(Spacer(1, 0.5*cm))
        
        # معلومات المستخدم
        if user_info:
            user_text = f"""
            <b>User:</b> {user_info.get('name', 'Anonymous')}<br/>
            <b>Portfolio Size:</b> {len(portfolio)} tickets<br/>
            <b>Created:</b> {datetime.now().strftime('%Y-%m-%d')}
            """
            story.append(Paragraph(user_text, styles['Normal']))
            story.append(Spacer(1, 0.5*cm))
        
        # التذاكر
        for i, ticket_data in enumerate(portfolio, 1):
            ticket = ticket_data.get('numbers', [])
            analysis = ticket_data.get('analysis', {})
            
            story.append(Paragraph(f"Ticket #{i}", styles['Heading3']))
            
            # الأرقام
            numbers_text = "  ".join([f"<font size='12'><b>{num}</b></font>" for num in ticket])
            story.append(Paragraph(numbers_text, styles['Normal']))
            
            # التحليل
            if analysis:
                analysis_text = f"""
                Sum: {analysis.get('sum', 0)} | 
                Odd: {analysis.get('odd', 0)} | 
                Hot Numbers: {analysis.get('hot_count', 0)} | 
                Quality Score: {analysis.get('quality_score', 0)}/10
                """
                story.append(Paragraph(analysis_text, styles['Small']))
            
            story.append(Spacer(1, 0.3*cm))
            
            # خط فاصل
            if i < len(portfolio):
                story.append(Paragraph("<hr width='80%'/>", styles['Normal']))
                story.append(Spacer(1, 0.3*cm))
        
        # إحصائيات المحفظة
        story.append(Spacer(1, 1*cm))
        story.append(Paragraph("Portfolio Statistics", styles['Heading2']))
        
        # حساب الإحصائيات
        if portfolio:
            total_tickets = len(portfolio)
            avg_sum = sum(t.get('analysis', {}).get('sum', 0) for t in portfolio) / total_tickets
            avg_quality = sum(t.get('analysis', {}).get('quality_score', 0) for t in portfolio) / total_tickets
            
            stats_text = f"""
            <b>Total Tickets:</b> {total_tickets}<br/>
            <b>Average Sum:</b> {avg_sum:.1f}<br/>
            <b>Average Quality Score:</b> {avg_quality:.1f}/10<br/>
            <b>Most Common Number:</b> {PDFGenerator._get_most_common_number(portfolio)}
            """
            story.append(Paragraph(stats_text, styles['Normal']))
        
        # التذييل
        story.append(Spacer(1, 2*cm))
        footer = Paragraph(
            "Keep this portfolio for future reference. Good luck! 🍀",
            ParagraphStyle('Footer', fontSize=9, textColor=colors.green, alignment=TA_CENTER)
        )
        story.append(footer)
        
        # بناء PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer
    
    @staticmethod
    def _get_most_common_number(portfolio: List[Dict]) -> str:
        """الحصول على الرقم الأكثر شيوعاً في المحفظة"""
        from collections import Counter
        
        all_numbers = []
        for ticket_data in portfolio:
            ticket = ticket_data.get('numbers', [])
            all_numbers.extend(ticket)
        
        if not all_numbers:
            return "N/A"
        
        counter = Counter(all_numbers)
        most_common = counter.most_common(1)[0]
        
        return f"{most_common[0]} (appears {most_common[1]} times)"