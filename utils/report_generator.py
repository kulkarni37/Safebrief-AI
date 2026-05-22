from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def create_pdf(summary, risk):

    pdf_file = "Safety_Report.pdf"

    doc = SimpleDocTemplate(pdf_file)

    styles = getSampleStyleSheet()

    elements = []

    title = Paragraph(
        "<b>SafeBrief AI Report</b>",
        styles['Title']
    )

    elements.append(title)

    elements.append(Spacer(1, 20))

    summary_text = Paragraph(
        f"<b>Incident Summary:</b><br/>{summary}",
        styles['BodyText']
    )

    elements.append(summary_text)

    elements.append(Spacer(1, 20))

    risk_text = Paragraph(
        f"<b>Risk Level:</b> {risk}",
        styles['BodyText']
    )

    elements.append(risk_text)

    elements.append(Spacer(1, 20))

    recommendation = Paragraph(
        """
        <b>Recommendations:</b><br/>
        • Improve maintenance<br/>
        • Conduct inspections<br/>
        • Train employees<br/>
        • Strengthen safety compliance
        """,
        styles['BodyText']
    )

    elements.append(recommendation)

    doc.build(elements)

    return pdf_file