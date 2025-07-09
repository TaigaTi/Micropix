from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import io
from pypdf import PdfReader, PdfWriter

def save_training_plot(history, filename='training_plot.png'):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_str = stream.getvalue()
    stream.close()
    return summary_str

def create_report_pdf(config, model, history, test_acc):
    buffer = io.BytesIO()

    save_training_plot(history)
    model_summary = get_model_summary(model)

    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    normal = styles['Normal']
    heading = styles['Heading1']
    mono = ParagraphStyle('monospace', fontName='Courier', fontSize=8, leading=10)

    elements = []

    elements.append(Paragraph("Model Training Report", heading))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Configuration and Parameters:", styles['Heading2']))
    for key, value in config.items():
        elements.append(Paragraph(f"<b>{key}:</b> {value}", normal))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Model Summary:", styles['Heading2']))
    for line in model_summary.split('\n'):
        elements.append(Paragraph(line.replace(' ', '&nbsp;'), mono))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"<b>Test Accuracy:</b> {test_acc:.4f}", normal))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Training Accuracy over Epochs:", styles['Heading2']))
    elements.append(Image('training_plot.png', width=6*inch, height=4*inch))
    elements.append(Spacer(1, 12))

    doc.build(elements)

    buffer.seek(0)
    return buffer

def export_report(config, model, history, test_acc, filename='report.pdf'):
    # Create new report PDF in memory
    new_pdf_buffer = create_report_pdf(config, model, history, test_acc)

    writer = PdfWriter()

    try:
        # Read existing PDF if it exists
        existing_reader = PdfReader(filename)
        for page in existing_reader.pages:
            writer.add_page(page)
    except FileNotFoundError:
        print(f"No existing PDF found, creating new file {filename}.")

    # Add new report pages
    new_reader = PdfReader(new_pdf_buffer)
    for page in new_reader.pages:
        writer.add_page(page)

    # Write all pages back to file (existing + new)
    with open(filename, 'wb') as f_out:
        writer.write(f_out)

    print(f"Appended report to {filename}")
