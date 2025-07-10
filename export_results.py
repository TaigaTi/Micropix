from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import io
from pypdf import PdfReader, PdfWriter
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, figsize=(8, 6), cmap='Blues', save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap,
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()


def save_training_plot(base_history, fine_tune_history=None, filename='training_plot.png', base_epochs=0):
    plt.figure(figsize=(8, 6))

    # Combine histories
    acc = base_history.history['accuracy']
    val_acc = base_history.history['val_accuracy']

    if fine_tune_history:
        acc += fine_tune_history.history['accuracy']
        val_acc += fine_tune_history.history['val_accuracy']

    # Plot
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')

    if fine_tune_history:
        plt.axvline(x=base_epochs - 1, color='gray', linestyle='--', label='Fine-tuning Start')

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

def create_report_pdf(config, model, history, fine_tune_history, y_true, y_pred, class_names, BASE_EPOCHS, test_acc):
    buffer = io.BytesIO()

    save_training_plot(history, fine_tune_history, filename='training_plot.png', base_epochs=BASE_EPOCHS)
    plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, save_path="confusion_matrix.png")
    
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
    
    elements.append(Paragraph("Confusion Matrix:", styles['Heading2']))
    elements.append(Image('confusion_matrix.png', width=6*inch, height=4*inch))
    elements.append(Spacer(1, 12))

    doc.build(elements)

    buffer.seek(0)
    return buffer

def export_report(config, model, history, fine_tune_history, y_true, y_pred, class_names, BASE_EPOCHS, test_acc, filename='report.pdf'):
    # Create new report PDF in memory
    new_pdf_buffer = create_report_pdf(config, model, history, fine_tune_history, y_true, y_pred, class_names, BASE_EPOCHS, test_acc)

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
