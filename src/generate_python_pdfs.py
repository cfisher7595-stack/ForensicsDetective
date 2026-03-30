#!/usr/bin/env python3
"""
Python PDF Generator for Forensic Research

Generates PDFs programmatically using ReportLab to create a third class
for PDF provenance detection. Uses the same Wikipedia content as Word/Google
to ensure consistent experimental conditions.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import docx

def extract_text_from_docx(docx_path):
    """Extract plain text from a Word document."""
    try:
        doc = docx.Document(docx_path)
        text_content = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content.append(paragraph.text.strip())
        
        return text_content
    except Exception as e:
        print(f"Error reading {docx_path}: {e}")
        return None

def create_pdf_from_text(text_content, output_path, title):
    """
    Generate PDF using ReportLab from text content.
    
    Args:
        text_content (list): List of paragraphs
        output_path (str): Output PDF file path
        title (str): Document title
    """
    # Create PDF document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Get default styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=12,
        alignment=0  # Left alignment
    )
    
    # Build story (content)
    story = []
    
    # Add title
    story.append(Paragraph(title.replace('_', ' '), title_style))
    story.append(Spacer(1, 12))
    
    # Add paragraphs
    for paragraph_text in text_content:
        if paragraph_text.strip():
            # Clean up text for ReportLab
            clean_text = paragraph_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(clean_text, body_style))
            story.append(Spacer(1, 6))
    
    # Build PDF
    doc.build(story)

def generate_python_pdfs(docx_dir='wikipedia_docs', output_dir='python_pdfs', max_files=None):
    """
    Generate Python PDFs from existing Word documents.
    
    Args:
        docx_dir (str): Directory containing .docx files
        output_dir (str): Directory to save Python-generated PDFs
        max_files (int): Limit number of files to process
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get list of Word documents
    docx_files = [f for f in os.listdir(docx_dir) if f.endswith('.docx')]
    
    if max_files:
        docx_files = docx_files[:max_files]
    
    print(f"Generating Python PDFs for {len(docx_files)} documents...")
    
    successful = 0
    failed = 0
    
    for i, docx_file in enumerate(docx_files, 1):
        try:
            # Extract text from Word document
            docx_path = os.path.join(docx_dir, docx_file)
            text_content = extract_text_from_docx(docx_path)
            
            if text_content is None:
                print(f"[{i:3d}/{len(docx_files)}] FAILED to read {docx_file}")
                failed += 1
                continue
            
            # Generate PDF
            pdf_filename = docx_file.replace('.docx', '.pdf')
            pdf_path = os.path.join(output_dir, pdf_filename)
            
            title = docx_file.replace('.docx', '').replace('_', ' ')
            create_pdf_from_text(text_content, pdf_path, title)
            
            print(f"[{i:3d}/{len(docx_files)}] {docx_file} -> {pdf_filename}")
            successful += 1
            
        except Exception as e:
            print(f"[{i:3d}/{len(docx_files)}] ERROR processing {docx_file}: {e}")
            failed += 1
    
    print(f"\nPython PDF generation complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    return successful, failed

def main():
    """Main execution function."""
    print("Python PDF Generator for Forensic Research")
    print("=" * 50)
    
    # Check if ReportLab is installed
    try:
        import reportlab
        print(f"ReportLab version: {reportlab.Version}")
    except ImportError:
        print("Error: ReportLab not installed. Run: pip install reportlab")
        return
    
    # Check if python-docx is installed
    try:
        import docx
        print(f"python-docx available")
    except ImportError:
        print("Error: python-docx not installed. Run: pip install python-docx")
        return
    
    print()
    
    # Generate Python PDFs
    successful, failed = generate_python_pdfs(max_files=100)  # Start with subset
    
    if successful > 0:
        print(f"\nNext step: Convert Python PDFs to binary images")
        print("Run: python pdf_to_binary_image.py --python-pdfs")

if __name__ == "__main__":
    main()