from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

def create_pdf_report():
    filename = "Excellia_RAG_Progress_Report.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    
    # Create custom styles with unique names
    styles.add(ParagraphStyle(
        name='CustomTitle', 
        parent=styles['Heading1'], 
        fontSize=24, 
        textColor=colors.HexColor('#1a237e'), 
        alignment=TA_CENTER, 
        spaceAfter=30
    ))
    
    styles.add(ParagraphStyle(
        name='CustomHeading1', 
        parent=styles['Heading1'], 
        fontSize=18, 
        textColor=colors.HexColor('#0d47a1'), 
        spaceAfter=12, 
        spaceBefore=18
    ))
    
    styles.add(ParagraphStyle(
        name='CustomHeading2', 
        parent=styles['Heading2'], 
        fontSize=15, 
        textColor=colors.HexColor('#1565c0'), 
        spaceAfter=10, 
        spaceBefore=14
    ))
    
    styles.add(ParagraphStyle(
        name='CustomBody', 
        parent=styles['Normal'], 
        fontSize=10, 
        leading=14, 
        alignment=TA_JUSTIFY, 
        spaceAfter=8
    ))
    
    story = []
    
    # Title Page
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("Excellia RAG Project", styles['CustomTitle']))
    story.append(Paragraph("Progress Report", styles['CustomTitle']))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Model Testing & Performance Optimization", styles['CustomHeading1']))
    story.append(Paragraph("July 30, 2026", styles['CustomBody']))
    story.append(PageBreak())
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['CustomHeading1']))
    story.append(Paragraph(
        "This report summarizes the progress made on the Excellia RAG system, "
        "including model testing, performance optimization, and architectural decisions.",
        styles['CustomBody']
    ))
    story.append(Spacer(1, 0.3*inch))
    
    # Model Testing
    story.append(Paragraph("1. Model Testing & Results", styles['CustomHeading1']))
    
    model_data = [
        ['Model', 'Parameters', 'Response Time', 'Status', 'Verdict'],
        ['qwen2.5:1.5b-instruct', '7B', '>200 seconds', '❌ Too Heavy', 'Rejected'],
        ['Phi-3 Mini', '3.8B', '~100 seconds', '⚠️ Acceptable', 'Borderline'],
        ['Llama 3.2:3B', '3B', '~20 seconds', '✅ Good', 'Selected'],
    ]
    
    t = Table(model_data, colWidths=[1.8*inch, 1*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d47a1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e3f2fd')),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Key Optimizations", styles['CustomHeading2']))
    optimizations = [
        "• Reduced DENSE_TOP_K from 25 to 15 (40% reduction)",
        "• Reduced SPARSE_TOP_K from 25 to 15 (40% reduction)",
        "• Reduced FUSED_TOP_N from 15 to 8 (47% reduction)",
        "• Disabled reranking for speed (100% reduction)",
        "• Optimized confidence threshold from 0.1 to 0.005"
    ]
    for opt in optimizations:
        story.append(Paragraph(opt, styles['CustomBody']))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(PageBreak())
    
    # System Architecture
    story.append(Paragraph("2. System Architecture", styles['CustomHeading1']))
    
    architecture = [
        "1. User Question Input",
        "2. Embedding Generation (FastEmbed - BAAI/bge-small-en-v1.5)",
        "3. Dense Search (Qdrant - Top 15)",
        "4. Sparse Search (BM25 - Top 15)",
        "5. Hybrid Fusion (RRF - K=60, Top 8)",
        "6. Reranking (DISABLED for performance)",
        "7. Context Building (Top 8 chunks)",
        "8. LLM Generation (Ollama - llama3.2:3b)",
        "9. Answer Output"
    ]
    for step in architecture:
        story.append(Paragraph(f"• {step}", styles['CustomBody']))
    story.append(Spacer(1, 0.3*inch))
    
    # Architecture Flow
    story.append(Paragraph("Architecture Flow", styles['CustomHeading2']))
    flow = [
        "User Question → Embedding → Dense Search → Sparse Search → RRF Fusion → Context → LLM → Answer"
    ]
    for f in flow:
        story.append(Paragraph(f, styles['CustomBody']))
    story.append(Spacer(1, 0.3*inch))
    story.append(PageBreak())
    
    # Performance Metrics
    story.append(Paragraph("3. Performance Metrics", styles['CustomHeading1']))
    
    perf_data = [
        ['Metric', 'Value', 'Status'],
        ['Embedding Model', 'BAAI/bge-small-en-v1.5', 'CPU Optimized'],
        ['Vector Database', 'Qdrant', 'Operational'],
        ['LLM Model', 'llama3.2:3b', 'Selected'],
        ['Chunk Size', '300 tokens', 'Optimal'],
        ['Average Response Time', '~20 seconds', 'Good'],
        ['Chunks Indexed', '10,869', 'Successful'],
    ]
    
    t2 = Table(perf_data, colWidths=[2*inch, 2*inch, 2.5*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d47a1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e3f2fd')),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Model Comparison", styles['CustomHeading2']))
    comparison = [
        "• Qwen-2.5:7B: 200s response time, too heavy for CPU",
        "• Phi-3 Mini: 100s response time, borderline performance",
        "• Llama 3.2:3B: 20s response time, optimal choice"
    ]
    for comp in comparison:
        story.append(Paragraph(comp, styles['CustomBody']))
    story.append(Spacer(1, 0.3*inch))
    story.append(PageBreak())
    
    # Key Learnings
    story.append(Paragraph("4. Key Learnings", styles['CustomHeading1']))
    learnings = [
        "• Larger models (7B) are not suitable for CPU-only deployment",
        "• 3B-4B parameter models provide optimal balance",
        "• Response time improved from 200s to 20s (10x faster)",
        "• Disabling reranking significantly improves performance",
        "• Confidence threshold needs careful tuning per document type",
        "• FastEmbed provides efficient CPU embeddings without PyTorch",
        "• Chunk size and overlap parameters affect retrieval quality",
        "• RRF fusion effectively combines dense and sparse search results"
    ]
    for learning in learnings:
        story.append(Paragraph(f"• {learning}", styles['CustomBody']))
    story.append(Spacer(1, 0.3*inch))
    
    # Challenges & Solutions
    story.append(Paragraph("5. Challenges & Solutions", styles['CustomHeading1']))
    
    challenges = [
        ("Model Size", "7B models too large for CPU", "Switched to 3B models (llama3.2:3b)"),
        ("Response Time", "Initial response time >200s", "Reduced parameters and disabled reranking → 20s"),
        ("Confidence Threshold", "UPI documents found but not reliable", "Lowered threshold from 0.1 to 0.005"),
        ("Embedding Speed", "Slow embedding generation", "Used FastEmbed with progress tracking"),
        ("Memory Usage", "High memory consumption", "Optimized chunk sizes and top_k values")
    ]
    
    challenge_data = [['Challenge', 'Issue', 'Solution']]
    for c in challenges:
        challenge_data.append([c[0], c[1], c[2]])
    
    t3 = Table(challenge_data, colWidths=[1.5*inch, 2.5*inch, 3*inch])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d47a1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e3f2fd')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t3)
    story.append(Spacer(1, 0.3*inch))
    story.append(PageBreak())
    
    # Test Results
    story.append(Paragraph("6. Test Results Summary", styles['CustomHeading1']))
    
    test_data = [
        ['Query Type', 'Result', 'Status'],
        ['Excellia Trade', '✅ Found relevant information', 'High Confidence'],
        ['UPI (after threshold fix)', '✅ Found relevant information', 'Medium Confidence'],
        ['Documentary Credits', '✅ Found relevant information', 'High Confidence'],
        ['SWIFT Processing', '✅ Found relevant information', 'High Confidence'],
    ]
    
    t4 = Table(test_data, colWidths=[2*inch, 3*inch, 2*inch])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d47a1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e3f2fd')),
    ]))
    story.append(t4)
    story.append(Spacer(1, 0.3*inch))
    
    # Conclusion
    story.append(Paragraph("7. Conclusion", styles['CustomHeading1']))
    conclusion = [
        "The Excellia RAG system has successfully evolved through multiple iterations of model testing and optimization.",
        "",
        "Key Achievements:",
        "• Working RAG pipeline with hybrid search",
        "• Successful document ingestion (Excellia Trade + UPI documents)",
        "• FastEmbed for CPU-optimized embeddings",
        "• Response time reduced by 10x (200s → 20s)",
        "• Confidence threshold tuned for better retrieval",
        "",
        "The system is now ready for testing with real users and can be further optimized based on usage patterns."
    ]
    for c in conclusion:
        story.append(Paragraph(c, styles['CustomBody']))
    
    story.append(Spacer(1, 0.5*inch))
    
    # Footer
    story.append(Paragraph("Document Generated: July 30, 2026", styles['CustomBody']))
    story.append(Paragraph("Version: 1.0", styles['CustomBody']))
    story.append(Paragraph("Status: ✅ Active & Operational", styles['CustomBody']))
    
    # Build PDF
    doc.build(story)
    print(f"✅ PDF generated successfully: {filename}")
    return filename

if __name__ == "__main__":
    create_pdf_report()