#!/usr/bin/env python
"""
Generate Excellia Trade Technical Documentation PDF
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, 
    Paragraph, 
    Spacer, 
    Table, 
    TableStyle,
    PageBreak,
    KeepTogether,
    Image,
    ListFlowable,
    ListItem,
    HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import utils
import os
from datetime import datetime

# Custom styles
def get_styles():
    styles = getSampleStyleSheet()
    
    # Title style
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a237e'),
        alignment=TA_CENTER,
        spaceAfter=30,
        spaceBefore=30
    ))
    
    # Heading 1 style
    styles.add(ParagraphStyle(
        name='CustomHeading1',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#0d47a1'),
        spaceAfter=12,
        spaceBefore=18,
        alignment=TA_LEFT
    ))
    
    # Heading 2 style
    styles.add(ParagraphStyle(
        name='CustomHeading2',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1565c0'),
        spaceAfter=10,
        spaceBefore=14,
        alignment=TA_LEFT
    ))
    
    # Heading 3 style
    styles.add(ParagraphStyle(
        name='CustomHeading3',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.HexColor('#1e88e5'),
        spaceAfter=8,
        spaceBefore=12,
        alignment=TA_LEFT
    ))
    
    # Body text style
    styles.add(ParagraphStyle(
        name='CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=8
    ))
    
    # Code style
    styles.add(ParagraphStyle(
        name='CustomCode',
        parent=styles['Normal'],
        fontSize=8,
        fontName='Courier',
        textColor=colors.HexColor('#263238'),
        backColor=colors.HexColor('#f5f5f5'),
        leftIndent=20,
        spaceAfter=8,
        alignment=TA_LEFT
    ))
    
    return styles

def create_pdf():
    # File name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Excellia_Trade_Technical_Documentation_{timestamp}.pdf"
    
    # Create PDF
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )
    
    styles = get_styles()
    story = []
    
    # Helper functions
    def add_heading(text, level=1):
        if level == 1:
            style = styles['CustomHeading1']
        elif level == 2:
            style = styles['CustomHeading2']
        else:
            style = styles['CustomHeading3']
        story.append(Paragraph(text, style))
        story.append(Spacer(1, 0.1*inch))
    
    def add_body(text):
        story.append(Paragraph(text, styles['CustomBody']))
    
    def add_spacer(height=0.1):
        story.append(Spacer(1, height*inch))
    
    def add_code(text):
        story.append(Paragraph(text, styles['CustomCode']))
    
    def add_table(data, col_widths=None):
        if col_widths is None:
            col_widths = [2*inch, 3*inch]
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e3f2fd')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bbdefb')),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.2*inch))
    
    # Title Page
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("EXCELLIA TRADE", styles['CustomTitle']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Technical Documentation", styles['CustomTitle']))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Cloud-Native Fintech Solution", styles['CustomHeading2']))
    story.append(Spacer(1, 0.5*inch))
    
    # Document Info Table
    info_data = [
        ['Property', 'Value'],
        ['Document Title', 'Excellia Trade - Technical Architecture & Solution Overview'],
        ['Version', '1.0'],
        ['Date', '2026'],
        ['Classification', 'Internal - Confidential'],
        ['Department', 'Product Development & Engineering'],
    ]
    add_table(info_data, [2*inch, 4.5*inch])
    
    story.append(PageBreak())
    
    # Table of Contents
    add_heading("Table of Contents", 1)
    toc_items = [
        "1. Executive Summary",
        "2. Product Overview", 
        "3. Technical Architecture",
        "4. Cloud Infrastructure",
        "5. Core Modules & Functionality",
        "6. Fintech Capabilities",
        "7. Security & Compliance",
        "8. Integration & Deployment",
        "9. Performance & Scalability",
        "10. Technical Specifications",
        "11. API Documentation",
        "12. Glossary"
    ]
    for item in toc_items:
        add_body(f"• {item}")
    
    story.append(PageBreak())
    
    # 1. Executive Summary
    add_heading("1. Executive Summary", 1)
    add_heading("1.1 Purpose", 2)
    add_body("Excellia Trade is a comprehensive, cloud-native solution designed to manage and secure international banking operations. The platform provides end-to-end management of trade finance operations including documentary credits, documentary collections, international guarantees, financing, and SWIFT message processing.")
    
    add_heading("1.2 Key Value Propositions", 2)
    value_props = [
        "• Complete Functional Coverage: End-to-end trade finance operations management",
        "• Optimized User Experience: Three-tier distributed architecture",
        "• Flexible Integration: SaaS or on-premise deployment options",
        "• Cloud-First Design: Modern, scalable cloud infrastructure",
        "• Regulatory Compliance: Built-in compliance with international banking regulations"
    ]
    for prop in value_props:
        add_body(prop)
    
    add_heading("1.3 Target Audience", 2)
    audience = [
        "• International banks and financial institutions",
        "• Trade finance departments",
        "• Corporate treasury teams",
        "• Financial operations managers",
        "• Compliance officers"
    ]
    for item in audience:
        add_body(item)
    
    story.append(PageBreak())
    
    # 2. Product Overview
    add_heading("2. Product Overview", 1)
    add_heading("2.1 Core Capabilities", 2)
    
    capabilities_data = [
        ['Module', 'Description', 'Key Features'],
        ['Documentary Credits', 'Management of letters of credit', 'Issuance, amendments, advices, presentations, settlements'],
        ['Documentary Collections', 'Collection processing', 'Outward/inward collections, clean collections, documentary collections'],
        ['International Guarantees', 'Guarantee management', 'Issuance, amendments, claims handling, expiry management'],
        ['Financing', 'Trade finance operations', 'Pre-shipment/post-shipment financing, discounting, factoring'],
        ['SWIFT Processing', 'SWIFT message handling', 'MT/MX messages, ISO 20022 compliance, real-time tracking'],
    ]
    add_table(capabilities_data, [1.5*inch, 2*inch, 3*inch])
    
    story.append(PageBreak())
    
    # 3. Technical Architecture
    add_heading("3. Technical Architecture", 1)
    add_heading("3.1 Architectural Principles", 2)
    
    principles = [
        "• Microservices Architecture: Domain-driven design, containerized services, RESTful APIs, event-driven communication",
        "• Cloud-Native Design: Kubernetes orchestration, auto-scaling capabilities, service mesh integration, multi-cloud compatibility",
        "• Multi-Tier Structure: Client tier (UI/UX), Agency tier (branch operations), Head Office tier (central management)"
    ]
    for item in principles:
        add_body(item)
    
    add_heading("3.2 Technology Stack", 2)
    
    tech_data = [
        ['Layer', 'Technology', 'Purpose'],
        ['Frontend', 'React.js / Angular', 'User interface'],
        ['Backend', 'Java / Spring Boot / Python', 'Business logic'],
        ['Database', 'PostgreSQL / MongoDB', 'Data storage'],
        ['Messaging', 'Apache Kafka', 'Event streaming'],
        ['Cache', 'Redis', 'Performance optimization'],
        ['Search', 'Elasticsearch', 'Document retrieval'],
        ['Orchestration', 'Kubernetes', 'Container management'],
        ['API Gateway', 'Kong / NGINX', 'API management'],
        ['Monitoring', 'Prometheus / Grafana', 'Observability'],
    ]
    add_table(tech_data, [1.5*inch, 2*inch, 3*inch])
    
    story.append(PageBreak())
    
    # 4. Cloud Infrastructure
    add_heading("4. Cloud Infrastructure", 1)
    add_heading("4.1 Cloud Deployment Models", 2)
    
    deployment_data = [
        ['Model', 'Description', 'Best For'],
        ['SaaS', 'Fully managed cloud service', 'Rapid deployment, reduced operational overhead'],
        ['On-Premise', 'Self-hosted within organization', 'Security requirements, data sovereignty'],
        ['Hybrid', 'Combination of both', 'Phased migration, specific compliance needs'],
    ]
    add_table(deployment_data, [1.5*inch, 2.5*inch, 3*inch])
    
    add_heading("4.2 Cloud Providers Supported", 2)
    providers_data = [
        ['Provider', 'Services Used', 'Region Support'],
        ['AWS', 'EKS, RDS, S3, ElastiCache', 'Global'],
        ['Azure', 'AKS, Azure SQL, Blob Storage', 'Global'],
        ['GCP', 'GKE, Cloud SQL, Cloud Storage', 'Global'],
    ]
    add_table(providers_data, [1.5*inch, 2.5*inch, 3*inch])
    
    story.append(PageBreak())
    
    # 5. Core Modules
    add_heading("5. Core Modules & Functionality", 1)
    
    # 5.1 Documentary Credit
    add_heading("5.1 Documentary Credit Management", 2)
    add_body("Features:")
    features = [
        "• Credit issuance and amendments",
        "• LC types: revocable, irrevocable, confirmed",
        "• Presentation handling",
        "• Discrepancy management",
        "• Payment settlement"
    ]
    for feature in features:
        add_body(feature)
    add_body("Workflow: Application → Issuance → Advice → Presentation → Examination → Settlement")
    
    # 5.2 Documentary Collections
    add_heading("5.2 Documentary Collections", 2)
    features = [
        "• Clean collections",
        "• Documentary collections",
        "• Outward collections",
        "• Inward collections",
        "• Protest handling"
    ]
    for feature in features:
        add_body(feature)
    
    # 5.3 International Guarantees
    add_heading("5.3 International Guarantees", 2)
    features = [
        "• Guarantee issuance",
        "• Bid bonds",
        "• Performance guarantees",
        "• Advance payment guarantees",
        "• Claims management",
        "• Expiry monitoring"
    ]
    for feature in features:
        add_body(feature)
    
    story.append(PageBreak())
    
    # 6. Fintech Capabilities
    add_heading("6. Fintech Capabilities", 1)
    add_heading("6.1 Digital Transformation Features", 2)
    
    fintech_data = [
        ['Capability', 'Description', 'Benefit'],
        ['Digital Onboarding', 'Electronic KYC/AML verification', 'Faster client onboarding'],
        ['Smart Document Processing', 'AI-powered document extraction', 'Reduced manual errors'],
        ['Real-Time Analytics', 'Live dashboards and KPIs', 'Better decision making'],
        ['Mobile Accessibility', 'Mobile-responsive interface', 'Anywhere, anytime access'],
        ['API-First Design', 'Full API coverage', 'Easy integration'],
    ]
    add_table(fintech_data, [1.5*inch, 2.5*inch, 2.5*inch])
    
    add_heading("6.2 AI & Machine Learning Integration", 2)
    ai_capabilities = [
        "• Document Processing: OCR, Intelligent Document Recognition, Automated Data Extraction, Classification & Categorization",
        "• Risk Management: Fraud Detection, Anomaly Detection, Predictive Analytics",
        "• Operational Efficiency: Intelligent Automation, Smart Workflows, Pattern Recognition"
    ]
    for item in ai_capabilities:
        add_body(item)
    
    story.append(PageBreak())
    
    # 7. Security & Compliance
    add_heading("7. Security & Compliance", 1)
    add_heading("7.1 Security Architecture", 2)
    
    security_data = [
        ['Layer', 'Security Controls'],
        ['Network', 'VPC, firewalls, WAF, DDoS protection'],
        ['Application', 'JWT/OAuth2, RBAC, MFA, encryption'],
        ['Data', 'AES-256 encryption, TLS 1.3, data masking'],
        ['Infrastructure', 'Security patches, vulnerability scanning, SIEM'],
    ]
    add_table(security_data, [2*inch, 4.5*inch])
    
    add_heading("7.2 Compliance Standards", 2)
    compliance = [
        "• SOC 2: Security, availability, processing integrity, confidentiality, privacy",
        "• ISO 27001: Information security management",
        "• PCI DSS: Payment card industry data security (if applicable)",
        "• GDPR: Data protection and privacy",
        "• SWIFT CSP: Customer Security Programme compliance",
        "• KYC/AML: Know Your Customer / Anti-Money Laundering"
    ]
    for item in compliance:
        add_body(item)
    
    story.append(PageBreak())
    
    # 8. Integration & Deployment
    add_heading("8. Integration & Deployment", 1)
    add_heading("8.1 Integration Capabilities", 2)
    integration = [
        "• Banking Systems: Core banking integration",
        "• Legacy Systems: Mainframe and legacy system connectivity",
        "• Third-Party Services: External service integration",
        "• API Gateway: REST APIs, Webhooks, Event Streams"
    ]
    for item in integration:
        add_body(item)
    
    add_heading("8.2 Deployment Options", 2)
    deploy_data = [
        ['Option', 'Description', 'Time to Deploy'],
        ['SaaS', 'Fully managed cloud', '1-2 weeks'],
        ['On-Premise', 'Self-hosted', '4-8 weeks'],
        ['Hybrid', 'Mixed deployment', '3-6 weeks'],
    ]
    add_table(deploy_data, [1.5*inch, 2.5*inch, 2.5*inch])
    
    story.append(PageBreak())
    
    # 9. Performance & Scalability
    add_heading("9. Performance & Scalability", 1)
    add_heading("9.1 Performance Metrics", 2)
    
    perf_data = [
        ['Metric', 'Target', 'Description'],
        ['Response Time', '< 200ms', 'Average API response time'],
        ['Throughput', '10,000+ TPS', 'Transactions per second'],
        ['Availability', '99.99%', 'System uptime'],
        ['Recovery Time', '< 15 min', 'RTO (Recovery Time Objective)'],
        ['Data Consistency', '< 1 sec', 'Replication lag'],
    ]
    add_table(perf_data, [1.5*inch, 1.5*inch, 3.5*inch])
    
    add_heading("9.2 Scalability Features", 2)
    scalability = [
        "• Horizontal Scaling: Add more instances as needed",
        "• Vertical Scaling: Upgrade resources as needed",
        "• Auto-Scaling: Dynamic resource allocation",
        "• Sharding: Data distribution across nodes",
        "• Caching: In-memory data acceleration",
        "• CDN: Content delivery network"
    ]
    for item in scalability:
        add_body(item)
    
    story.append(PageBreak())
    
    # 10. Technical Specifications
    add_heading("10. Technical Specifications", 1)
    add_heading("10.1 Software Requirements", 2)
    
    sw_req_data = [
        ['Component', 'Version', 'Description'],
        ['Database', 'PostgreSQL 14+', 'Primary database'],
        ['Cache', 'Redis 6+', 'In-memory caching'],
        ['Message Queue', 'Apache Kafka 2.8+', 'Event streaming'],
        ['Search Engine', 'Elasticsearch 7.x', 'Document search'],
        ['Container Runtime', 'Docker 20.10+', 'Containerization'],
        ['Orchestration', 'Kubernetes 1.24+', 'Container orchestration'],
        ['API Gateway', 'Kong 2.8+', 'API management'],
    ]
    add_table(sw_req_data, [1.5*inch, 1.5*inch, 3.5*inch])
    
    add_heading("10.2 Hardware Requirements", 2)
    
    hw_data = [
        ['Environment', 'CPU', 'Memory', 'Storage', 'Network'],
        ['Development', '4 cores', '8 GB', '100 GB', '1 Gbps'],
        ['Staging', '8 cores', '16 GB', '500 GB', '1 Gbps'],
        ['Production', '16+ cores', '32+ GB', '1+ TB', '10 Gbps'],
    ]
    add_table(hw_data, [1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    
    story.append(PageBreak())
    
    # 11. API Documentation
    add_heading("11. API Documentation", 1)
    add_heading("11.1 Base URL", 2)
    add_code("https://api.excellia.com/v1/")
    
    add_heading("11.2 Authentication", 2)
    add_code("Authorization: Bearer <access_token>")
    
    add_heading("11.3 Core APIs", 2)
    
    api_data = [
        ['API', 'Method', 'Endpoint', 'Description'],
        ['Create LC', 'POST', '/credits', 'Create new letter of credit'],
        ['View LC', 'GET', '/credits/{id}', 'View LC details'],
        ['Amend LC', 'PUT', '/credits/{id}', 'Amend existing LC'],
        ['Create Collection', 'POST', '/collections', 'Create new collection'],
        ['Create Guarantee', 'POST', '/guarantees', 'Create new guarantee'],
        ['Process SWIFT', 'POST', '/swift/process', 'Process SWIFT message'],
        ['Search', 'GET', '/search', 'Search operations'],
        ['Reports', 'GET', '/reports/{type}', 'Generate reports'],
    ]
    add_table(api_data, [1.2*inch, 0.8*inch, 1.5*inch, 3*inch])
    
    story.append(PageBreak())
    
    # 12. Glossary
    add_heading("12. Glossary", 1)
    
    glossary_data = [
        ['Term', 'Definition'],
        ['Documentary Credit (LC)', 'A commitment by a bank to pay a seller on behalf of a buyer'],
        ['SWIFT', 'Society for Worldwide Interbank Financial Telecommunication'],
        ['ISO 20022', 'International standard for financial messaging'],
        ['Fintech', 'Financial technology'],
        ['SaaS', 'Software as a Service'],
        ['RAG', 'Retrieval-Augmented Generation'],
        ['KYC', 'Know Your Customer'],
        ['AML', 'Anti-Money Laundering'],
        ['MT Message', 'SWIFT message type (legacy)'],
        ['MX Message', 'ISO 20022 message format'],
        ['LC', 'Letter of Credit'],
        ['RTO', 'Recovery Time Objective'],
        ['RPO', 'Recovery Point Objective'],
        ['RBAC', 'Role-Based Access Control'],
        ['MFA', 'Multi-Factor Authentication'],
        ['TPS', 'Transactions Per Second'],
        ['CDN', 'Content Delivery Network'],
    ]
    add_table(glossary_data, [1.5*inch, 5*inch])
    
    # Footer
    story.append(PageBreak())
    story.append(Spacer(1, 2*inch))
    add_body("© 2026 Excellia RAG. All rights reserved.")
    add_body("This document contains proprietary information and is confidential.")
    add_body("Unauthorized reproduction or distribution is prohibited.")
    
    # Build PDF
    doc.build(story)
    print(f"✅ PDF generated successfully: {filename}")
    return filename

if __name__ == "__main__":
    # Install required package first:
    # pip install reportlab
    create_pdf()