"""
Email Agent Tool for NL2Q Analyst
Enables the agent to send analysis results via email

Features:
- Send analysis results with charts and tables
- Support for multiple recipients
- HTML formatted emails with embedded visualizations
- Attachment support for CSV exports
- Integration with Dynamic Agent Orchestrator
"""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.mime.image import MIMEImage
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import base64
import io

# Import for converting matplotlib/plotly charts to images
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available - chart embedding will be limited")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("Pandas not available - CSV export will be limited")

logger = logging.getLogger(__name__)


class EmailAgent:
    """
    Agent tool for sending analysis results via email
    """
    
    def __init__(self, 
                 smtp_server: str = None,
                 smtp_port: int = None,
                 sender_email: str = None,
                 sender_password: str = None):
        """
        Initialize Email Agent
        
        Args:
            smtp_server: SMTP server address (e.g., 'smtp.gmail.com')
            smtp_port: SMTP port (587 for TLS, 465 for SSL)
            sender_email: Sender email address
            sender_password: Sender email password or app password
        """
        # Load from environment variables if not provided
        self.smtp_server = smtp_server or os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = smtp_port or int(os.getenv('SMTP_PORT', '587'))
        self.sender_email = sender_email or os.getenv('SENDER_EMAIL')
        self.sender_password = sender_password or os.getenv('SENDER_PASSWORD')
        
        # Validate configuration
        if not self.sender_email or not self.sender_password:
            logger.warning("Email credentials not configured. Email sending will fail.")
            logger.info("Set SENDER_EMAIL and SENDER_PASSWORD environment variables")
    
    def send_analysis_email(self,
                           recipients: List[str],
                           subject: str,
                           analysis_summary: str,
                           data: Optional[List[Dict]] = None,
                           charts: Optional[List[Dict]] = None,
                           sql_query: Optional[str] = None,
                           cc: Optional[List[str]] = None,
                           bcc: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Send analysis results via email
        
        Args:
            recipients: List of recipient email addresses
            subject: Email subject line
            analysis_summary: Text summary of the analysis
            data: Query results data (list of dicts)
            charts: Chart configurations/data
            sql_query: SQL query that was executed
            cc: CC recipients
            bcc: BCC recipients
            
        Returns:
            Dict with status and details
        """
        try:
            # Validate configuration
            if not self.sender_email or not self.sender_password:
                return {
                    "status": "error",
                    "error": "Email credentials not configured. Please set SENDER_EMAIL and SENDER_PASSWORD environment variables."
                }
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            if cc:
                msg['Cc'] = ', '.join(cc)
            
            # Generate email body
            html_body = self._generate_html_email(
                analysis_summary=analysis_summary,
                data=data,
                charts=charts,
                sql_query=sql_query
            )
            
            # Attach HTML body
            html_part = MIMEText(html_body, 'html')
            msg.attach(html_part)
            
            # Attach CSV if data is provided
            if data and PANDAS_AVAILABLE:
                csv_attachment = self._create_csv_attachment(data)
                if csv_attachment:
                    msg.attach(csv_attachment)
            
            # Send email
            all_recipients = recipients + (cc or []) + (bcc or [])
            
            # Try SMTP with STARTTLS (port 587) first
            try:
                with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
                    server.starttls()
                    server.login(self.sender_email, self.sender_password)
                    server.send_message(msg)
            except (smtplib.SMTPException, TimeoutError, OSError) as smtp_error:
                # If port 587 fails, try SSL on port 465
                logger.warning(f"Port 587 failed ({smtp_error}), trying SSL on port 465...")
                with smtplib.SMTP_SSL(self.smtp_server, 465, timeout=10) as server:
                    server.login(self.sender_email, self.sender_password)
                    server.send_message(msg)
            
            logger.info(f"Email sent successfully to {len(recipients)} recipient(s)")
            
            return {
                "status": "success",
                "message": f"Analysis email sent to {', '.join(recipients)}",
                "recipients": recipients,
                "subject": subject,
                "timestamp": datetime.now().isoformat()
            }
            
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP Authentication failed: {e}")
            return {
                "status": "error",
                "error": "Email authentication failed. Please check your email credentials.",
                "details": str(e),
                "recipients": recipients
            }
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error: {e}")
            return {
                "status": "error",
                "error": "Failed to send email due to SMTP error.",
                "details": str(e),
                "recipients": recipients
            }
        except Exception as e:
            logger.error(f"Error sending email: {e}", exc_info=True)
            return {
                "status": "error",
                "error": f"Failed to send email: {str(e)}",
                "recipients": recipients
            }
    
    def _generate_html_email(self,
                            analysis_summary: str,
                            data: Optional[List[Dict]] = None,
                            charts: Optional[List[Dict]] = None,
                            sql_query: Optional[str] = None) -> str:
        """
        Generate HTML formatted email body
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 28px;
                }}
                .header p {{
                    margin: 10px 0 0 0;
                    opacity: 0.9;
                }}
                .section {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    border-left: 4px solid #667eea;
                }}
                .section h2 {{
                    margin-top: 0;
                    color: #667eea;
                    font-size: 20px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 15px;
                    background: white;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}
                th {{
                    background: #667eea;
                    color: white;
                    padding: 12px;
                    text-align: left;
                    font-weight: 600;
                }}
                td {{
                    padding: 10px 12px;
                    border-bottom: 1px solid #e5e7eb;
                }}
                tr:hover {{
                    background: #f8f9fa;
                }}
                .sql-code {{
                    background: #1e1e1e;
                    color: #d4d4d4;
                    padding: 15px;
                    border-radius: 5px;
                    font-family: 'Courier New', monospace;
                    font-size: 13px;
                    overflow-x: auto;
                    white-space: pre-wrap;
                }}
                .footer {{
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 2px solid #e5e7eb;
                    text-align: center;
                    color: #6b7280;
                    font-size: 12px;
                }}
                .badge {{
                    display: inline-block;
                    padding: 4px 12px;
                    background: #10b981;
                    color: white;
                    border-radius: 12px;
                    font-size: 12px;
                    font-weight: 600;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 NL2Q Analysis Report</h1>
                <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
            
            <div class="section">
                <h2>📝 Analysis Summary</h2>
                <p>{analysis_summary}</p>
            </div>
        """
        
        # Add data table if provided
        if data and len(data) > 0:
            # Limit to first 50 rows for email
            display_data = data[:50]
            total_rows = len(data)
            
            html += f"""
            <div class="section">
                <h2>📋 Query Results</h2>
                <p>
                    <span class="badge">Total: {total_rows} rows</span>
                    {f'<span style="color: #6b7280; font-size: 13px; margin-left: 10px;">(Showing first 50 rows)</span>' if total_rows > 50 else ''}
                </p>
                <table>
                    <thead>
                        <tr>
                            {''.join(f'<th>{col}</th>' for col in display_data[0].keys())}
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(
                            '<tr>' + ''.join(f'<td>{str(val) if val is not None else "-"}</td>' for val in row.values()) + '</tr>'
                            for row in display_data
                        )}
                    </tbody>
                </table>
            </div>
            """
        
        # Add SQL query if provided
        if sql_query:
            html += f"""
            <div class="section">
                <h2>🔍 SQL Query</h2>
                <div class="sql-code">{sql_query}</div>
            </div>
            """
        
        # Add footer
        html += f"""
            <div class="footer">
                <p>
                    <strong>NL2Q Analyst</strong> - Natural Language to Query AI Assistant<br>
                    This is an automated email. Results are generated by AI and should be verified.
                </p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_csv_attachment(self, data: List[Dict]) -> Optional[MIMEBase]:
        """
        Create CSV attachment from data
        """
        try:
            if not PANDAS_AVAILABLE:
                return None
            
            # Convert to DataFrame and then CSV
            df = pd.DataFrame(data)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            # Create attachment
            attachment = MIMEBase('text', 'csv')
            attachment.set_payload(csv_data.encode('utf-8'))
            encoders.encode_base64(attachment)
            
            filename = f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            attachment.add_header('Content-Disposition', f'attachment; filename={filename}')
            
            return attachment
            
        except Exception as e:
            logger.error(f"Error creating CSV attachment: {e}")
            return None


# Example usage and configuration
if __name__ == "__main__":
    # Example: Send a test email
    email_agent = EmailAgent()
    
    sample_data = [
        {"Territory": "North", "Sales": 150000, "Target": 200000},
        {"Territory": "South", "Sales": 180000, "Target": 200000},
        {"Territory": "East", "Sales": 220000, "Target": 200000},
    ]
    
    result = email_agent.send_analysis_email(
        recipients=["recipient@example.com"],
        subject="Q4 Sales Analysis Report",
        analysis_summary="Sales performance analysis for Q4 2024 showing East region exceeding targets.",
        data=sample_data,
        sql_query="SELECT Territory, SUM(Sales) as Sales, Target FROM sales_data GROUP BY Territory"
    )
    
    print(json.dumps(result, indent=2))
