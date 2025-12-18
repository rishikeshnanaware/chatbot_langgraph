# email_server.py


from fastmcp import FastMCP
import smtplib
import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
from typing import Optional  # ADD THIS


load_dotenv()


mcp = FastMCP("Email Server")


# Email configuration
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
IMAP_PORT = int(os.getenv("IMAP_PORT", "993"))




@mcp.tool()
def send_email(to: str, subject: str, body: str, cc: Optional[str] = None) -> dict:  # CHANGED: Added Optional
    """
    Send an email using SMTP.
   
    Args:
        to: Recipient email address
        subject: Email subject
        body: Email body text
        cc: Optional CC recipients (comma-separated). Leave empty if not needed.
   
    Returns:
        Dictionary with success status and message
    """
    # Validate credentials
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        return {
            "success": False,
            "error": "Email credentials not configured. Please set EMAIL_ADDRESS and EMAIL_APP_PASSWORD in .env file"
        }
   
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to
        msg['Subject'] = subject
       
        # Only add CC if it's provided and not None
        if cc:  # CHANGED: Added condition
            msg['Cc'] = cc
       
        msg.attach(MIMEText(body, 'plain'))
       
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
           
            recipients = [to]
            # Only add CC recipients if cc is not None
            if cc:  # CHANGED: Added condition
                recipients.extend([addr.strip() for addr in cc.split(',')])
           
            server.send_message(msg)
       
        return {"success": True, "message": f"Email sent successfully to {to}"}
   
    except smtplib.SMTPAuthenticationError as e:
        return {
            "success": False,
            "error": f"Authentication failed. Please check your EMAIL_APP_PASSWORD in .env file. Error: {str(e)}"
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to send email: {str(e)}"}




@mcp.tool()
def get_recent_emails(count: int = 10, folder: str = "INBOX") -> list:
    """
    Retrieve recent emails from inbox.
   
    Args:
        count: Number of recent emails to fetch (default: 10, max: 50)
        folder: Email folder to search (default: INBOX)
   
    Returns:
        List of email dictionaries with from, subject, date, and body preview
    """
    try:
        if count > 50:
            count = 50
           
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        mail.select(folder)
       
        _, messages = mail.search(None, 'ALL')
        email_ids = messages[0].split()
       
        if not email_ids:
            return [{"message": "No emails found"}]
       
        recent_emails = []
        # Get the most recent emails (reverse order)
        for email_id in reversed(email_ids[-count:]):
            _, msg_data = mail.fetch(email_id, '(RFC822)')
           
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                   
                    subject = msg['subject'] or "(No Subject)"
                    from_addr = msg['from']
                    date = msg['date']
                   
                    # Get email body
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                try:
                                    body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                                    break
                                except:
                                    body = "(Unable to decode body)"
                    else:
                        try:
                            body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                        except:
                            body = "(Unable to decode body)"
                   
                    # Preview first 300 characters
                    body_preview = body[:300] + "..." if len(body) > 300 else body
                   
                    recent_emails.append({
                        "from": from_addr,
                        "subject": subject,
                        "date": date,
                        "body_preview": body_preview.strip()
                    })
       
        mail.close()
        mail.logout()
       
        return recent_emails
   
    except Exception as e:
        return [{"error": f"Failed to fetch emails: {str(e)}"}]




@mcp.tool()
def search_emails(query: str, folder: str = "INBOX", max_results: int = 20) -> list:
    """
    Search emails by keyword in subject or sender.
   
    Args:
        query: Search query (searches in subject and from fields)
        folder: Email folder to search (default: INBOX)
        max_results: Maximum number of results (default: 20, max: 50)
   
    Returns:
        List of matching emails with from, subject, and date
    """
    try:
        if max_results > 50:
            max_results = 50
           
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        mail.select(folder)
       
        # Search by subject or from
        _, messages = mail.search(None, f'(OR SUBJECT "{query}" FROM "{query}")')
        email_ids = messages[0].split()
       
        if not email_ids:
            return [{"message": f"No emails found matching '{query}'"}]
       
        results = []
        for email_id in reversed(email_ids[-max_results:]):
            _, msg_data = mail.fetch(email_id, '(RFC822)')
           
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                   
                    results.append({
                        "from": msg['from'],
                        "subject": msg['subject'] or "(No Subject)",
                        "date": msg['date'],
                    })
       
        mail.close()
        mail.logout()
       
        return results
   
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]




@mcp.tool()
def get_unread_emails(count: int = 10) -> list:
    """
    Get unread emails from inbox.
   
    Args:
        count: Maximum number of unread emails to fetch (default: 10, max: 50)
   
    Returns:
        List of unread emails
    """
    try:
        if count > 50:
            count = 50
           
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        mail.select('INBOX')
       
        _, messages = mail.search(None, 'UNSEEN')
        email_ids = messages[0].split()
       
        if not email_ids:
            return [{"message": "No unread emails"}]
       
        unread_emails = []
        for email_id in reversed(email_ids[-count:]):
            _, msg_data = mail.fetch(email_id, '(RFC822)')
           
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                   
                    subject = msg['subject'] or "(No Subject)"
                    from_addr = msg['from']
                    date = msg['date']
                   
                    # Get email body preview
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                try:
                                    body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                                    break
                                except:
                                    body = "(Unable to decode)"
                    else:
                        try:
                            body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                        except:
                            body = "(Unable to decode)"
                   
                    body_preview = body[:200] + "..." if len(body) > 200 else body
                   
                    unread_emails.append({
                        "from": from_addr,
                        "subject": subject,
                        "date": date,
                        "body_preview": body_preview.strip()
                    })
       
        mail.close()
        mail.logout()
       
        return unread_emails
   
    except Exception as e:
        return [{"error": f"Failed to fetch unread emails: {str(e)}"}]




if __name__ == "__main__":
    mcp.run()
