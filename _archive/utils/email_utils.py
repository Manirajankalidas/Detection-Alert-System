import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime


def build_email_html(cam_name, location, violation):
    from datetime import datetime
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""
    <div style="font-family: Arial, sans-serif; padding:10px; color:#333;">

        <h2 style="color:#2c3e50;">🚨 AI Safety Alert Notification</h2>

        <p>Dear Customer,</p>

        <p>
            Our AI-powered monitoring system has detected a 
            <b>{violation}</b> event in real-time.
            Please review the details below for immediate awareness.
        </p>

        <!-- ✅ KEEP THIS (INNER DESIGN ONLY) -->
        <div style="background:#f2f4f7; padding:12px; border-left:4px solid #007bff;">
            <b>Alert Details:</b><br><br>
            📍 <b>Location:</b> {location}<br>
            🎥 <b>Camera:</b> {cam_name}<br>
            ⚠️ <b>Event:</b> {violation}<br>
            🕒 <b>Detected At:</b> {time_str}
        </div>

        <p style="margin-top:12px;">
            📸 The detected image is attached for your reference.
        </p>

        <p>
            Our AI system continuously monitors your environment to enhance safety,
            reduce risks, and ensure rapid response.
        </p>

        <hr style="margin:15px 0;">

        <p style="font-size:14px; color:#555;">
            Need assistance? Contact us at 
            <a href="mailto:support@visioniq.ai">support@visioniq.ai</a>
        </p>

        <p>
            Regards,<br>
            <b>VisionIQ AI Monitoring System</b>
        </p>

    </div>
    """


def send_alert_email(to_email, subject, html_content, image_path):
    FROM_EMAIL = "quanticdemo@gmail.com"
    APP_PASSWORD = "jtgttfwdjrjriiap"

    msg = MIMEMultipart()
    msg['From'] = FROM_EMAIL
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(html_content, 'html'))

    if image_path:
        with open(image_path, 'rb') as f:
            img = MIMEImage(f.read())
            img.add_header('Content-Disposition', 'attachment', filename="alert.jpg")
            msg.attach(img)

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(FROM_EMAIL, APP_PASSWORD)
        server.send_message(msg)

    print("📧 Email Sent")