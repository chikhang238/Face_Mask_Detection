import os
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

def send_mail():
    sender_email = "YOUR-EMAIL"
    receiver_email = "ADMIN-EMAIL"
    password = "YOUR PASSWORD"

    message = MIMEMultipart("alternative")
    message["Subject"] = "[FACE MASK DETECTION] ALERT EMAIL"
    message["From"] = sender_email
    message["To"] = receiver_email

    # Create the plain-text and HTML version of your message
    text = """ALERT !!!
    - There are some people who do not wear face masks that is not safe for COVID-19 situation.
    """

    # Turn these into plain/html MIMEText objects
    part1 = MIMEText(text, "plain")
    fp = open('alert.jpg', 'rb')
    msgImage = MIMEImage(fp.read())
    os.remove('alert.jpg')
    fp.close()

    # Add HTML/plain-text parts to MIMEMultipart message
    # The email client will try to render the last part first
    message.attach(part1)
    message.attach(msgImage)

    # Create secure connection with server and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(
            sender_email, receiver_email, message.as_string()
        )
