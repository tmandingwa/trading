# emailer.py
from __future__ import annotations

import smtplib
from email.message import EmailMessage
from typing import Optional

from config import (
    EMAIL_ALERTS_ENABLED,
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS,
    ALERT_TO_EMAIL, ALERT_FROM_EMAIL,
)

def send_email(subject: str, body: str) -> bool:
    if not EMAIL_ALERTS_ENABLED:
        return False

    if not SMTP_USER or not SMTP_PASS or not ALERT_TO_EMAIL:
        # missing config
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = ALERT_FROM_EMAIL or SMTP_USER
    msg["To"] = ALERT_TO_EMAIL
    msg.set_content(body)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as s:
            s.ehlo()
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        return True
    except Exception:
        return False
