import os

bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
workers = 1
threads = 2
timeout = 180
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info").lower()
