import os
from dotenv import load_dotenv

load_dotenv("mail.env")

print("🔧 Checking Email Configuration:")
print(f"MAIL_USERNAME: {os.getenv('MAIL_USERNAME')}")
print(f"MAIL_PASSWORD: {'*' * len(os.getenv('MAIL_PASSWORD')) if os.getenv('MAIL_PASSWORD') else 'NOT SET'}")
print(f"MAIL_SERVER: {os.getenv('MAIL_SERVER')}")
print(f"MAIL_PORT: {os.getenv('MAIL_PORT')}")