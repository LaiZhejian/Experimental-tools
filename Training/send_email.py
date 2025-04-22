import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr
import socks
import socket
import os


def send_experiment_completion_email(source_name, subject, email_content):
    http_proxy = os.getenv("HTTP_PROXY")
    addr, port = http_proxy.split(":")
    socks.set_default_proxy(socks.SOCKS5, addr, int(port))
    socket.socket = socks.socksocket  # 替换 socket
    
    # ======== 配置部分 ========
    sender_email = ""
    sender_password = ""
    receiver_email = ""
    smtp_server = "smtp.qq.com"
    smtp_port = 465

    # ======== 创建邮件 ========
    message = MIMEText(email_content, 'plain', 'utf-8')
    message['From'] = formataddr((source_name, sender_email))
    message['To'] = formataddr(("收件人", receiver_email))
    message['Subject'] = Header(subject, 'utf-8')

    # ======== 发送邮件 ========
    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, [receiver_email], message.as_string())
            server.quit()
        print("邮件发送成功！")
    except Exception as e:
        print(f"邮件发送失败: {e}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="发送实验完成通知邮件")
#     parser.add_argument('--name', '-n', required=True, help='实验名称')
#     parser.add_argument('--params', '-p', default="无参数", help='实验参数描述（字符串）')
#     args = parser.parse_args()

#     send_experiment_completion_email(args.name, args.params)
