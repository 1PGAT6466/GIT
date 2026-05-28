"""
一键部署 kb-server 到 172.25.30.200
"""
import paramiko
import os
import sys

HOST = "172.25.30.200"
USER = "feng-shaoxuan"
PWD = "123456"
REMOTE_DIR = "/home/feng-shaoxuan/kb-server"

LOCAL_FILES = [
    "server.py",
    "static/index.html",
    "static/admin/index.html",
]

def deploy():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"[1/4] 连接 {USER}@{HOST}...")
    ssh.connect(HOST, username=USER, password=PWD)

    sftp = ssh.open_sftp()
    base = os.path.dirname(os.path.abspath(__file__))

    for f in LOCAL_FILES:
        local_path = os.path.join(base, f)
        remote_path = f"{REMOTE_DIR}/{f}"
        if not os.path.exists(local_path):
            print(f"  ⚠ 本地文件不存在: {local_path}")
            continue
        try:
            sftp.put(local_path, remote_path)
            print(f"  ✓ 上传: {f}")
        except Exception as e:
            print(f"  ✗ 上传失败 {f}: {e}")

    sftp.close()

    print(f"\n[2/4] 重启 kb-server 服务...")
    stdin, stdout, stderr = ssh.exec_command("sudo systemctl restart kb-server")
    out = stdout.read().decode()
    err = stderr.read().decode()
    if out: print(f"  stdout: {out}")
    if err: print(f"  stderr: {err}")

    print(f"\n[3/4] 验证端点...")
    import time
    time.sleep(2)
    endpoints = [
        "/api/health", "/api/stats", "/api/admin/stats",
        "/api/admin/server-status", "/api/admin/recent-activities",
        "/api/admin/ai-search-logs", "/", "/admin"
    ]
    import urllib.request
    all_ok = True
    for ep in endpoints:
        try:
            url = f"http://{HOST}:8080{ep}"
            req = urllib.request.Request(url)
            resp = urllib.request.urlopen(req, timeout=10)
            print(f"  ✓ {ep} → {resp.status}")
        except Exception as e:
            print(f"  ✗ {ep} → {str(e)[:60]}")
            all_ok = False

    if all_ok:
        print(f"\n[4/4] ✅ 部署完成，所有端点验证通过")
    else:
        print(f"\n[4/4] ⚠️ 部署完成，部分端点有问题")

    ssh.close()

if __name__ == "__main__":
    deploy()
