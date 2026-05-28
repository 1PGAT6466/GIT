"""重启 kb-server 服务"""
import paramiko

HOST = "172.25.30.200"
USER = "feng-shaoxuan"
PWD = "123456"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PWD)

stdin, stdout, stderr = ssh.exec_command("echo '123456' | sudo -S systemctl restart kb-server")
stdin.write("123456\n")
stdin.flush()
out = stdout.read().decode()
err = stderr.read().decode()
exit_code = stdout.channel.recv_exit_status()
print(f"exit={exit_code}")
if out: print(f"out: {out}")
if err: print(f"err: {err}")

# 等一会再检查
import time
time.sleep(3)

stdin2, stdout2, stderr2 = ssh.exec_command("systemctl is-active kb-server && systemctl status kb-server --no-pager -l | head -8")
print(stdout2.read().decode())

ssh.close()
