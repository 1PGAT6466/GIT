# 宝利根知识库后端依赖检查脚本
# 运行：python3 check_deps.py

required_packages = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "sentence_transformers": "sentence-transformers",
    "psutil": "psutil",
    "docx": "python-docx",
    "openpyxl": "openpyxl",
    "pptx": "python-pptx",
    "numpy": "numpy",
    "faiss": "faiss-cpu",
    "jieba": "jieba",
}

missing = []
installed = []

for mod_name, pkg_name in required_packages.items():
    try:
        __import__(mod_name)
        installed.append(pkg_name)
        print(f"  OK  {pkg_name}")
    except ImportError:
        missing.append(pkg_name)
        print(f"  MISS  {pkg_name}")

print("\n" + "=" * 50)
if missing:
    print(f"缺少 {len(missing)} 个依赖:")
    for p in missing:
        print(f"  - {p}")
    print("\n安装命令:")
    print(f"pip install {' '.join(missing)} -i https://pypi.tuna.tsinghua.edu.cn/simple")
else:
    print(f"全部 {len(installed)} 个依赖已安装")
