# 宝利根企业智慧平台 — 项目总结

> v2.1 · 2026-05-27  
> 负责人：冯少轩 | 知识库 URL: `http://172.25.50.200:8080`

---

## 一、项目概述

宝利根企业智慧平台是一个**私有的企业级语义搜索与知识管理平台**，面向全公司员工，支持多格式文档上传、向量化存储、语义检索和 AI 智能问答。

---

## 二、核心架构

```
┌─────────────────────────────────────────────────┐
│              Browser (任何设备)                    │
│          http://172.25.50.200:8080               │
└─────────────────┬───────────────────────────────┘
                  │ HTTP
┌─────────────────▼───────────────────────────────┐
│           FastAPI (Python 3.10)                   │
│  ┌─────────────────────────────────────────┐    │
│  │ 路由层: /api/search /api/upload /api/...│    │
│  ├─────────────────────────────────────────┤    │
│  │ 文本提取: txt/pdf/docx/xlsx/pptx/dxf/   │    │
│  │           dwg/stl/obj/step...           │    │
│  ├─────────────────────────────────────────┤    │
│  │ 文本清洗: 脱敏、去HTML、去空行          │    │
│  ├─────────────────────────────────────────┤    │
│  │ 智能分段: 500字/段 + 句子边界            │    │
│  ├─────────────────────────────────────────┤    │
│  │ 向量化: Sentence-Transformers (384维)    │    │
│  ├─────────────────────────────────────────┤    │
│  │ 存储: Milvus Lite (嵌入式向量数据库)     │    │
│  └─────────────────────────────────────────┘    │
│             ▲ 每日凌晨2点自动备份                │
└─────────────────────────────────────────────────┘
```

### 技术栈

| 层级 | 技术 | 版本/说明 |
|------|------|----------|
| Web 框架 | FastAPI | 异步高性能 |
| 向量数据库 | Milvus Lite | 嵌入式，无需 Docker |
| 嵌入模型 | paraphrase-multilingual-MiniLM-L12-v2 | 384 维，中英文多语言 |
| PDF 解析 | pdftotext + Tesseract OCR | OCR 支持中英文 |
| DOCX 解析 | python-docx | |
| XLSX 解析 | openpyxl | |
| PPTX 解析 | python-pptx | |
| DXF 解析 | ezdxf | CAD 图纸 |
| DWG 解析 | strings (兜底) | LibreDWG 可选 |
| 3D 解析 | strings | STL/OBJ/STEP/IGES/GLTF |
| 部署 | systemd + Ubuntu | 开机自启 |
| AI 问答 | Mimo API (mimo-v2-omni) | 需配置 API Key |

---

## 三、文件结构

```
~/kb-server/
├── server.py              # 后端主程序 (≈900 行)
├── requirements.txt       # Python 依赖
├── static/
│   └── index.html         # 前端单页应用 (≈430 行)
├── data/
│   └── milvus.db          # 向量数据库 (~MB级)
├── uploads/               # 原始文件存档
│   └── YYYY-MM/
├── backups/               # 数据库备份 (保留7天)
│   └── milvus_YYYYMMDD.db
└── logs/                  # 搜索日志
    └── search_YYYYMMDD.jsonl
```

---

## 四、功能清单

### ✅ 已上线 (v2.1)

| # | 功能 | 说明 |
|---|------|------|
| 1 | **多格式上传** | 支持 28 种文件格式 |
| 2 | **语义搜索** | 向量相似度检索，Top-K 返回 |
| 3 | **AI 智能问答** | 基于知识库上下文 + Mimo 大模型回答 |
| 4 | **智能分类** | 自动识别文件类别（网络建设/IT资产/采购合同等） |
| 5 | **内容审核** | 自动检测并脱敏密码、手机号、身份证等敏感信息 |
| 6 | **去重上传** | SHA256 哈希查重，同文件不入库 |
| 7 | **上传进度条** | 实时百分比 + 批量上传 |
| 8 | **PDF + OCR** | pdftotext 提取文字 + Tesseract 中文 OCR 兜底 |
| 9 | **搜索历史** | 最近 30 条搜索记录 |
| 10 | **搜索高亮** | 关键词橙色标记 |
| 11 | **每日备份** | 凌晨 2 点自动备份，保留 7 天 |
| 12 | **重启不丢数据** | Milvus 持久化，服务重启数据完好 |
| 13 | **systemd 自启** | 开机自动启动 kb-server |
| 14 | **200MB 大文件** | 支持超大文件上传 |
| 15 | **CAD 图纸** | .dxf (ezdxf 完整解析) / .dwg (strings 提取) |
| 16 | **三维模型** | .stl/.obj/.step/.stp/.iges/.igs/.gltf/.glb |
| 17 | **文档删除** | DELETE /api/documents/{hash} |
| 18 | **文件列表** | GET /api/documents |
| 19 | **文件夹上传** | 拖拽或选择整个文件夹 |
| 20 | **常用工具面板** | 12 个工具快捷入口（URL 待配置） |
| 21 | **常见问题速查** | 15 个预设问题一键搜索 |

### 🔜 待开发

| # | 功能 | 优先级 |
|---|------|--------|
| A | CAD 图纸深度解析（图层、尺寸、块引用） | 高 |
| B | 3D 模型更丰富的元数据提取（体积、面数） | 中 |
| C | Embedding 模型可切换配置 | 中 |
| D | 域名 `polygon.com` 配置 | 中 |
| E | TOOL_URLS 真实地址填充 | 高 |
| F | 用户认证与权限管理 | 高 |
| G | 文档版本管理 | 低 |
| H | 删除确认二次弹窗 | 低 |
| I | 搜索历史本地清理逻辑 | 低 |

---

## 五、支持的 28 种文件格式

| 类别 | 格式 | 解析方式 |
|------|------|---------|
| 纯文本 | .txt .md .csv .cfg .log .ini .conf .json .xml .html .htm | 直接读取 |
| Word | .docx .doc | python-docx |
| Excel | .xlsx .xls | openpyxl |
| PDF | .pdf | pdftotext → Tesseract OCR |
| PPT | .pptx .ppt | python-pptx |
| CAD | .dxf | ezdxf (文字+图层+实体统计) |
| CAD | .dwg | dwgread / strings 兜底 |
| 3D 网格 | .stl .obj | strings 文本提取 |
| 3D 工程 | .step .stp .iges .igs | strings 文本提取 |
| 3D 场景 | .gltf .glb | strings 文本提取 |

---

## 六、API 端点一览

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/` | 前端首页 |
| GET | `/api/health` | 健康检查 |
| GET | `/api/stats` | 服务统计（文件数、片段数、版本、运行时间） |
| GET | `/api/search?q=&top_k=` | 语义搜索 |
| GET | `/api/documents` | 文档列表 |
| GET | `/api/search-history` | 搜索历史（30条） |
| POST | `/api/upload` | 一步上传入库 |
| POST | `/api/upload/preview` | 上传并预览分段 |
| POST | `/api/upload/confirm` | 确认入库 |
| DELETE | `/api/documents/{file_hash}` | 删除文档 |

---

## 七、安全保障

| 措施 | 实现 |
|------|------|
| 文件名安全 | `_sanitize_filename()` 防路径穿越 |
| 扩展名白名单 | 仅 28 种格式，拒绝其他 |
| 文件大小限制 | 200MB（可配置） |
| SHA256 去重 | 同文件不重复入库 |
| 敏感信息脱敏 | 密码/Token/身份证/手机号正则检测 |
| SQL/向量注入防护 | Pydantic 模型校验 + Milvus 参数化查询 |
| CORS | 允许所有来源（内网使用） |
| API 文档隐藏 | docs_url=None, redoc_url=None |

---

## 八、数据统计 (截至 2026-05-27)

| 指标 | 数值 |
|------|------|
| 已入库文件 | 19 |
| 向量片段 | 1,651 |
| 嵌入维度 | 384 |
| 数据库大小 | 约 8 MB |
| 版本 | v2.1 |
| 服务端口 | 8080 |
| 运行时长 | systemd 管理，持续运行 |

---

## 九、部署信息

| 项目 | 值 |
|------|-----|
| 虚拟机 IP | 172.25.50.200 |
| 本机管理 IP | 172.25.80.11 |
| 服务端口 | 8080 |
| systemd 服务 | kb-server.service |
| 部署路径 | /home/feng-shaoxuan/kb-server/ |
| 开发路径 | C:\Users\Feng Shaoxuan\.easyclaw\workspace\kb-server\ |
| 前端源文件 | E:\公司知识库\宝利根平台.html |

### 常用运维命令

```bash
sudo systemctl status kb-server     # 查看状态
sudo systemctl restart kb-server    # 重启
sudo journalctl -u kb-server -f     # 实时日志
```

---

## 十、部署同步方式

由于 EasyClaw 的 URL 过滤机制，文件传输使用以下方法：

```bash
# 1. 在本机启动 HTTP 中转服务器
# 2. 在虚拟机执行 Python 拉取
python3 -c "import urllib.request; urllib.request.urlretrieve('http://172.25.80.11:9998/server.py', '/home/feng-shaoxuan/kb-server/server.py'); print('done')"
python3 -c "import urllib.request; urllib.request.urlretrieve('http://172.25.80.11:9998/index.html', '/home/feng-shaoxuan/kb-server/static/index.html'); print('done')"
sudo systemctl restart kb-server
```

---

*本文档由系统自动生成，每次重大更新后刷新。*
