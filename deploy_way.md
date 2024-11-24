# 项目部署文档

## 1. 引言

本部署文档旨在详细说明 **FitFormAI** 项目的部署流程和方法。该项目旨在为用户提供智能化的健身姿势纠正建议和广泛的社交平台功能，包括聊天和浏览帖子等。系统由前端、后端和算法模块组成，采用先进的深度学习技术（如 YOLOv8-Pose）进行人体姿势识别。

## 2. 目标

- **确保系统在生产环境中稳定、高效地运行**
- **提供清晰的部署步骤，方便维护和更新**
- **确保部署过程遵循国际标准和最佳实践**

## 3. 定义与缩略语

- **AI**：人工智能（Artificial Intelligence）
- **API**：应用程序编程接口（Application Programming Interface）
- **GPU**：图形处理单元（Graphics Processing Unit）
- **SSL**：安全套接字层（Secure Sockets Layer）
- **HTTPS**：超文本传输安全协议（Hypertext Transfer Protocol Secure）
- **CI/CD**：持续集成/持续部署（Continuous Integration/Continuous Deployment）

## 4. 参考资料

1. **FitFormAI 项目计划任务书**
2. **后端接口设计文档**
3. **前端设计文档**
4. **算法模块设计文档**
5. **软件工程开发规范**（中国信息通信研究院，2020）

## 5. 系统概述

**FitFormAI** 系统主要由以下三个组件组成：

- **前端应用程序**：提供用户界面，与用户交互。
- **后端服务**：处理业务逻辑、数据存储和与前端及算法模块的通信。
- **算法模块**：利用深度学习模型进行人体姿势识别和分析。

## 6. 部署环境

### 6.1 硬件要求

- **服务器数量**：至少 2 台
  - **前端服务器**：1 台
  - **后端和算法服务器（含 GPU）**：1 台
- **CPU**：多核处理器，推荐 ≥ 4 核心
- **内存**：至少 16GB
- **存储空间**：≥ 200GB，SSD 优先
- **网络带宽**：千兆位以太网

### 6.2 软件要求

- **操作系统**：Ubuntu 20.04 LTS
- **Docker**：用于容器化部署
- **Docker Compose**：管理多个容器
- **数据库**：MySQL 8.0
- **Redis**：用于缓存和消息队列
- **Python**：版本 3.8 或以上（后端和算法模块）
- **Flutter**：用于前端应用开发
- **Xcode**：用于编译和发布 iOS 应用
- **CUDA 和 cuDNN**：用于 GPU 加速（算法服务器）

## 7. 部署前准备

### 7.1 网络与安全

- **域名配置**：确定系统使用的域名，并在 DNS 中设置相应的解析。
- **防火墙设置**：开放必要的端口，如 80（HTTP）、443（HTTPS）、22（SSH）。
- **SSL 证书**：申请并配置 SSL 证书，确保数据传输的安全性。

### 7.2 配置管理

- **版本控制系统**：使用 Git 管理代码仓库，确保代码的版本一致性。
- **配置文件管理**：将敏感配置（如数据库密码、API 秘钥）使用环境变量或安全的配置管理工具管理。

## 8. 部署步骤

### 8.1 前端部署

#### 8.1.1 获取代码

```bash
git clone https://github.com/xjtuMaster/fitformai-frontend.git
cd fitformai-frontend
```

#### 8.1.2 安装依赖

```bash
flutter pub get
```

#### 8.1.3 构建 iOS 应用

```bash
flutter build ios
```

#### 8.1.4 部署到 iOS 设备

- 打开 Xcode，选择 `Runner.xcworkspace` 文件。
- 配置签名和团队信息。
- 选择目标设备，点击运行按钮进行部署。

### 8.2 后端和算法模块部署

#### 8.2.1 获取代码

```bash
git clone https://github.com/xjtuMaster/fitformai-backend.git
cd fitformai-backend
```

#### 8.2.2 环境设置

- 创建 Python 虚拟环境：
  ```bash
  python -m venv venv
  source venv/bin/activate
  ```
- 安装依赖：
  ```bash
  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```
- 配置环境变量，创建 `.env` 文件，包含数据库连接、Redis 配置、秘钥等信息。

#### 8.2.3 数据库和 Redis 设置

- 安装并配置数据库（MySQL）。
- 安装并配置 Redis。
- 运行数据库迁移：
  ```bash
  python manage.py migrate
  ```
- 创建超级用户：
  ```bash
  python manage.py createsuperuser
  ```

#### 8.2.4 启动服务

- 启动 Gunicorn 服务：
  ```bash
  gunicorn -w 4 -b 0.0.0.0:8000 run:app
  ```
- 启动 Redis 服务：
  ```bash
  redis-server
  ```
- 启动算法服务：
  ```bash
  python run_algorithm_service.py
  ```

### 8.3 容器化部署

- 使用 Docker 和 Docker Compose 进行容器化部署，统一管理各个服务。
- 编写 `docker-compose.yml` 文件，定义前端、后端、算法模块和数据库的服务。
- 运行部署：
  ```bash
  docker-compose up -d
  ```

### 8.4 示例 `docker-compose.yml` 文件

```yaml
version: '3.8'

services:
  frontend:
    image: your-repo/fitformai-frontend
    ports:
      - "80:80"
    volumes:
      - ./frontend:/usr/share/nginx/html
    restart: always

  backend:
    image: your-repo/fitformai-backend
    command: gunicorn -w 4 -b 0.0.0.0:8000 run:app
    volumes:
      - ./backend:/app
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/fitformai
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    ports:
      - "8000:8000"
    restart: always

  algorithm:
    image: your-repo/fitformai-algorithm
    command: python run_algorithm_service.py
    volumes:
      - ./algorithm:/app
    environment:
      - CUDA_VISIBLE_DEVICES=0
    depends_on:
      - backend
    restart: always

  db:
    image: postgres:13
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: fitformai
    volumes:
      - db_data:/var/lib/postgresql/data
    restart: always

  redis:
    image: redis:latest
    ports:
      - "6379:6379"
    restart: always

volumes:
  db_data:
```

## 9. 部署后验证

### 9.1 功能验证

- **前端**：在 iOS 设备上运行应用，验证页面加载正常，各功能可用。
- **后端 API**：使用工具（如Apifox）测试各 API 接口，验证返回结果正确。
- **算法模块**：调用算法服务接口，验证姿势识别和分析功能正常。

### 9.2 性能测试

- 使用压力测试工具模拟高并发请求，验证系统的性能和稳定性。

### 9.3 安全测试

- 检查 SSL 证书是否正确配置，网站是否通过 HTTPS 访问。
- 进行基础的安全测试，确保不存在常见的安全漏洞。

## 10. 回滚计划

如在部署后发现严重问题，需要回滚到上一个稳定版本。

- **备份数据**：确保在部署前已备份数据库和关键数据。
- **代码回滚**：通过 Git 回滚到上一个稳定的提交。
  ```bash
  git reset --hard [commit_hash]
  ```
- **重新部署**：按照部署步骤重新部署回滚后的代码。
- **验证**：进行部署后的验证，确保系统恢复正常。

## 11. 维护与支持

### 11.1 日志与监控

- **日志管理**：配置日志收集和管理工具分析系统日志。
- **性能监控**：使用监控工具监控系统性能指标。

### 11.2 定期更新

- **安全更新**：定期更新操作系统和依赖库的安全补丁。
- **功能更新**：按照版本计划，部署新的功能和改进。

### 11.3 技术支持

- **联系方式**：
  - **项目经理**：项目经理 薛志恒
  - **后端负责人**：后端负责人 林博涵
  - **算法负责人**：算法团队负责人 薛志恒
  - **前端负责人**：前端团队负责人 陈林涛

## 12. 附录

### 12.1 部署检查清单

- 服务器环境配置完成
- 防火墙和网络配置正确
- SSL 证书配置完成
- 前端代码构建并部署
- 后端服务启动并正常运行
- 算法模块服务启动并正常运行
- 数据库配置和迁移完成
- API 接口测试通过
- 前端与后端接口联调通过
- 性能和安全测试完成

### 12.2 常见问题与解决方案

- **问题**：前端页面无法加载或显示错误
  - **解决方案**：检查 Web 服务器配置和静态文件路径，确保前端构建正确

- **问题**：后端 API 返回 500 错误
  - **解决方案**：查看后端日志，定位具体错误，检查环境变量和依赖项

- **问题**：算法服务响应缓慢
  - **解决方案**：确认 GPU 是否正常工作，优化模型推理性能

---

**注意**：本部署文档遵循国际标准和最佳实践，确保系统在生产环境中安全、稳定、高效地运行。请在部署过程中严格按照步骤执行，任何疑问请及时与相关负责人联系。