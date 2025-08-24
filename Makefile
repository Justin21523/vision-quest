# Makefile
# VisionQuest 開發工具

.PHONY: setup install dev test clean build docker

# 環境設置
setup:
	@echo "🔧 設置開發環境..."
	python scripts/setup_env.py

install:
	@echo "📦 安裝依賴..."
	python scripts/setup_env.py --install-deps

# 開發模式
dev:
	@echo "🚀 啟動開發服務..."
	python scripts/start_services.py

backend:
	@echo "🔥 僅啟動後端..."
	uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

gradio:
	@echo "📊 啟動 Gradio UI..."
	cd frontend/gradio_app && python app.py

react:
	@echo "⚛️  啟動 React UI..."
	cd frontend/react_app && npm run dev

pyqt:
	@echo "🖥️  啟動 PyQt 桌面應用..."
	cd frontend/pyqt_app && python app.py

# 測試
test:
	@echo "🧪 執行測試..."
	cd backend && python -m pytest tests/ -v

test-api:
	@echo "🔍 測試 API 端點..."
	cd backend && python -m pytest tests/test_api/ -v

lint:
	@echo "🔍 代碼檢查..."
	cd backend && ruff check . && mypy .

format:
	@echo "🎨 格式化代碼..."
	cd backend && black . && isort .

# 模型管理
download-models:
	@echo "📥 下載預設模型..."
	python scripts/download_models.py --all

download-model:
	@echo "📥 下載指定模型..."
	@read -p "請輸入模型名稱: " model; \
	python scripts/download_models.py --model $model

# 清理
clean:
	@echo "🧹 清理暫存檔案..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -rf backend/.coverage
	rm -rf frontend/react_app/dist
	rm -rf frontend/react_app/node_modules/.cache

# Docker
docker-build:
	@echo "🐳 建立 Docker 映像..."
	docker-compose build

docker-up:
	@echo "🚀 啟動 Docker 服務..."
	docker-compose up -d

docker-down:
	@echo "🛑 停止 Docker 服務..."
	docker-compose down

docker-logs:
	@echo "📋 查看 Docker 日誌..."
	docker-compose logs -f

# 說明
help:
	@echo "VisionQuest 開發工具"
	@echo ""
	@echo "環境設置:"
	@echo "  make setup          設置開發環境"
	@echo "  make install        安裝依賴"
	@echo ""
	@echo "開發模式:"
	@echo "  make dev           啟動所有服務"
	@echo "  make backend       僅啟動後端 API"
	@echo "  make gradio        啟動 Gradio UI"
	@echo "  make react         啟動 React UI"
	@echo "  make pyqt          啟動 PyQt 桌面應用"
	@echo ""
	@echo "測試與檢查:"
	@echo "  make test          執行所有測試"
	@echo "  make test-api      測試 API 端點"
	@echo "  make lint          代碼檢查"
	@echo "  make format        格式化代碼"
	@echo ""
	@echo "模型管理:"
	@echo "  make download-models   下載預設模型"
	@echo "  make download-model    下載指定模型"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  建立映像"
	@echo "  make docker-up     啟動容器"
	@echo "  make docker-down   停止容器"