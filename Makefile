.PHONY: help dev prod build test clean logs shell db-shell redis-shell minio-cli backup restore frontend frontend-dev

# Default target
help:
	@echo "Thinkerbell Docker Management"
	@echo "============================"
	@echo ""
	@echo "Available commands:"
	@echo "  dev          - Start development environment with hot-reload"
	@echo "  prod         - Start production environment"
	@echo "  build        - Build all Docker images"
	@echo "  test         - Run tests in container"
	@echo "  clean        - Stop containers and remove volumes"
	@echo "  logs         - Show logs from all services"
	@echo "  shell        - Open shell in API container"
	@echo "  db-shell     - Open PostgreSQL shell"
	@echo "  redis-shell  - Open Redis shell"
	@echo "  minio-cli    - Open MinIO client shell"
	@echo "  backup       - Backup all data"
	@echo "  restore      - Restore from backup"
	@echo "  health       - Check service health"
	@echo "  frontend     - Start frontend only (production)"
	@echo "  frontend-dev - Start frontend only (development)"
	@echo "  frontend-test- Run frontend tests"
	@echo ""

# Development environment
dev:
	@echo "🚀 Starting Thinkerbell development environment..."
	docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up

# Production environment
prod:
	@echo "🏭 Starting Thinkerbell production environment..."
	docker-compose up -d
	@echo "✅ Services started. Check status with: make logs"

# Build all images
build:
	@echo "🔨 Building Thinkerbell Docker images..."
	docker-compose build

# Build with no cache
build-fresh:
	@echo "🔨 Building Thinkerbell Docker images (no cache)..."
	docker-compose build --no-cache

# Run tests
test:
	@echo "🧪 Running tests..."
	docker-compose exec api pytest

# Run tests with coverage
test-cov:
	@echo "🧪 Running tests with coverage..."
	docker-compose exec api pytest --cov=app --cov-report=html

# Clean up
clean:
	@echo "🧹 Cleaning up containers and volumes..."
	docker-compose down -v
	docker system prune -f

# Show logs
logs:
	docker-compose logs -f

# Show logs for specific service
logs-api:
	docker-compose -f docker/docker-compose.yml logs -f api

logs-db:
	docker-compose -f docker/docker-compose.yml logs -f postgres

logs-worker:
	docker-compose -f docker/docker-compose.yml logs -f worker

# Shell access
shell:
	@echo "🐚 Opening shell in API container..."
	docker-compose -f docker/docker-compose.yml exec api bash

# Database shell
db-shell:
	@echo "🗄️ Opening PostgreSQL shell..."
	docker-compose -f docker/docker-compose.yml exec postgres psql -U thinkerbell_user -d thinkerbell

# Redis shell
redis-shell:
	@echo "📦 Opening Redis shell..."
	docker-compose -f docker/docker-compose.yml exec redis redis-cli

# MinIO client shell
minio-cli:
	@echo "☁️ Opening MinIO client shell..."
	docker-compose -f docker/docker-compose.yml exec minio mc config host add local http://localhost:9000 thinkerbell-admin thinkerbell-secret-2025

# Backup data
backup:
	@echo "💾 Creating backup..."
	mkdir -p backups
	docker-compose -f docker/docker-compose.yml exec postgres pg_dump -U thinkerbell_user thinkerbell > backups/db-backup-$(shell date +%Y%m%d-%H%M%S).sql
	@echo "✅ Database backup created in backups/"

# Health check
health:
	@echo "🏥 Checking service health..."
	@curl -s http://localhost:8000/health || echo "❌ API not responding"
	@docker-compose -f docker/docker-compose.yml ps

# Service status
status:
	@echo "📊 Service status:"
	docker-compose -f docker/docker-compose.yml ps

# Restart services
restart:
	@echo "🔄 Restarting services..."
	docker-compose -f docker/docker-compose.yml restart

# Restart specific service
restart-api:
	docker-compose -f docker/docker-compose.yml restart api

restart-worker:
	docker-compose -f docker/docker-compose.yml restart worker

# Scale workers
scale-workers:
	@echo "⚖️ Scaling workers to $(NUM) instances..."
	docker-compose -f docker/docker-compose.yml up -d --scale worker=$(NUM)

# Initialize database
init-db:
	@echo "🗄️ Initializing database..."
	docker-compose -f docker/docker-compose.yml exec api alembic upgrade head

# Create new migration
migration:
	@echo "📝 Creating new migration: $(MSG)"
	docker-compose -f docker/docker-compose.yml exec api alembic revision --autogenerate -m "$(MSG)"

# Setup environment (first-time setup)
setup:
	@echo "🎯 Setting up Thinkerbell environment..."
	@if [ ! -f .env ]; then \
		echo "📋 Creating .env file..."; \
		cp docker/env.example .env; \
		echo "✅ Please edit .env file with your configuration"; \
	else \
		echo "⚠️ .env file already exists"; \
	fi
	@echo "🔨 Building images..."
	$(MAKE) build
	@echo "🚀 Starting services..."
	$(MAKE) prod
	@echo "⏳ Waiting for services to be ready..."
	sleep 30
	@echo "🗄️ Initializing database..."
	$(MAKE) init-db
	@echo "✅ Setup complete! Visit http://localhost:8000"

# Quick development setup
dev-setup:
	@echo "🔧 Setting up development environment..."
	@if [ ! -f .env ]; then \
		cp docker/env.example .env; \
		sed -i 's/ENVIRONMENT=production/ENVIRONMENT=development/' .env; \
		sed -i 's/DEBUG=false/DEBUG=true/' .env; \
	fi
	$(MAKE) build
	@echo "✅ Development setup complete! Run 'make dev' to start"

# Stop all services
stop:
	@echo "🛑 Stopping all services..."
	docker-compose -f docker/docker-compose.yml down

# View resource usage
stats:
	@echo "📈 Resource usage:"
	docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

# Frontend commands
frontend:
	@echo "🎨 Starting frontend (production)..."
	cd thinkerbell && docker build -t thinkerbell-frontend . && docker run -p 3000:3000 thinkerbell-frontend

frontend-dev:
	@echo "🎨 Starting frontend (development)..."
	cd thinkerbell && npm run dev

frontend-test:
	@echo "🧪 Running frontend tests..."
	cd thinkerbell && npm run test

frontend-build:
	@echo "🔨 Building frontend..."
	cd thinkerbell && npm run build

frontend-lint:
	@echo "🔍 Linting frontend code..."
	cd thinkerbell && npm run lint

frontend-shell:
	@echo "🐚 Opening shell in frontend container..."
	docker-compose -f docker/docker-compose.yml exec frontend sh

logs-frontend:
	docker-compose -f docker/docker-compose.yml logs -f frontend
