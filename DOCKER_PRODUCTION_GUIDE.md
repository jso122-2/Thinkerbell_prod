# Thinkerbell Production Docker Guide

## Overview

This guide covers the complete dockerized production setup for Thinkerbell, including both the React frontend and Python backend with all supporting services.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Nginx       │    │   React App     │    │   Python API    │
│  (Port 80/443)  │────│   (Port 3000)   │────│   (Port 8000)   │
│   Load Balancer │    │   Frontend      │    │    Backend      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                                              │
         │              ┌─────────────────┐            │
         │              │   PostgreSQL    │            │
         │              │   (Port 5432)   │            │
         │              │    Database     │            │
         │              └─────────────────┘            │
         │                                              │
         │              ┌─────────────────┐            │
         │              │     Redis       │            │
         │              │   (Port 6379)   │────────────┘
         │              │     Cache       │
         │              └─────────────────┘
         │
         │              ┌─────────────────┐
         │              │     MinIO       │
         └──────────────│   (Port 9000)   │
                        │ Object Storage  │
                        └─────────────────┘
```

## Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Make (optional, for convenience commands)

### 1. Clone and Setup

```bash
git clone <repository>
cd Thinkerbell
```

### 2. Environment Configuration

```bash
# Copy and edit environment file
cp docker/env.example .env

# Edit .env with your production values
# Important: Change all default passwords!
```

### 3. Production Deployment

```bash
# Option 1: Using Make (recommended)
make setup

# Option 2: Manual commands
docker-compose -f docker/docker-compose.yml up -d
```

### 4. Verify Deployment

```bash
# Check service health
make health

# View logs
make logs

# Check individual services
docker-compose -f docker/docker-compose.yml ps
```

Access the application:
- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost/docs
- **MinIO Console**: http://localhost:9001

## Development Setup

### Quick Development Start

```bash
# Setup development environment
make dev-setup

# Start development servers with hot-reload
make dev
```

This starts:
- React frontend with hot-reload (port 3000)
- Python API with hot-reload (port 8000)
- All supporting services
- Development debugging tools

### Frontend Development

```bash
# Frontend-only commands
make frontend-dev    # Start dev server
make frontend-test   # Run tests
make frontend-lint   # Lint code
make frontend-build  # Build for production
```

### Backend Development

```bash
# Backend-only commands
make shell          # API container shell
make db-shell      # PostgreSQL shell
make redis-shell   # Redis shell
make logs-api      # API logs only
```

## Production Configuration

### Environment Variables

Key production environment variables in `.env`:

```bash
# Security
POSTGRES_PASSWORD=your-secure-password
MINIO_SECRET_KEY=your-minio-secret-key

# API Configuration
ENVIRONMENT=production
DEBUG=false

# Database
DATABASE_URL=postgresql://thinkerbell_user:password@postgres:5432/thinkerbell

# Redis
REDIS_URL=redis://redis:6379

# MinIO Object Storage
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=thinkerbell-admin
MINIO_SECRET_KEY=your-secret-key
MINIO_BUCKET=thinkerbell-models
```

### SSL/HTTPS Setup

1. Generate SSL certificates:
```bash
mkdir -p docker/ssl
# Place your SSL certificates:
# docker/ssl/thinkerbell.crt
# docker/ssl/thinkerbell.key
```

2. Enable HTTPS in `docker/nginx.conf`:
```bash
# Uncomment the HTTPS server block
# Update certificate paths
# Enable HTTP to HTTPS redirect
```

### Security Hardening

1. **Change Default Passwords**: Update all default passwords in `.env`
2. **Firewall**: Configure firewall to only allow necessary ports
3. **SSL**: Enable HTTPS in production
4. **Secrets Management**: Use Docker secrets or external secret managers
5. **Container Security**: Non-root users are already configured

## Service Management

### Common Commands

```bash
# Service control
make prod           # Start production
make dev            # Start development
make stop           # Stop all services
make restart        # Restart all services
make clean          # Clean up containers and volumes

# Monitoring
make health         # Check service health
make logs           # View all logs
make stats          # Resource usage
make status         # Service status

# Database
make backup         # Backup database
make init-db        # Initialize database
make migration      # Create migration
```

### Scaling Services

```bash
# Scale workers
make scale-workers NUM=3

# Scale manually
docker-compose -f docker/docker-compose.yml up -d --scale worker=3
```

## Monitoring and Logging

### Health Checks

All services include health checks:
- **API**: `GET /health`
- **Frontend**: `GET /health`
- **PostgreSQL**: `pg_isready`
- **Redis**: `redis-cli ping`
- **MinIO**: Internal health endpoint

### Log Management

```bash
# View logs
make logs                 # All services
make logs-api            # API only
make logs-frontend       # Frontend only
make logs-db             # Database only

# Log rotation is configured for production
```

### Monitoring Tools

Optional monitoring services:
- **Flower**: Celery task monitoring (port 5555)
- **Prometheus**: Metrics collection (add custom config)
- **Grafana**: Metrics visualization (add custom config)

## Backup and Recovery

### Database Backup

```bash
# Create backup
make backup

# Manual backup
docker-compose exec postgres pg_dump -U thinkerbell_user thinkerbell > backup.sql

# Restore from backup
docker-compose exec -T postgres psql -U thinkerbell_user -d thinkerbell < backup.sql
```

### Complete System Backup

```bash
# Stop services
make stop

# Backup volumes
docker run --rm -v postgres_data:/data -v $(pwd)/backups:/backup alpine tar czf /backup/postgres-$(date +%Y%m%d).tar.gz -C /data .
docker run --rm -v redis_data:/data -v $(pwd)/backups:/backup alpine tar czf /backup/redis-$(date +%Y%m%d).tar.gz -C /data .
docker run --rm -v minio_data:/data -v $(pwd)/backups:/backup alpine tar czf /backup/minio-$(date +%Y%m%d).tar.gz -C /data .

# Restart services
make prod
```

## Performance Optimization

### Frontend Optimizations

- **Code Splitting**: Implemented in Vite config
- **Gzip Compression**: Enabled in Nginx
- **Static Asset Caching**: 1-year cache for static files
- **Bundle Analysis**: Use `npm run build` to analyze bundle size

### Backend Optimizations

- **Connection Pooling**: Configured for PostgreSQL
- **Redis Caching**: Implemented for API responses
- **Background Tasks**: Celery workers for async processing
- **Resource Limits**: Set appropriate memory/CPU limits

### Infrastructure Optimizations

- **Nginx Caching**: Static file caching configured
- **Rate Limiting**: API rate limiting enabled
- **Health Checks**: Prevent traffic to unhealthy containers
- **Multi-stage Builds**: Optimized Docker images

## Troubleshooting

### Common Issues

1. **Services won't start**:
   ```bash
   # Check logs
   make logs
   # Check disk space
   df -h
   # Check Docker daemon
   docker system info
   ```

2. **Database connection issues**:
   ```bash
   # Check database status
   make db-shell
   # Verify connection string in .env
   # Check network connectivity
   docker network ls
   ```

3. **Frontend build failures**:
   ```bash
   # Clear node_modules
   cd thinkerbell && rm -rf node_modules && npm install
   # Check for TypeScript errors
   cd thinkerbell && npm run type-check
   ```

4. **Memory issues**:
   ```bash
   # Check resource usage
   make stats
   # Increase Docker memory limits
   # Scale down non-essential services
   ```

### Log Locations

- **Application Logs**: `logs/` directory
- **Container Logs**: `docker-compose logs [service]`
- **Nginx Logs**: Inside nginx container at `/var/log/nginx/`

## Maintenance

### Regular Tasks

1. **Update Dependencies**: Monthly security updates
2. **Log Rotation**: Logs are rotated automatically
3. **Database Maintenance**: Weekly `VACUUM` and `ANALYZE`
4. **Backup Verification**: Test backup restoration monthly
5. **Security Updates**: Keep base images updated

### Updates and Migrations

```bash
# Pull latest images
docker-compose pull

# Apply database migrations
make init-db

# Rolling update (zero downtime)
docker-compose up -d --no-deps api
docker-compose up -d --no-deps worker
```

## Support and Documentation

- **API Documentation**: Available at `/docs` when running
- **Architecture Diagrams**: See `Blueprints/` directory
- **Configuration Examples**: See `docker/` directory
- **Troubleshooting**: Check logs with `make logs`

For additional support, check the project documentation or create an issue in the repository.




