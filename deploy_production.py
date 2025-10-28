#!/usr/bin/env python3
"""
Production Deployment Script
Automates the deployment of Blackhole Infiverse to production
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionDeployer:
    """Production deployment automation"""
    
    def __init__(self, project_dir: str = "."):
        self.project_dir = Path(project_dir)
        self.deployment_log = []
        
    def check_system_requirements(self) -> bool:
        """Check if system meets requirements"""
        print("=" * 60)
        print("CHECKING SYSTEM REQUIREMENTS")
        print("=" * 60)
        
        requirements = {
            'python': '3.8',
            'pip': True,
            'git': True,
            'disk_space': 100,  # GB
            'ram': 8,  # GB
        }
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            print(f"❌ Python version {python_version.major}.{python_version.minor} is too old")
            return False
        print(f"✅ Python version {python_version.major}.{python_version.minor} OK")
        
        # Check pip
        try:
            subprocess.run([sys.executable, '-m', 'pip', '--version'], check=True, capture_output=True)
            print("✅ pip available")
        except subprocess.CalledProcessError:
            print("❌ pip not available")
            return False
        
        # Check git
        try:
            subprocess.run(['git', '--version'], check=True, capture_output=True)
            print("✅ git available")
        except subprocess.CalledProcessError:
            print("❌ git not available")
            return False
        
        # Check disk space
        disk_usage = shutil.disk_usage(self.project_dir)
        free_gb = disk_usage.free / (1024**3)
        if free_gb < requirements['disk_space']:
            print(f"❌ Insufficient disk space: {free_gb:.1f}GB available, {requirements['disk_space']}GB required")
            return False
        print(f"✅ Disk space OK: {free_gb:.1f}GB available")
        
        return True
    
    def create_production_structure(self):
        """Create production directory structure"""
        print("\n" + "=" * 60)
        print("CREATING PRODUCTION STRUCTURE")
        print("=" * 60)
        
        production_dirs = [
            'logs',
            'data/cache',
            'data/features',
            'data/additional',
            'models',
            'config',
            'scripts',
            'backups'
        ]
        
        for dir_path in production_dirs:
            full_path = self.project_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_path}")
    
    def setup_environment(self):
        """Setup production environment"""
        print("\n" + "=" * 60)
        print("SETTING UP PRODUCTION ENVIRONMENT")
        print("=" * 60)
        
        # Create .env file if it doesn't exist
        env_file = self.project_dir / '.env'
        if not env_file.exists():
            env_content = """# Production Environment Variables
JWT_SECRET_KEY=your-secret-key-change-this
DATABASE_URL=postgresql://user:password@localhost:5432/blackhole_db
REDIS_URL=redis://localhost:6379
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
ENVIRONMENT=production
DEBUG=False
"""
            with open(env_file, 'w') as f:
                f.write(env_content)
            print("✅ Created .env file")
        else:
            print("✅ .env file already exists")
    
    def install_dependencies(self):
        """Install production dependencies"""
        print("\n" + "=" * 60)
        print("INSTALLING DEPENDENCIES")
        print("=" * 60)
        
        try:
            # Install Python dependencies
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ], check=True)
            print("✅ Python dependencies installed")
            
            # Install system dependencies (Ubuntu/Debian)
            system_deps = [
                'postgresql-client',
                'redis-tools',
                'curl',
                'htop'
            ]
            
            for dep in system_deps:
                try:
                    subprocess.run(['sudo', 'apt', 'install', '-y', dep], check=True)
                    print(f"✅ Installed {dep}")
                except subprocess.CalledProcessError:
                    print(f"⚠️  Could not install {dep} (may need manual installation)")
                    
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
        
        return True
    
    def setup_database(self):
        """Setup production database"""
        print("\n" + "=" * 60)
        print("SETTING UP DATABASE")
        print("=" * 60)
        
        try:
            # Create database
            subprocess.run([
                'sudo', '-u', 'postgres', 'createdb', 'blackhole_db'
            ], check=True)
            print("✅ Database created")
            
            # Create user (optional)
            subprocess.run([
                'sudo', '-u', 'postgres', 'psql', '-c',
                "CREATE USER blackhole_user WITH PASSWORD 'blackhole_pass';"
            ], check=True)
            print("✅ Database user created")
            
            # Grant permissions
            subprocess.run([
                'sudo', '-u', 'postgres', 'psql', '-c',
                "GRANT ALL PRIVILEGES ON DATABASE blackhole_db TO blackhole_user;"
            ], check=True)
            print("✅ Database permissions granted")
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Database setup failed: {e}")
            print("Please setup database manually")
    
    def create_systemd_services(self):
        """Create systemd services for production"""
        print("\n" + "=" * 60)
        print("CREATING SYSTEMD SERVICES")
        print("=" * 60)
        
        # API Server service
        api_service = f"""[Unit]
Description=Blackhole Infiverse API Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory={self.project_dir}
Environment=PATH={self.project_dir}/venv/bin
ExecStart={self.project_dir}/venv/bin/python api/server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        with open('/tmp/blackhole-api.service', 'w') as f:
            f.write(api_service)
        
        # LangGraph service
        langgraph_service = f"""[Unit]
Description=Blackhole Infiverse LangGraph Workflow
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory={self.project_dir}
Environment=PATH={self.project_dir}/venv/bin
ExecStart={self.project_dir}/venv/bin/python langgraph_workflow.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        with open('/tmp/blackhole-langgraph.service', 'w') as f:
            f.write(langgraph_service)
        
        print("✅ Systemd service files created")
        print("To install services:")
        print("sudo cp /tmp/blackhole-api.service /etc/systemd/system/")
        print("sudo cp /tmp/blackhole-langgraph.service /etc/systemd/system/")
        print("sudo systemctl daemon-reload")
        print("sudo systemctl enable blackhole-api blackhole-langgraph")
        print("sudo systemctl start blackhole-api blackhole-langgraph")
    
    def create_monitoring_scripts(self):
        """Create monitoring and maintenance scripts"""
        print("\n" + "=" * 60)
        print("CREATING MONITORING SCRIPTS")
        print("=" * 60)
        
        # Health check script
        health_check = """#!/bin/bash
# Health check script for Blackhole Infiverse

echo "=== BLACKHOLE INFIVERSE HEALTH CHECK ==="
echo "Date: $(date)"
echo ""

# Check API server
echo "1. Checking API Server..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API Server: OK"
else
    echo "❌ API Server: FAILED"
fi

# Check LangGraph workflow
echo "2. Checking LangGraph Workflow..."
if pgrep -f "langgraph_workflow.py" > /dev/null; then
    echo "✅ LangGraph Workflow: OK"
else
    echo "❌ LangGraph Workflow: FAILED"
fi

# Check system resources
echo "3. Checking System Resources..."
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory Usage: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
echo "Disk Usage: $(df -h / | awk 'NR==2{printf "%s", $5}')"

echo ""
echo "=== HEALTH CHECK COMPLETE ==="
"""
        
        with open('scripts/health_check.sh', 'w') as f:
            f.write(health_check)
        
        # Make executable
        os.chmod('scripts/health_check.sh', 0o755)
        print("✅ Health check script created")
        
        # Backup script
        backup_script = """#!/bin/bash
# Backup script for Blackhole Infiverse

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "=== CREATING BACKUP ==="
echo "Backup directory: $BACKUP_DIR"

# Backup database
echo "Backing up database..."
pg_dump blackhole_db > "$BACKUP_DIR/database.sql"

# Backup models
echo "Backing up models..."
cp -r models/ "$BACKUP_DIR/"

# Backup data
echo "Backing up data..."
cp -r data/ "$BACKUP_DIR/"

# Backup logs
echo "Backing up logs..."
cp -r logs/ "$BACKUP_DIR/"

echo "✅ Backup complete: $BACKUP_DIR"
"""
        
        with open('scripts/backup.sh', 'w') as f:
            f.write(backup_script)
        
        os.chmod('scripts/backup.sh', 0o755)
        print("✅ Backup script created")
    
    def create_documentation(self):
        """Create production documentation"""
        print("\n" + "=" * 60)
        print("CREATING PRODUCTION DOCUMENTATION")
        print("=" * 60)
        
        # Production README
        production_readme = """# Blackhole Infiverse - Production Deployment

## Quick Start

1. **Start Services**
   ```bash
   sudo systemctl start blackhole-api blackhole-langgraph
   ```

2. **Check Health**
   ```bash
   ./scripts/health_check.sh
   ```

3. **View Logs**
   ```bash
   sudo journalctl -u blackhole-api -f
   sudo journalctl -u blackhole-langgraph -f
   ```

## API Endpoints

- **Health**: `GET /health`
- **Predictions**: `POST /prediction_agent/tools/predict`
- **Feedback**: `POST /prediction_agent/tools/feedback`

## Monitoring

- **Health Check**: `./scripts/health_check.sh`
- **Backup**: `./scripts/backup.sh`
- **Logs**: `tail -f logs/api_server.log`

## Troubleshooting

1. **Service not starting**: Check logs with `journalctl -u service-name`
2. **Database connection**: Verify DATABASE_URL in .env
3. **Port conflicts**: Check if port 8000 is available

## Support

- **Logs**: Check `logs/` directory
- **Health**: Run health check script
- **Backup**: Regular backups in `backups/` directory
"""
        
        with open('PRODUCTION_README.md', 'w') as f:
            f.write(production_readme)
        
        print("✅ Production documentation created")
    
    def deploy(self):
        """Run complete deployment"""
        print("🚀 STARTING PRODUCTION DEPLOYMENT")
        print("=" * 60)
        
        # Check requirements
        if not self.check_system_requirements():
            print("❌ System requirements not met")
            return False
        
        # Create structure
        self.create_production_structure()
        
        # Setup environment
        self.setup_environment()
        
        # Install dependencies
        if not self.install_dependencies():
            print("❌ Dependency installation failed")
            return False
        
        # Setup database
        self.setup_database()
        
        # Create services
        self.create_systemd_services()
        
        # Create monitoring
        self.create_monitoring_scripts()
        
        # Create documentation
        self.create_documentation()
        
        print("\n" + "=" * 60)
        print("DEPLOYMENT COMPLETE")
        print("=" * 60)
        print("✅ Production deployment successful!")
        print("\nNext steps:")
        print("1. Configure .env file with production values")
        print("2. Install systemd services")
        print("3. Start services: sudo systemctl start blackhole-api blackhole-langgraph")
        print("4. Run health check: ./scripts/health_check.sh")
        print("=" * 60)
        
        return True


def main():
    """Main deployment function"""
    deployer = ProductionDeployer()
    deployer.deploy()


if __name__ == "__main__":
    main()
