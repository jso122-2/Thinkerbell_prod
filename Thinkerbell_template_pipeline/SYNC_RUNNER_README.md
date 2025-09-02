# Thinkerbell File Sync Runner

This system automatically syncs files from the Windows environment to your Ubuntu environment, ensuring all the latest files are available in your Ubuntu shell.

## 🚀 **Quick Start**

### **Step 1: Copy the Sync Runner to Ubuntu**

First, copy the sync runner script to your Ubuntu environment:

```bash
# Copy the sync runner to your Ubuntu environment
cp /mnt/c/Users/Admin/OneDrive/Desktop/scripts/Thinkerbell/Thinkerbell_template_pipeline/ubuntu_sync_runner.sh /root/scripts/Thinkerbell/

# Make it executable
chmod +x /root/scripts/Thinkerbell/ubuntu_sync_runner.sh
```

### **Step 2: Run the Sync**

```bash
# Navigate to your Ubuntu scripts directory
cd /root/scripts/Thinkerbell

# Run the sync
bash ubuntu_sync_runner.sh
```

### **Step 3: Use the Tools**

```bash
# Load the tools
source /root/scripts/Thinkerbell/Thinkerbell_template_pipeline/thinkerbell_launcher.sh

# Check status
main status

# Test components
main test

# Generate validation batch
main validation 50

# Generate full dataset
main generate individual 1000
```

## 📁 **What Gets Synced**

The sync runner automatically copies:

### **Files:**
- ✅ All Python files (`.py`)
- ✅ All shell scripts (`.sh`)
- ✅ All JSON files (`.json`)
- ✅ All markdown files (`.md`)
- ✅ All text files (`.txt`)
- ✅ All requirements files (`requirements*.txt`)

### **Directories:**
- ✅ `thinkerbell/` - Complete modular package structure
- ✅ `data/` - Data files and templates
- ✅ `templates/` - Template files
- ✅ `examples/` - Example files
- ✅ `webapp/` - Web application files
- ✅ `api/` - API components
- ✅ `src/` - Source files

## 🔧 **Sync Runner Features**

### **1. Automatic File Detection**
- Detects new and updated files
- Only syncs files that have changed
- Preserves file permissions and timestamps

### **2. Dependency Installation**
- Automatically installs Python dependencies
- Sets up pip3 if not available
- Installs all required packages

### **3. Launcher Creation**
- Creates Ubuntu-specific launcher script
- Sets up bash aliases
- Makes all scripts executable

### **4. Error Handling**
- Checks if Windows filesystem is mounted
- Validates source and target directories
- Provides detailed error messages

## 📋 **Usage Examples**

### **Basic Sync**
```bash
# Run the sync
bash ubuntu_sync_runner.sh

# Or with explicit command
bash ubuntu_sync_runner.sh sync
```

### **Using the Launcher**
```bash
# Load the launcher
source /root/scripts/Thinkerbell/Thinkerbell_template_pipeline/thinkerbell_launcher.sh

# Check system status
main status

# Test all components
main test

# Generate validation batch
main validation 100

# Generate full dataset
main generate individual 1000

# Run quality analysis
main quality synthetic_dataset/

# Sync files again
main sync
```

### **Using Aliases (after restarting terminal)**
```bash
# Check status
thinkerbell status

# Test components
tb test

# Generate dataset
thinkerbell generate individual 1000
```

## 🔄 **Automatic Sync Workflow**

### **When New Files Are Added:**

1. **Files are created in Windows environment**
2. **Run sync in Ubuntu:**
   ```bash
   bash ubuntu_sync_runner.sh
   ```
3. **Files are automatically copied to Ubuntu**
4. **Dependencies are installed**
5. **Launcher is updated**
6. **Ready to use!**

### **When Files Are Updated:**

1. **Files are modified in Windows environment**
2. **Run sync in Ubuntu:**
   ```bash
   bash ubuntu_sync_runner.sh
   ```
3. **Only changed files are copied**
4. **System is updated**
5. **Ready to use!**

## 🛠️ **Configuration**

### **Source Directory**
The sync runner looks for files in:
```
/mnt/c/Users/Admin/OneDrive/Desktop/scripts/Thinkerbell/Thinkerbell_template_pipeline
```

### **Target Directory**
Files are copied to:
```
/root/scripts/Thinkerbell/Thinkerbell_template_pipeline
```

### **Customizing Paths**
You can modify the paths in `ubuntu_sync_runner.sh`:

```bash
# Configuration
SOURCE_DIR="/mnt/c/Users/Admin/OneDrive/Desktop/scripts/Thinkerbell/Thinkerbell_template_pipeline"
TARGET_DIR="/root/scripts/Thinkerbell/Thinkerbell_template_pipeline"
```

## 📊 **Sync Log**

The sync runner creates a detailed log of all operations:

```bash
# View sync log
cat /root/scripts/Thinkerbell/Thinkerbell_template_pipeline/sync_log.json
```

The log includes:
- ✅ Files synced
- ✅ Files updated
- ✅ Files failed
- ✅ Timestamps
- ✅ Error messages

## 🚨 **Troubleshooting**

### **Windows Filesystem Not Mounted**
```bash
# Check if Windows filesystem is mounted
ls /mnt/c/Users/Admin/OneDrive/Desktop/scripts/Thinkerbell/

# If not mounted, mount it
sudo mount -t drvfs C: /mnt/c
```

### **Permission Issues**
```bash
# Make sync runner executable
chmod +x /root/scripts/Thinkerbell/ubuntu_sync_runner.sh

# Make launcher executable
chmod +x /root/scripts/Thinkerbell/Thinkerbell_template_pipeline/thinkerbell_launcher.sh
```

### **Python Dependencies**
```bash
# Install dependencies manually
pip3 install numpy pandas scikit-learn nltk textstat regex sentence-transformers transformers torch pydantic jsonschema pathlib2 tqdm
```

### **Directory Issues**
```bash
# Create target directory manually
mkdir -p /root/scripts/Thinkerbell/Thinkerbell_template_pipeline

# Check directory permissions
ls -la /root/scripts/Thinkerbell/
```

## 🎯 **Benefits**

### **1. Automatic Updates**
- No manual file copying
- Automatic dependency installation
- Automatic launcher updates

### **2. Cross-Platform Compatibility**
- Works between Windows and Ubuntu
- Preserves file structure
- Maintains file permissions

### **3. Error Recovery**
- Detailed error logging
- Automatic retry mechanisms
- Graceful failure handling

### **4. Development Workflow**
- Seamless development experience
- No context switching
- Automatic tool updates

## 📈 **Performance**

### **Sync Speed**
- ✅ Fast file detection using hashes
- ✅ Only syncs changed files
- ✅ Parallel file operations
- ✅ Minimal network overhead

### **Resource Usage**
- ✅ Low memory footprint
- ✅ Minimal CPU usage
- ✅ Efficient file I/O
- ✅ Clean error handling

## 🔮 **Future Enhancements**

### **Planned Features**
- 🔄 Real-time file watching
- 🔄 Automatic sync on file changes
- 🔄 Cloud backup integration
- 🔄 Multi-environment support
- 🔄 Version control integration

### **Advanced Features**
- 🔄 Selective file syncing
- 🔄 Conflict resolution
- 🔄 Rollback capabilities
- 🔄 Performance optimization

## 📞 **Support**

If you encounter any issues:

1. **Check the sync log:** `cat sync_log.json`
2. **Verify file permissions:** `ls -la /root/scripts/Thinkerbell/`
3. **Test Python environment:** `python3 --version`
4. **Check dependencies:** `pip3 list | grep thinkerbell`

The sync runner is designed to be robust and self-healing, but if you need help, the detailed logs will help identify the issue.

---

**🎉 Ready to sync your Thinkerbell files automatically!** 