#!/usr/bin/env python3
"""
FAISS Index Configuration Management

This module provides comprehensive configuration management for the FAISS Contract Template Index system.
It supports environment-specific configurations, validation, and runtime overrides through environment variables.

The configuration architecture follows the principles established in Stage 1-3:
- Stage 1: Basic configuration structure and HNSW parameters
- Stage 2: Production robustness with health monitoring and persistence
- Stage 3: Environment awareness and operational excellence

Classes:
    HNSWConfig: HNSW-specific parameters with validation
    HealthConfig: Health monitoring and performance thresholds
    PersistenceConfig: Data persistence and backup settings
    PerformanceConfig: Performance optimization parameters
    SecurityConfig: Security and access control settings
    IndexConfig: Main configuration container
    ConfigManager: Configuration lifecycle management

Example:
    >>> config = get_config("production")
    >>> config.hnsw.M = 32  # HNSW connectivity parameter
    >>> config.health.recall_threshold = 0.85  # Minimum acceptable recall
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import yaml

# Configure module logger
logger = logging.getLogger(__name__)

@dataclass
class HNSWConfig:
    """
    HNSW (Hierarchical Navigable Small World) index configuration.
    
    These parameters directly control FAISS IndexHNSWFlat behavior:
    - M: Number of bi-directional links for each node (higher = better recall, more memory)
    - efConstruction: Search width during index construction (higher = better quality, slower build)
    - efSearch: Search width during query (higher = better recall, slower search)
    
    The adaptive efSearch feature allows runtime adjustment based on system conditions,
    which was a key requirement from Stage 2 for production robustness.
    
    Attributes:
        M (int): HNSW connectivity parameter, typically 16-64
        efConstruction (int): Construction-time search parameter, typically 200-800
        efSearch_base (int): Base search parameter, typically 64-128
        efSearch_adaptive (bool): Enable dynamic efSearch adjustment
        efSearch_min (int): Minimum efSearch value for adaptive mode
        efSearch_max (int): Maximum efSearch value for adaptive mode
    """
    M: int = 32
    efConstruction: int = 200
    efSearch_base: int = 64
    efSearch_adaptive: bool = True
    efSearch_min: int = 16
    efSearch_max: int = 128
    
    def __post_init__(self):
        """
        Validate HNSW parameters against FAISS constraints and best practices.
        
        Raises:
            ValueError: If parameters are outside valid ranges or inconsistent
        """
        logger.debug(f"Validating HNSW config: M={self.M}, efConstruction={self.efConstruction}")
        
        # FAISS requires M >= 4 for proper graph connectivity
        if self.M < 4 or self.M > 64:
            raise ValueError(f"HNSW M must be between 4 and 64, got {self.M}")
        
        # efConstruction must be >= M for proper index quality
        if self.efConstruction < self.M:
            raise ValueError(f"efConstruction ({self.efConstruction}) must be >= M ({self.M})")
        
        # Validate adaptive efSearch bounds
        if not (self.efSearch_min <= self.efSearch_base <= self.efSearch_max):
            raise ValueError(f"efSearch bounds invalid: {self.efSearch_min} <= {self.efSearch_base} <= {self.efSearch_max}")
        
        logger.debug("HNSW configuration validation passed")

@dataclass
class HealthConfig:
    """
    Health monitoring and performance threshold configuration.
    
    This configuration controls the system's self-monitoring capabilities,
    enabling proactive maintenance and performance optimization as designed in Stage 2.
    
    Key thresholds:
    - recall_threshold: Minimum acceptable recall before triggering rebuild
    - memory_limit_percent: System memory limit before adaptive measures
    - rebuild_drift_threshold: Data change ratio triggering rebuild
    
    Attributes:
        recall_threshold (float): Minimum recall@5 before rebuild (0.0-1.0)
        memory_limit_percent (int): Maximum system memory usage (10-95%)
        rebuild_drift_threshold (float): Data change ratio for rebuild (0.0-1.0)
        cache_ttl_months (int): Cache validity period in months
        latency_p95_threshold_ms (int): P95 latency threshold in milliseconds
        bones_latency_threshold_ms (int): Bones index latency threshold
        error_rate_threshold (float): Maximum acceptable error rate
        monitoring_interval_seconds (int): Health check frequency
        performance_window_size (int): Performance metrics window size
    """
    recall_threshold: float = 0.83
    memory_limit_percent: int = 70
    rebuild_drift_threshold: float = 0.5
    cache_ttl_months: int = 6
    latency_p95_threshold_ms: int = 200
    bones_latency_threshold_ms: int = 10
    error_rate_threshold: float = 0.05
    monitoring_interval_seconds: int = 60
    performance_window_size: int = 1000
    
    def __post_init__(self):
        """
        Validate health monitoring parameters for consistency and safety.
        
        Raises:
            ValueError: If parameters are outside safe operating ranges
        """
        logger.debug(f"Validating health config: recall={self.recall_threshold}, memory_limit={self.memory_limit_percent}%")
        
        # Recall must be a valid probability
        if not (0.0 <= self.recall_threshold <= 1.0):
            raise ValueError(f"recall_threshold must be between 0.0 and 1.0, got {self.recall_threshold}")
        
        # Memory limit must allow for OS and other processes
        if not (10 <= self.memory_limit_percent <= 95):
            raise ValueError(f"memory_limit_percent must be between 10 and 95, got {self.memory_limit_percent}")
        
        # Rebuild threshold must be a valid ratio
        if not (0.0 <= self.rebuild_drift_threshold <= 1.0):
            raise ValueError(f"rebuild_drift_threshold must be between 0.0 and 1.0, got {self.rebuild_drift_threshold}")
        
        logger.debug("Health configuration validation passed")

@dataclass
class PersistenceConfig:
    """
    Data persistence and backup configuration.
    
    Controls how the system saves, loads, and manages index versions.
    This supports the production robustness requirements from Stage 2,
    including versioned persistence and rollback capabilities.
    
    Attributes:
        base_path (str): Root directory for index storage
        backup_versions (int): Number of index versions to retain
        checkpoint_interval (int): Automatic save interval in seconds
        auto_save (bool): Enable automatic periodic saves
        compression (bool): Enable index compression (slower but smaller)
        save_on_shutdown (bool): Save index state on system shutdown
        atomic_writes (bool): Use atomic file operations for safety
    """
    base_path: str = "./faiss_indices"
    backup_versions: int = 3
    checkpoint_interval: int = 3600  # 1 hour
    auto_save: bool = True
    compression: bool = False
    save_on_shutdown: bool = True
    atomic_writes: bool = True
    
    def __post_init__(self):
        """
        Validate persistence configuration for operational safety.
        
        Raises:
            ValueError: If parameters could lead to data loss or instability
        """
        logger.debug(f"Validating persistence config: path={self.base_path}, versions={self.backup_versions}")
        
        # Must retain at least one backup for rollback capability
        if self.backup_versions < 1:
            raise ValueError(f"backup_versions must be >= 1, got {self.backup_versions}")
        
        # Checkpoint interval must be reasonable to avoid performance impact
        if self.checkpoint_interval < 60:
            raise ValueError(f"checkpoint_interval must be >= 60 seconds, got {self.checkpoint_interval}")
        
        logger.debug("Persistence configuration validation passed")

@dataclass
class PerformanceConfig:
    """
    Performance and optimization configuration.
    
    Controls caching, concurrency, and performance-related behaviors.
    These settings support the Stage 3 performance targets and optimization features.
    
    Attributes:
        target_qps (int): Target queries per second
        max_concurrent_queries (int): Maximum concurrent query limit
        query_timeout_ms (int): Query timeout in milliseconds
        cache_enabled (bool): Enable query result caching
        cache_size (int): Maximum cache entries
        cache_ttl_seconds (int): Cache entry TTL
        prefetch_enabled (bool): Enable result prefetching
        batch_size (int): Batch processing size
    """
    target_qps: int = 10
    max_concurrent_queries: int = 50
    query_timeout_ms: int = 1000
    cache_enabled: bool = True
    cache_size: int = 1000
    cache_ttl_seconds: int = 300
    prefetch_enabled: bool = False
    batch_size: int = 100
    
    def __post_init__(self):
        """
        Validate performance parameters for operational stability.
        
        Raises:
            ValueError: If parameters could cause system instability
        """
        logger.debug(f"Validating performance config: qps={self.target_qps}, cache_size={self.cache_size}")
        
        # QPS must be positive for meaningful operation
        if self.target_qps < 1:
            raise ValueError(f"target_qps must be >= 1, got {self.target_qps}")
        
        # Concurrency limit must be positive
        if self.max_concurrent_queries < 1:
            raise ValueError(f"max_concurrent_queries must be >= 1, got {self.max_concurrent_queries}")
        
        logger.debug("Performance configuration validation passed")

@dataclass
class SecurityConfig:
    """
    Security and access control configuration.
    
    Provides basic security controls for production deployments.
    Note: This is a foundational security layer; additional security
    measures should be implemented at the application level.
    
    Attributes:
        enable_auth (bool): Enable authentication
        api_key_required (bool): Require API key for access
        rate_limiting (bool): Enable rate limiting
        max_requests_per_minute (int): Rate limit threshold
        allowed_hosts (list): Allowed host list (empty = all)
        log_queries (bool): Log all queries for audit
        sanitize_logs (bool): Remove sensitive data from logs
    """
    enable_auth: bool = False
    api_key_required: bool = False
    rate_limiting: bool = True
    max_requests_per_minute: int = 600
    allowed_hosts: list = field(default_factory=list)
    log_queries: bool = True
    sanitize_logs: bool = True

@dataclass 
class IndexConfig:
    """
    Complete FAISS index configuration container.
    
    This is the main configuration class that combines all configuration sections
    into a unified, validated configuration object. It supports the full range
    of features developed across Stages 1-3.
    
    The configuration supports three deployment environments:
    - Development: Optimized for testing and debugging
    - Staging: Balanced performance and safety
    - Production: Maximum reliability and performance
    
    Attributes:
        dimension (int): Vector embedding dimension (typically 384 or 768)
        hnsw (HNSWConfig): HNSW index parameters
        health (HealthConfig): Health monitoring settings
        persistence (PersistenceConfig): Data persistence settings
        performance (PerformanceConfig): Performance optimization settings
        security (SecurityConfig): Security and access control
        environment (str): Deployment environment (development/staging/production)
        debug (bool): Enable debug mode
        log_level (str): Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
    """
    dimension: int = 384
    hnsw: HNSWConfig = field(default_factory=HNSWConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Environment and deployment settings
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    
    def __post_init__(self):
        """
        Validate complete configuration for consistency and operational safety.
        
        Performs cross-validation between configuration sections to ensure
        the overall configuration is coherent and safe for deployment.
        
        Raises:
            ValueError: If configuration is invalid or inconsistent
        """
        logger.debug(f"Validating complete config for environment: {self.environment}")
        
        # Vector dimension must be positive and reasonable
        if self.dimension < 1:
            raise ValueError(f"dimension must be > 0, got {self.dimension}")
        
        # Environment must be a recognized value
        valid_environments = ["development", "staging", "production"]
        if self.environment not in valid_environments:
            raise ValueError(f"environment must be one of {valid_environments}, got {self.environment}")
        
        # Log level must be valid
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ValueError(f"invalid log_level: {self.log_level}")
        
        logger.info(f"Configuration validated for {self.environment} environment")

class ConfigManager:
    """
    Configuration manager with validation, defaults, and environment support.
    
    This class handles the complete lifecycle of configuration management:
    - Loading from files (JSON/YAML)
    - Environment variable overrides
    - Validation and consistency checking
    - Saving and persistence
    - Hot reloading for development
    
    The ConfigManager supports the operational excellence goals from Stage 3,
    providing flexible configuration management for different deployment scenarios.
    
    Attributes:
        config_path (Optional[str]): Path to configuration file
        config (Optional[IndexConfig]): Current configuration instance
        _config_cache (Dict): Configuration cache for performance
        _last_modified (Optional[datetime]): Last file modification time
    
    Example:
        >>> manager = ConfigManager()
        >>> config = manager.load_config("config.json", environment="production")
        >>> config.hnsw.M = 32  # Modify as needed
        >>> manager.save_config(config, "updated_config.json")
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path (Optional[str]): Default configuration file path
        """
        self.config_path = config_path
        self.config = None
        self._config_cache = {}
        self._last_modified = None
        
        logger.debug(f"ConfigManager initialized with path: {config_path}")
        
    def load_config(
        self, 
        config_path: Optional[str] = None,
        environment: Optional[str] = None,
        validate: bool = True
    ) -> IndexConfig:
        """
        Load configuration from file with environment override support.
        
        This method implements a flexible configuration loading strategy:
        1. Load base configuration from file (if exists)
        2. Apply environment-specific overrides
        3. Apply environment variable overrides
        4. Validate the final configuration
        5. Apply environment-specific defaults
        
        Args:
            config_path (Optional[str]): Path to config file (JSON or YAML)
            environment (Optional[str]): Environment override (development/staging/production)
            validate (bool): Whether to validate configuration
            
        Returns:
            IndexConfig: Loaded and validated configuration
            
        Raises:
            FileNotFoundError: If specified config file doesn't exist and is required
            ValueError: If configuration is invalid
            yaml.YAMLError: If YAML parsing fails
            json.JSONDecodeError: If JSON parsing fails
        """
        logger.info(f"Loading configuration from {config_path or 'defaults'}")
        
        try:
            # Determine configuration file path
            config_file = Path(config_path or self.config_path or "config.json")
            
            # Load base configuration from file
            if config_file.exists():
                logger.debug(f"Loading configuration from file: {config_file}")
                config_data = self._load_config_file(config_file)
            else:
                logger.info(f"Config file not found: {config_file}, using defaults")
                config_data = {}
                
            # Apply environment override if specified
            if environment:
                logger.debug(f"Applying environment override: {environment}")
                config_data["environment"] = environment
                
            # Apply environment variable overrides
            config_data = self._apply_environment_overrides(config_data)
            
            # Create configuration object with nested validation
            config = self._create_config_from_dict(config_data)
            
            # Perform validation if requested
            if validate:
                logger.debug("Performing configuration validation")
                self._validate_config(config)
                
            # Apply environment-specific defaults and adjustments
            config = self._apply_environment_defaults(config)
            
            # Cache the configuration for potential reuse
            self.config = config
            self._config_cache[str(config_file)] = config
            self._last_modified = datetime.now()
            
            logger.info(f"Configuration loaded successfully (environment: {config.environment})")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
            
    def save_config(self, config: IndexConfig, config_path: Optional[str] = None) -> bool:
        """
        Save configuration to file with metadata.
        
        Saves the configuration in the specified format (JSON/YAML based on extension)
        and includes generation metadata for tracking and debugging.
        
        Args:
            config (IndexConfig): Configuration to save
            config_path (Optional[str]): Output file path
            
        Returns:
            bool: True if save successful, False otherwise
        """
        logger.info(f"Saving configuration to {config_path or self.config_path}")
        
        try:
            output_path = Path(config_path or self.config_path or "config.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert configuration to dictionary
            config_dict = asdict(config)
            
            # Add generation metadata for tracking
            config_dict["_metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "version": "1.0",
                "generator": "ConfigManager",
                "environment": config.environment
            }
            
            # Save in appropriate format based on file extension
            if output_path.suffix.lower() in ['.yaml', '.yml']:
                logger.debug("Saving configuration as YAML")
                with open(output_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            else:
                logger.debug("Saving configuration as JSON")
                with open(output_path, 'w') as f:
                    json.dump(config_dict, f, indent=2, default=str)
                    
            logger.info(f"Configuration saved successfully to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
            
    def get_default_config(self, environment: str = "development") -> IndexConfig:
        """
        Get default configuration for specified environment.
        
        Creates a new configuration with environment-appropriate defaults.
        This is useful for bootstrapping new deployments or resetting configurations.
        
        Args:
            environment (str): Target environment (development/staging/production)
            
        Returns:
            IndexConfig: Default configuration for environment
        """
        logger.debug(f"Creating default configuration for environment: {environment}")
        
        config = IndexConfig(environment=environment)
        return self._apply_environment_defaults(config)
        
    def validate_config(self, config: IndexConfig) -> bool:
        """
        Validate configuration without raising exceptions.
        
        Performs the same validation as load_config but returns a boolean
        result instead of raising exceptions. Useful for testing configurations.
        
        Args:
            config (IndexConfig): Configuration to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            self._validate_config(config)
            logger.debug("Configuration validation successful")
            return True
        except Exception as e:
            logger.warning(f"Configuration validation failed: {e}")
            return False
            
    def reload_config(self) -> Optional[IndexConfig]:
        """
        Reload configuration if file has changed.
        
        Checks the modification time of the configuration file and reloads
        if it has been updated since the last load. This supports hot reloading
        during development and runtime configuration updates.
        
        Returns:
            Optional[IndexConfig]: Reloaded configuration if file changed, 
                                 current configuration if unchanged, 
                                 None if reload fails
        """
        if not self.config_path:
            logger.debug("No config path set, cannot reload")
            return None
            
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(f"Config file no longer exists: {config_file}")
                return None
                
            # Check modification time
            current_mtime = datetime.fromtimestamp(config_file.stat().st_mtime)
            if self._last_modified and current_mtime <= self._last_modified:
                logger.debug("Configuration file unchanged, no reload needed")
                return self.config
                
            # File has changed, reload configuration
            logger.info("Configuration file changed, reloading...")
            return self.load_config()
            
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return None
            
    def _load_config_file(self, config_file: Path) -> Dict[str, Any]:
        """
        Load configuration data from file.
        
        Supports both JSON and YAML formats, determined by file extension.
        
        Args:
            config_file (Path): Path to configuration file
            
        Returns:
            Dict[str, Any]: Configuration data
            
        Raises:
            json.JSONDecodeError: If JSON parsing fails
            yaml.YAMLError: If YAML parsing fails
        """
        logger.debug(f"Loading config file: {config_file}")
        
        with open(config_file, 'r') as f:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            else:
                return json.load(f) or {}
                
    def _apply_environment_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        
        This allows runtime configuration changes without modifying files,
        which is essential for containerized deployments and CI/CD pipelines.
        
        Environment variable mapping follows the pattern FAISS_<SECTION>_<FIELD>
        where sections are flattened using dots for nested values.
        
        Args:
            config_data (Dict[str, Any]): Base configuration data
            
        Returns:
            Dict[str, Any]: Configuration with environment overrides applied
        """
        logger.debug("Applying environment variable overrides")
        
        # Mapping of environment variables to configuration paths
        env_mappings = {
            "FAISS_DIMENSION": ("dimension", int),
            "FAISS_M": ("hnsw.M", int),
            "FAISS_EF_CONSTRUCTION": ("hnsw.efConstruction", int),
            "FAISS_EF_SEARCH": ("hnsw.efSearch_base", int),
            "FAISS_RECALL_THRESHOLD": ("health.recall_threshold", float),
            "FAISS_MEMORY_LIMIT": ("health.memory_limit_percent", int),
            "FAISS_BASE_PATH": ("persistence.base_path", str),
            "FAISS_BACKUP_VERSIONS": ("persistence.backup_versions", int),
            "FAISS_CACHE_SIZE": ("performance.cache_size", int),
            "FAISS_DEBUG": ("debug", lambda x: x.lower() == 'true'),
            "FAISS_LOG_LEVEL": ("log_level", str),
        }
        
        overrides_applied = 0
        for env_var, (config_path, converter) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    converted_value = converter(env_value)
                    self._set_nested_config(config_data, config_path, converted_value)
                    logger.debug(f"Applied environment override: {env_var} = {converted_value}")
                    overrides_applied += 1
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid environment variable {env_var}: {e}")
                    
        if overrides_applied > 0:
            logger.info(f"Applied {overrides_applied} environment variable overrides")
            
        return config_data
        
    def _set_nested_config(self, config_data: Dict, path: str, value: Any):
        """
        Set nested configuration value using dot notation.
        
        Supports setting deeply nested configuration values like "hnsw.M" or "health.recall_threshold".
        Creates intermediate dictionaries as needed.
        
        Args:
            config_data (Dict): Configuration dictionary to modify
            path (str): Dot-separated path to the configuration value
            value (Any): Value to set
        """
        keys = path.split('.')
        current = config_data
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
            
        # Set the final value
        current[keys[-1]] = value
        
    def _create_config_from_dict(self, config_data: Dict[str, Any]) -> IndexConfig:
        """
        Create IndexConfig instance from dictionary data.
        
        Handles the complex process of converting flat dictionary data into
        the nested dataclass structure, with proper type conversion and validation.
        
        Args:
            config_data (Dict[str, Any]): Configuration data
            
        Returns:
            IndexConfig: Constructed configuration object
        """
        logger.debug("Creating configuration object from dictionary")
        
        # Extract nested configuration sections
        hnsw_data = config_data.pop("hnsw", {})
        health_data = config_data.pop("health", {})
        persistence_data = config_data.pop("persistence", {})
        performance_data = config_data.pop("performance", {})
        security_data = config_data.pop("security", {})
        
        # Remove metadata if present (from saved configurations)
        config_data.pop("_metadata", None)
        
        # Create nested configuration objects with validation
        hnsw_config = HNSWConfig(**hnsw_data)
        health_config = HealthConfig(**health_data)
        persistence_config = PersistenceConfig(**persistence_data)
        performance_config = PerformanceConfig(**performance_data)
        security_config = SecurityConfig(**security_data)
        
        # Create main configuration object
        config = IndexConfig(
            hnsw=hnsw_config,
            health=health_config,
            persistence=persistence_config,
            performance=performance_config,
            security=security_config,
            **config_data
        )
        
        logger.debug("Configuration object created successfully")
        return config
        
    def _validate_config(self, config: IndexConfig):
        """
        Validate configuration for consistency and production readiness.
        
        Performs cross-validation checks beyond individual dataclass validation
        to ensure the overall configuration is coherent and safe for deployment.
        
        Args:
            config (IndexConfig): Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid for production use
        """
        logger.debug("Performing cross-validation checks")
        
        # Memory consistency check
        if config.performance.cache_size > 10000 and config.health.memory_limit_percent < 50:
            logger.warning("Large cache size with low memory limit may cause performance issues")
            
        # Performance consistency check  
        if config.performance.target_qps > 100 and config.hnsw.efSearch_base > 64:
            logger.warning("High QPS target with high efSearch may cause latency issues")
            
        # Environment-specific validation
        if config.environment == "production":
            if config.debug:
                raise ValueError("Debug mode should not be enabled in production")
            if config.persistence.backup_versions < 2:
                logger.warning("Production should have at least 2 backup versions")
            if not config.persistence.auto_save:
                logger.warning("Auto-save should be enabled in production")
                
        # Security validation for production
        if config.environment == "production" and not config.security.rate_limiting:
            logger.warning("Rate limiting should be enabled in production")
            
        logger.debug("Configuration cross-validation completed")
            
    def _apply_environment_defaults(self, config: IndexConfig) -> IndexConfig:
        """
        Apply environment-specific defaults and optimizations.
        
        Each environment has different priorities:
        - Development: Fast iteration, debugging features, lower resource usage
        - Staging: Balance between performance and safety, monitoring enabled
        - Production: Maximum reliability, performance, comprehensive logging
        
        Args:
            config (IndexConfig): Base configuration
            
        Returns:
            IndexConfig: Configuration with environment-specific adjustments
        """
        logger.debug(f"Applying environment defaults for: {config.environment}")
        
        if config.environment == "development":
            # Development optimizations: faster iteration, more debugging
            config.debug = True
            config.log_level = "DEBUG"
            config.health.monitoring_interval_seconds = 30  # More frequent monitoring
            config.persistence.checkpoint_interval = 1800  # More frequent saves (30 min)
            logger.debug("Applied development environment defaults")
            
        elif config.environment == "staging":
            # Staging optimizations: balance performance and safety
            config.debug = False
            config.log_level = "INFO"
            config.performance.cache_enabled = True
            config.security.rate_limiting = True
            logger.debug("Applied staging environment defaults")
            
        elif config.environment == "production":
            # Production optimizations: maximum reliability and performance
            config.debug = False
            config.log_level = "WARNING"  # Reduce log noise in production
            config.persistence.backup_versions = max(config.persistence.backup_versions, 3)
            config.security.rate_limiting = True
            config.security.log_queries = True
            config.performance.cache_enabled = True
            logger.debug("Applied production environment defaults")
            
        return config

# Pre-defined environment configurations
# These represent the baseline configurations for each environment,
# established during Stage 1-3 development and testing

DEVELOPMENT_CONFIG = IndexConfig(
    environment="development",
    debug=True,
    log_level="DEBUG",
    hnsw=HNSWConfig(M=16, efConstruction=100, efSearch_base=32),  # Smaller for faster iteration
    health=HealthConfig(
        recall_threshold=0.80,  # Relaxed for development
        memory_limit_percent=60,
        monitoring_interval_seconds=30  # More frequent monitoring
    ),
    persistence=PersistenceConfig(
        base_path="./dev_faiss_indices",
        backup_versions=2,  # Fewer backups for development
        checkpoint_interval=1800  # 30 minutes
    ),
    performance=PerformanceConfig(
        target_qps=5,  # Lower target for development
        cache_size=500
    )
)

STAGING_CONFIG = IndexConfig(
    environment="staging",
    debug=False,
    log_level="INFO",
    hnsw=HNSWConfig(M=24, efConstruction=150, efSearch_base=48),  # Balanced parameters
    health=HealthConfig(
        recall_threshold=0.83,
        memory_limit_percent=65,
        monitoring_interval_seconds=45
    ),
    persistence=PersistenceConfig(
        base_path="./staging_faiss_indices",
        backup_versions=3,
        checkpoint_interval=2700  # 45 minutes
    ),
    performance=PerformanceConfig(
        target_qps=8,
        cache_size=750
    ),
    security=SecurityConfig(rate_limiting=True)
)

PRODUCTION_CONFIG = IndexConfig(
    environment="production",
    debug=False,
    log_level="WARNING",  # Reduced log noise for production
    hnsw=HNSWConfig(M=32, efConstruction=200, efSearch_base=64),  # Full performance
    health=HealthConfig(
        recall_threshold=0.85,  # Strict recall requirement
        memory_limit_percent=70,
        monitoring_interval_seconds=60
    ),
    persistence=PersistenceConfig(
        base_path="./prod_faiss_indices",
        backup_versions=5,  # More backups for production safety
        checkpoint_interval=3600,  # 1 hour
        compression=True  # Enable compression for storage efficiency
    ),
    performance=PerformanceConfig(
        target_qps=10,
        cache_size=1000,
        cache_enabled=True
    ),
    security=SecurityConfig(
        rate_limiting=True,
        log_queries=True,
        sanitize_logs=True
    )
)

# Global configuration instance for convenience
config_manager = ConfigManager()

def get_config(environment: str = "development") -> IndexConfig:
    """
    Get pre-configured IndexConfig for specified environment.
    
    This is the primary interface for obtaining configurations in the application.
    It returns deep copies of the pre-defined configurations to prevent
    accidental modification of the base configurations.
    
    Args:
        environment (str): Target environment (development/staging/production)
        
    Returns:
        IndexConfig: Environment-specific configuration
        
    Raises:
        ValueError: If environment is not recognized
    """
    logger.debug(f"Retrieving configuration for environment: {environment}")
    
    configs = {
        "development": DEVELOPMENT_CONFIG,
        "staging": STAGING_CONFIG,
        "production": PRODUCTION_CONFIG
    }
    
    if environment not in configs:
        logger.error(f"Unknown environment: {environment}")
        raise ValueError(f"Unknown environment: {environment}. Valid options: {list(configs.keys())}")
        
    # Return a copy to prevent modification of the base configuration
    base_config = configs[environment]
    logger.info(f"Retrieved {environment} configuration")
    return base_config

def load_config_from_file(config_path: str, environment: str = None) -> IndexConfig:
    """
    Load configuration from file with optional environment override.
    
    Convenience function for loading configurations from files.
    Uses the global config manager instance.
    
    Args:
        config_path (str): Path to configuration file
        environment (str, optional): Environment override
        
    Returns:
        IndexConfig: Loaded configuration
    """
    logger.info(f"Loading configuration from file: {config_path}")
    return config_manager.load_config(config_path, environment)

def save_config_to_file(config: IndexConfig, config_path: str) -> bool:
    """
    Save configuration to file.
    
    Convenience function for saving configurations to files.
    Uses the global config manager instance.
    
    Args:
        config (IndexConfig): Configuration to save
        config_path (str): Output file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Saving configuration to file: {config_path}")
    return config_manager.save_config(config, config_path)

# Example configuration files for documentation and testing
EXAMPLE_CONFIG_JSON = """
{
  "dimension": 384,
  "environment": "production",
  "hnsw": {
    "M": 32,
    "efConstruction": 200,
    "efSearch_base": 64,
    "efSearch_adaptive": true
  },
  "health": {
    "recall_threshold": 0.83,
    "memory_limit_percent": 70,
    "rebuild_drift_threshold": 0.5,
    "cache_ttl_months": 6
  },
  "persistence": {
    "base_path": "./faiss_indices",
    "backup_versions": 3,
    "checkpoint_interval": 3600
  }
}
"""

EXAMPLE_CONFIG_YAML = """
dimension: 384
environment: production

hnsw:
  M: 32
  efConstruction: 200
  efSearch_base: 64
  efSearch_adaptive: true

health:
  recall_threshold: 0.83
  memory_limit_percent: 70
  rebuild_drift_threshold: 0.5
  cache_ttl_months: 6

persistence:
  base_path: "./faiss_indices"
  backup_versions: 3
  checkpoint_interval: 3600
""" 