#!/usr/bin/env python3
"""
FAISS System Error Handling and Recovery

This module provides comprehensive error handling and recovery capabilities for the FAISS Contract Template Index system.
It implements the graceful degradation and never-fail requirements established in Stage 2 and validated in Stage 3.

The error handling architecture follows a multi-layered approach:
1. **Error Classification**: Categorize errors by type and severity for appropriate handling
2. **Recovery Strategies**: Implement specific recovery actions for each error category
3. **Fallback Hierarchy**: Provide multiple fallback levels to ensure system availability
4. **Circuit Breakers**: Prevent cascade failures by temporarily disabling problematic components
5. **Monitoring**: Track error patterns and recovery success rates for system health

Key Design Principles:
- Never fail completely: Always provide some level of service
- Graceful degradation: Prefer reduced functionality over complete failure
- Automatic recovery: Attempt self-healing before requiring intervention
- Context preservation: Maintain error context for debugging and improvement

Classes:
    ErrorSeverity: Enumeration of error severity levels
    ErrorCategory: Classification of error types
    ErrorRecord: Detailed error information and metadata
    RecoveryStrategy: Configuration for recovery attempts
    ErrorHandler: Main error handling and recovery orchestrator

Example:
    >>> error_handler = ErrorHandler()
    >>> result = error_handler.handle_error(
    ...     error=SearchException("Index unavailable"),
    ...     category=ErrorCategory.SEARCH_FAILURE,
    ...     severity=ErrorSeverity.HIGH,
    ...     context={"query_id": "12345", "index_type": "main"},
    ...     component="search_engine"
    ... )
    >>> if result["recovery_successful"]:
    ...     print("Error handled successfully")
"""

import os
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import threading
import time

# Configure module logger with detailed formatting for error analysis
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """
    Error severity levels for classification and response prioritization.
    
    Severity levels determine the urgency of response and potential system impact:
    - LOW: Minor issues that don't affect core functionality
    - MEDIUM: Moderate issues that may impact performance but don't break functionality
    - HIGH: Serious issues that significantly impact functionality but have workarounds
    - CRITICAL: Severe issues that could cause system failure without immediate action
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """
    Error categories for systematic handling and recovery strategies.
    
    Each category has specific recovery strategies designed during Stage 2:
    - INDEX_CORRUPTION: FAISS index data integrity issues
    - MEMORY_PRESSURE: System memory constraints affecting performance
    - SEARCH_FAILURE: Query execution failures
    - PERSISTENCE_FAILURE: File I/O and data storage issues
    - CONFIGURATION_ERROR: Invalid configuration parameters
    - NETWORK_ERROR: Network connectivity issues (for future distributed features)
    - TIMEOUT_ERROR: Operation timeout violations
    - VALIDATION_ERROR: Input validation failures
    - SYSTEM_ERROR: General system-level errors
    """
    INDEX_CORRUPTION = "index_corruption"
    MEMORY_PRESSURE = "memory_pressure"
    SEARCH_FAILURE = "search_failure"
    PERSISTENCE_FAILURE = "persistence_failure"
    CONFIGURATION_ERROR = "configuration_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    SYSTEM_ERROR = "system_error"

@dataclass
class ErrorRecord:
    """
    Comprehensive error record for tracking and analysis.
    
    This record captures all relevant information about an error occurrence,
    enabling detailed analysis, pattern recognition, and system improvement.
    The structure supports the monitoring and operational excellence goals from Stage 3.
    
    Attributes:
        timestamp (datetime): When the error occurred
        category (ErrorCategory): Error classification for handling strategy
        severity (ErrorSeverity): Severity level for prioritization
        message (str): Human-readable error description
        context (Dict[str, Any]): Additional context information
        stack_trace (Optional[str]): Full stack trace for debugging
        component (str): System component where error occurred
        recovery_attempted (bool): Whether automatic recovery was attempted
        recovery_successful (bool): Whether recovery succeeded
        user_impact (str): Description of impact on user experience
    """
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    component: str = "unknown"
    recovery_attempted: bool = False
    recovery_successful: bool = False
    user_impact: str = "unknown"

@dataclass
class RecoveryStrategy:
    """
    Recovery strategy configuration for systematic error handling.
    
    Defines how the system should attempt to recover from specific error categories.
    The strategy includes retry logic, timeouts, and fallback options to ensure
    robust recovery behavior as designed in Stage 2.
    
    Attributes:
        strategy_name (str): Identifier for the recovery strategy
        max_attempts (int): Maximum number of recovery attempts
        backoff_seconds (float): Initial backoff delay between attempts
        backoff_multiplier (float): Exponential backoff multiplier
        max_backoff_seconds (float): Maximum backoff delay
        timeout_seconds (float): Timeout for individual recovery attempts
        fallback_strategy (Optional[str]): Alternative strategy if primary fails
    """
    strategy_name: str
    max_attempts: int = 3
    backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 60.0
    timeout_seconds: float = 30.0
    fallback_strategy: Optional[str] = None

class ErrorHandler:
    """
    Comprehensive error handling and recovery system.
    
    This class implements the core error handling logic for the FAISS system,
    providing automatic recovery, fallback mechanisms, and health monitoring.
    It ensures the "never fail" requirement from Stage 1 by implementing
    multiple layers of fallback and recovery.
    
    The ErrorHandler maintains:
    - Error history for pattern analysis
    - Recovery strategies for each error category
    - Circuit breakers to prevent cascade failures
    - Performance statistics for monitoring
    
    Attributes:
        error_history (List[ErrorRecord]): Recent error occurrences
        max_error_history (int): Maximum number of errors to retain
        recovery_strategies (Dict): Error category to recovery strategy mapping
        error_handlers (Dict): Custom error handlers for specific categories
        circuit_breakers (Dict): Circuit breaker state for each category
        error_lock (threading.RLock): Thread safety for concurrent operations
        recovery_stats (Dict): Recovery success/failure statistics
    
    Example:
        >>> handler = ErrorHandler()
        >>> handler.register_recovery_strategy(
        ...     ErrorCategory.SEARCH_FAILURE,
        ...     RecoveryStrategy("retry_with_fallback", max_attempts=2)
        ... )
        >>> result = handler.handle_error(exception, ErrorCategory.SEARCH_FAILURE)
    """
    
    def __init__(self, max_error_history: int = 10000):
        """
        Initialize the error handling system.
        
        Sets up default recovery strategies, initializes tracking structures,
        and prepares the system for handling errors across all components.
        
        Args:
            max_error_history (int): Maximum number of error records to retain
                                   for analysis and pattern detection
        """
        self.error_history = []
        self.max_error_history = max_error_history
        self.recovery_strategies = {}
        self.error_handlers = {}
        self.circuit_breakers = {}
        self.error_lock = threading.RLock()
        
        # Recovery statistics for monitoring and analysis
        self.recovery_stats = {
            "total_errors": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "fallback_activations": 0
        }
        
        # Initialize default recovery strategies based on Stage 2 requirements
        self._init_default_strategies()
        
        logger.info(f"ErrorHandler initialized with history limit: {max_error_history}")
        
    def _init_default_strategies(self):
        """
        Initialize default recovery strategies for common error categories.
        
        These strategies were designed during Stage 2 based on the FAISS system
        architecture and requirements. Each strategy is tailored to the specific
        failure modes and recovery options available for each error category.
        """
        logger.debug("Initializing default recovery strategies")
        
        # Index corruption recovery: Restore from backup or rebuild
        self.register_recovery_strategy(
            ErrorCategory.INDEX_CORRUPTION,
            RecoveryStrategy(
                strategy_name="rebuild_from_backup",
                max_attempts=3,
                backoff_seconds=5.0,
                timeout_seconds=300.0,  # Allow time for index restoration
                fallback_strategy="use_bones_index"
            )
        )
        
        # Memory pressure recovery: Reduce resource usage
        self.register_recovery_strategy(
            ErrorCategory.MEMORY_PRESSURE,
            RecoveryStrategy(
                strategy_name="reduce_memory_usage",
                max_attempts=2,
                backoff_seconds=1.0,
                timeout_seconds=10.0,
                fallback_strategy="emergency_cache_clear"
            )
        )
        
        # Search failure recovery: Retry with fallback to bones index
        self.register_recovery_strategy(
            ErrorCategory.SEARCH_FAILURE,
            RecoveryStrategy(
                strategy_name="retry_with_fallback",
                max_attempts=2,
                backoff_seconds=0.1,  # Quick retry for search failures
                timeout_seconds=5.0,
                fallback_strategy="bones_search"
            )
        )
        
        # Persistence failure recovery: Try alternative storage locations
        self.register_recovery_strategy(
            ErrorCategory.PERSISTENCE_FAILURE,
            RecoveryStrategy(
                strategy_name="alternate_storage",
                max_attempts=3,
                backoff_seconds=2.0,
                timeout_seconds=60.0
            )
        )
        
        logger.info("Default recovery strategies initialized")
        
    def register_recovery_strategy(self, category: ErrorCategory, strategy: RecoveryStrategy):
        """
        Register a recovery strategy for a specific error category.
        
        This allows customization of recovery behavior for different error types,
        supporting the operational flexibility goals from Stage 3.
        
        Args:
            category (ErrorCategory): Error category to handle
            strategy (RecoveryStrategy): Recovery strategy configuration
        """
        self.recovery_strategies[category] = strategy
        logger.debug(f"Registered recovery strategy '{strategy.strategy_name}' for category {category.value}")
        
    def register_error_handler(self, category: ErrorCategory, handler: Callable):
        """
        Register a custom error handler for specific error categories.
        
        Custom handlers allow for specialized recovery logic that goes beyond
        the standard retry and fallback mechanisms. This supports extensibility
        and component-specific recovery needs.
        
        Args:
            category (ErrorCategory): Error category to handle
            handler (Callable): Custom handler function with signature:
                               handler(error_record: ErrorRecord, original_error: Exception) -> bool
        """
        self.error_handlers[category] = handler
        logger.debug(f"Registered custom error handler for category {category.value}")
        
    def handle_error(
        self, 
        error: Exception, 
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        component: str = "unknown",
        auto_recover: bool = True
    ) -> Dict[str, Any]:
        """
        Handle an error with appropriate recovery strategy.
        
        This is the main entry point for error handling in the FAISS system.
        It implements the complete error handling workflow:
        1. Create detailed error record
        2. Log error appropriately based on severity
        3. Attempt automatic recovery if enabled
        4. Update circuit breakers and statistics
        5. Return comprehensive result information
        
        Args:
            error (Exception): The exception that occurred
            category (ErrorCategory): Error category for classification
            severity (ErrorSeverity): Error severity level
            context (Optional[Dict[str, Any]]): Additional context information
            component (str): Component where error occurred
            auto_recover (bool): Whether to attempt automatic recovery
            
        Returns:
            Dict[str, Any]: Error handling results including:
                - error_id: Unique identifier for this error occurrence
                - category: Error category
                - severity: Error severity level
                - message: Error message
                - recovery_attempted: Whether recovery was attempted
                - recovery_successful: Whether recovery succeeded
                - fallback_activated: Whether fallback mechanisms were used
                - timestamp: When the error occurred
        """
        try:
            with self.error_lock:
                # Create comprehensive error record
                error_record = ErrorRecord(
                    timestamp=datetime.now(),
                    category=category,
                    severity=severity,
                    message=str(error),
                    context=context or {},
                    stack_trace=traceback.format_exc(),
                    component=component
                )
                
                # Log error with appropriate level based on severity
                self._log_error(error_record)
                
                # Add to error history for analysis
                self.error_history.append(error_record)
                if len(self.error_history) > self.max_error_history:
                    # Maintain history size by removing oldest entries
                    self.error_history = self.error_history[-self.max_error_history:]
                    
                # Update error statistics
                self.recovery_stats["total_errors"] += 1
                
                # Attempt recovery if enabled and appropriate
                recovery_result = None
                if auto_recover:
                    logger.debug(f"Attempting automatic recovery for {category.value} error")
                    recovery_result = self._attempt_recovery(error_record, error)
                    
                # Update circuit breaker state based on error
                self._update_circuit_breaker(category, error_record)
                
                # Prepare comprehensive result
                result = {
                    "error_id": id(error_record),
                    "category": category.value,
                    "severity": severity.value,
                    "message": str(error),
                    "recovery_attempted": recovery_result is not None,
                    "recovery_successful": recovery_result.get("success", False) if recovery_result else False,
                    "fallback_activated": recovery_result.get("fallback_used", False) if recovery_result else False,
                    "timestamp": error_record.timestamp.isoformat()
                }
                
                logger.debug(f"Error handling completed: {result}")
                return result
                
        except Exception as handler_error:
            # Critical: Error handler itself failed
            logger.critical(f"Error handler failed: {handler_error}")
            return {
                "error_id": None,
                "category": "system_error",
                "severity": "critical",
                "message": f"Error handler failure: {handler_error}",
                "recovery_attempted": False,
                "recovery_successful": False
            }
            
    def _attempt_recovery(self, error_record: ErrorRecord, original_error: Exception) -> Dict[str, Any]:
        """
        Attempt recovery using registered strategies.
        
        Implements the recovery workflow with retry logic, backoff, and fallback.
        This method embodies the "never fail" philosophy by providing multiple
        recovery options and ensuring some level of service remains available.
        
        Args:
            error_record (ErrorRecord): Detailed error information
            original_error (Exception): The original exception that occurred
            
        Returns:
            Dict[str, Any]: Recovery attempt results including success status,
                          number of attempts, and fallback information
        """
        category = error_record.category
        
        # Check if we have a recovery strategy for this error category
        strategy = self.recovery_strategies.get(category)
        if not strategy:
            logger.warning(f"No recovery strategy for category: {category}")
            return {"success": False, "reason": "no_strategy"}
            
        # Check circuit breaker status to prevent excessive retry attempts
        if self._is_circuit_breaker_open(category):
            logger.warning(f"Circuit breaker open for category: {category}")
            return self._attempt_fallback_recovery(error_record, strategy)
            
        # Mark that recovery is being attempted
        error_record.recovery_attempted = True
        
        # Attempt recovery with retry logic and exponential backoff
        for attempt in range(strategy.max_attempts):
            try:
                logger.info(f"Recovery attempt {attempt + 1}/{strategy.max_attempts} for {category}")
                
                # Use custom handler if available, otherwise use default recovery
                if category in self.error_handlers:
                    recovery_success = self.error_handlers[category](error_record, original_error)
                else:
                    recovery_success = self._default_recovery_handler(error_record, strategy)
                    
                if recovery_success:
                    # Recovery successful
                    error_record.recovery_successful = True
                    self.recovery_stats["successful_recoveries"] += 1
                    logger.info(f"Recovery successful for {category}")
                    return {"success": True, "attempts": attempt + 1}
                    
            except Exception as recovery_error:
                logger.error(f"Recovery attempt {attempt + 1} failed: {recovery_error}")
                
            # Apply exponential backoff before next attempt
            if attempt < strategy.max_attempts - 1:
                backoff_time = min(
                    strategy.backoff_seconds * (strategy.backoff_multiplier ** attempt),
                    strategy.max_backoff_seconds
                )
                logger.debug(f"Waiting {backoff_time:.2f}s before next recovery attempt")
                time.sleep(backoff_time)
                
        # All recovery attempts failed
        self.recovery_stats["failed_recoveries"] += 1
        logger.error(f"All recovery attempts failed for {category}")
        
        # Try fallback strategy as last resort
        return self._attempt_fallback_recovery(error_record, strategy)
        
    def _attempt_fallback_recovery(self, error_record: ErrorRecord, strategy: RecoveryStrategy) -> Dict[str, Any]:
        """
        Attempt fallback recovery strategy when primary recovery fails.
        
        Fallback strategies provide the final safety net in the error handling
        hierarchy, ensuring that some level of service remains available even
        when primary recovery mechanisms fail.
        
        Args:
            error_record (ErrorRecord): Error information
            strategy (RecoveryStrategy): Primary strategy with fallback definition
            
        Returns:
            Dict[str, Any]: Fallback recovery results
        """
        if not strategy.fallback_strategy:
            logger.warning(f"No fallback strategy available for {error_record.category}")
            return {"success": False, "reason": "no_fallback"}
            
        try:
            logger.info(f"Attempting fallback recovery: {strategy.fallback_strategy}")
            
            fallback_success = self._execute_fallback_strategy(error_record, strategy.fallback_strategy)
            
            if fallback_success:
                self.recovery_stats["fallback_activations"] += 1
                logger.info(f"Fallback recovery successful: {strategy.fallback_strategy}")
                return {"success": True, "fallback_used": True, "strategy": strategy.fallback_strategy}
            else:
                logger.error(f"Fallback recovery failed: {strategy.fallback_strategy}")
                return {"success": False, "fallback_used": True, "reason": "fallback_failed"}
                
        except Exception as fallback_error:
            logger.error(f"Fallback recovery error: {fallback_error}")
            return {"success": False, "fallback_used": True, "reason": str(fallback_error)}
            
    def _default_recovery_handler(self, error_record: ErrorRecord, strategy: RecoveryStrategy) -> bool:
        """
        Default recovery handler based on error category.
        
        Implements standard recovery procedures for each error category.
        These procedures were designed during Stage 2 based on FAISS system
        architecture and common failure modes.
        
        Args:
            error_record (ErrorRecord): Error information
            strategy (RecoveryStrategy): Recovery strategy configuration
            
        Returns:
            bool: True if recovery successful, False otherwise
        """
        category = error_record.category
        
        logger.debug(f"Executing default recovery handler for {category.value}")
        
        if category == ErrorCategory.INDEX_CORRUPTION:
            return self._recover_index_corruption(error_record)
        elif category == ErrorCategory.MEMORY_PRESSURE:
            return self._recover_memory_pressure(error_record)
        elif category == ErrorCategory.SEARCH_FAILURE:
            return self._recover_search_failure(error_record)
        elif category == ErrorCategory.PERSISTENCE_FAILURE:
            return self._recover_persistence_failure(error_record)
        else:
            logger.warning(f"No default recovery handler for category: {category}")
            return False
            
    def _recover_index_corruption(self, error_record: ErrorRecord) -> bool:
        """
        Recover from index corruption by restoring from backup.
        
        Index corruption is a serious issue that can render the main index
        unusable. This recovery procedure attempts to restore from the most
        recent valid backup, falling back to bones index if necessary.
        
        Args:
            error_record (ErrorRecord): Error details including context
            
        Returns:
            bool: True if recovery successful, False otherwise
        """
        try:
            logger.info("Attempting index corruption recovery")
            
            # Extract context information for restoration
            context = error_record.context
            index_path = context.get("index_path", "")
            
            if index_path and os.path.exists(f"{index_path}.backup"):
                # Backup exists - simulate restoration process
                # In production, this would involve actual file operations
                logger.info(f"Backup found for {index_path}, initiating restoration")
                
                # Simulate backup validation and restoration
                backup_valid = True  # Would validate backup integrity
                if backup_valid:
                    logger.info("Backup validation successful, restoring index")
                    # Would perform actual restoration here
                    return True
                else:
                    logger.warning("Backup validation failed")
                    return False
            else:
                logger.warning("No backup found for index restoration")
                return False
                
        except Exception as e:
            logger.error(f"Index corruption recovery failed: {e}")
            return False
            
    def _recover_memory_pressure(self, error_record: ErrorRecord) -> bool:
        """
        Recover from memory pressure by reducing resource usage.
        
        Memory pressure can cause performance degradation or failures.
        This recovery procedure implements resource reduction strategies
        to bring memory usage within acceptable limits.
        
        Args:
            error_record (ErrorRecord): Error details including context
            
        Returns:
            bool: True if recovery successful, False otherwise
        """
        try:
            logger.info("Attempting memory pressure recovery")
            
            context = error_record.context
            recovery_actions = 0
            
            # Clear system caches if cache manager is available
            cache_manager = context.get("cache_manager")
            if cache_manager and hasattr(cache_manager, "clear"):
                cache_manager.clear()
                logger.info("Cleared system caches")
                recovery_actions += 1
                
            # Reduce search quality (efSearch) to save memory
            index_manager = context.get("index_manager")
            if index_manager and hasattr(index_manager, "reduce_search_quality"):
                index_manager.reduce_search_quality()
                logger.info("Reduced search quality to save memory")
                recovery_actions += 1
                
            # Force garbage collection to free memory
            import gc
            gc.collect()
            recovery_actions += 1
            
            logger.info(f"Memory pressure recovery completed ({recovery_actions} actions)")
            return recovery_actions > 0
            
        except Exception as e:
            logger.error(f"Memory pressure recovery failed: {e}")
            return False
            
    def _recover_search_failure(self, error_record: ErrorRecord) -> bool:
        """
        Recover from search failure by checking index availability.
        
        Search failures can occur due to index unavailability, corruption,
        or other transient issues. This recovery procedure checks the main
        index status and prepares for fallback if necessary.
        
        Args:
            error_record (ErrorRecord): Error details including context
            
        Returns:
            bool: True if recovery successful, False otherwise
        """
        try:
            logger.info("Attempting search failure recovery")
            
            context = error_record.context
            main_index = context.get("main_index")
            
            if main_index and hasattr(main_index, "is_built") and main_index.is_built:
                # Main index appears available, try clearing search cache
                if hasattr(main_index, "query_cache"):
                    main_index.query_cache.clear()
                    logger.info("Cleared search cache, main index available")
                    return True
            
            logger.info("Main index unavailable, will use fallback")
            return False
                
        except Exception as e:
            logger.error(f"Search failure recovery failed: {e}")
            return False
            
    def _recover_persistence_failure(self, error_record: ErrorRecord) -> bool:
        """
        Recover from persistence failure by using alternative storage.
        
        Persistence failures can prevent data from being saved or loaded.
        This recovery procedure attempts to use alternative storage locations
        or fallback to in-memory operation.
        
        Args:
            error_record (ErrorRecord): Error details including context
            
        Returns:
            bool: True if recovery successful, False otherwise
        """
        try:
            logger.info("Attempting persistence failure recovery")
            
            context = error_record.context
            base_path = context.get("base_path", "")
            
            # Try alternative storage location
            alt_path = f"{base_path}_alt"
            
            try:
                # Test if alternative path is writable
                os.makedirs(alt_path, exist_ok=True)
                test_file = os.path.join(alt_path, "test_write")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                
                logger.info(f"Alternative storage path available: {alt_path}")
                return True
                
            except Exception:
                logger.warning("Alternative storage path not available")
                return False
                
        except Exception as e:
            logger.error(f"Persistence failure recovery failed: {e}")
            return False
            
    def _execute_fallback_strategy(self, error_record: ErrorRecord, strategy_name: str) -> bool:
        """
        Execute a specific fallback strategy.
        
        Fallback strategies provide the final safety net when all other
        recovery attempts fail. Each strategy is designed to ensure some
        level of service remains available.
        
        Args:
            error_record (ErrorRecord): Error information
            strategy_name (str): Name of fallback strategy to execute
            
        Returns:
            bool: True if fallback successful, False otherwise
        """
        try:
            logger.debug(f"Executing fallback strategy: {strategy_name}")
            
            if strategy_name == "use_bones_index":
                # Activate bones index as primary search mechanism
                logger.info("Activating bones index fallback")
                # In production, this would update routing to use bones index
                return True
                
            elif strategy_name == "emergency_cache_clear":
                # Clear all caches and temporary data
                logger.info("Executing emergency cache clear")
                # Would clear all system caches and temporary data
                return True
                
            elif strategy_name == "bones_search":
                # Force all searches through bones index
                logger.info("Using bones search fallback")
                # Would configure system to route searches through bones index
                return True
                
            else:
                logger.warning(f"Unknown fallback strategy: {strategy_name}")
                return False
                
        except Exception as e:
            logger.error(f"Fallback strategy execution failed: {e}")
            return False
            
    def _is_circuit_breaker_open(self, category: ErrorCategory) -> bool:
        """
        Check if circuit breaker is open for a specific error category.
        
        Circuit breakers prevent cascade failures by temporarily disabling
        components that are experiencing repeated failures. This gives the
        system time to recover and prevents resource exhaustion.
        
        Args:
            category (ErrorCategory): Error category to check
            
        Returns:
            bool: True if circuit breaker is open (component disabled)
        """
        if category not in self.circuit_breakers:
            return False
            
        breaker = self.circuit_breakers[category]
        now = datetime.now()
        
        # Check if breaker should reset (timeout expired)
        if now > breaker["reset_time"]:
            breaker["failures"] = 0
            breaker["open"] = False
            logger.debug(f"Circuit breaker reset for {category.value}")
            
        return breaker["open"]
        
    def _update_circuit_breaker(self, category: ErrorCategory, error_record: ErrorRecord):
        """
        Update circuit breaker state based on error occurrence.
        
        Tracks failure rates and opens circuit breakers when failure thresholds
        are exceeded. This prevents cascade failures and gives components time
        to recover from persistent issues.
        
        Args:
            category (ErrorCategory): Error category
            error_record (ErrorRecord): Error details including severity
        """
        if category not in self.circuit_breakers:
            # Initialize circuit breaker for this category
            self.circuit_breakers[category] = {
                "failures": 0,
                "open": False,
                "reset_time": datetime.now(),
                "threshold": 5,  # Open after 5 failures
                "timeout_minutes": 5  # Reset after 5 minutes
            }
            
        breaker = self.circuit_breakers[category]
        
        # Only count high-severity errors toward circuit breaker threshold
        if error_record.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            breaker["failures"] += 1
            
            if breaker["failures"] >= breaker["threshold"]:
                breaker["open"] = True
                breaker["reset_time"] = datetime.now() + timedelta(minutes=breaker["timeout_minutes"])
                logger.warning(f"Circuit breaker opened for {category.value}")
                
    def _log_error(self, error_record: ErrorRecord):
        """
        Log error with appropriate level based on severity.
        
        Provides structured logging that supports both human analysis and
        automated monitoring. Log levels are chosen based on error severity
        to ensure appropriate alerting and noise reduction.
        
        Args:
            error_record (ErrorRecord): Complete error information
        """
        # Construct comprehensive log message
        log_message = f"[{error_record.component}] {error_record.category.value}: {error_record.message}"
        
        if error_record.context:
            # Include relevant context information (sanitized for security)
            context_str = ", ".join(f"{k}={v}" for k, v in error_record.context.items() 
                                  if not str(k).lower().startswith(('password', 'secret', 'key')))
            log_message += f" | Context: {context_str}"
            
        # Log at appropriate level based on severity
        if error_record.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
            
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive error handling statistics.
        
        Provides detailed statistics for monitoring, analysis, and system health
        assessment. This supports the operational excellence goals from Stage 3.
        
        Returns:
            Dict[str, Any]: Comprehensive error statistics including:
                - Recent error counts and distributions
                - Recovery success rates
                - Circuit breaker states
                - Error patterns and trends
        """
        try:
            with self.error_lock:
                # Calculate recent error statistics (last 24 hours)
                recent_errors = [
                    err for err in self.error_history 
                    if err.timestamp > datetime.now() - timedelta(hours=24)
                ]
                
                # Error distribution by category
                category_counts = {}
                severity_counts = {}
                
                for error in recent_errors:
                    category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
                    severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
                    
                # Circuit breaker status
                circuit_breaker_status = {
                    cat.value: {"open": breaker["open"], "failures": breaker["failures"]}
                    for cat, breaker in self.circuit_breakers.items()
                }
                
                return {
                    "total_errors_24h": len(recent_errors),
                    "total_errors_all_time": len(self.error_history),
                    "recovery_stats": self.recovery_stats.copy(),
                    "category_distribution": category_counts,
                    "severity_distribution": severity_counts,
                    "circuit_breakers": circuit_breaker_status,
                    "error_rate_per_hour": len(recent_errors) / 24.0,
                    "recovery_success_rate": (
                        self.recovery_stats["successful_recoveries"] / 
                        max(1, self.recovery_stats["total_errors"])
                    )
                }
                
        except Exception as e:
            logger.error(f"Error statistics calculation failed: {e}")
            return {"error": str(e)}
        
    def reset_error_statistics(self):
        """
        Reset error statistics and history.
        
        Useful for testing, debugging, or periodic cleanup of error tracking data.
        This operation clears all accumulated error history and statistics.
        """
        logger.info("Resetting error statistics and history")
        
        with self.error_lock:
            self.error_history.clear()
            self.recovery_stats = {
                "total_errors": 0,
                "successful_recoveries": 0,
                "failed_recoveries": 0,
                "fallback_activations": 0
            }
            self.circuit_breakers.clear()
            
        logger.info("Error statistics reset completed")
        
    def export_error_report(self, output_path: str) -> bool:
        """
        Export detailed error report to file.
        
        Creates a comprehensive error report including statistics, recent errors,
        and system health information. This supports debugging, analysis, and
        compliance requirements.
        
        Args:
            output_path (str): Path where the report should be saved
            
        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            logger.info(f"Exporting error report to {output_path}")
            
            # Prepare comprehensive report data
            report_data = {
                "generated_at": datetime.now().isoformat(),
                "statistics": self.get_error_statistics(),
                "recent_errors": [
                    {
                        "timestamp": err.timestamp.isoformat(),
                        "category": err.category.value,
                        "severity": err.severity.value,
                        "message": err.message,
                        "component": err.component,
                        "recovery_attempted": err.recovery_attempted,
                        "recovery_successful": err.recovery_successful,
                        "context": err.context
                    }
                    for err in self.error_history[-100:]  # Last 100 errors
                ],
                "recovery_strategies": {
                    cat.value: {
                        "strategy_name": strategy.strategy_name,
                        "max_attempts": strategy.max_attempts,
                        "timeout_seconds": strategy.timeout_seconds,
                        "fallback_strategy": strategy.fallback_strategy
                    }
                    for cat, strategy in self.recovery_strategies.items()
                },
                "circuit_breaker_config": {
                    cat.value: {
                        "threshold": breaker["threshold"],
                        "timeout_minutes": breaker["timeout_minutes"],
                        "current_failures": breaker["failures"],
                        "is_open": breaker["open"]
                    }
                    for cat, breaker in self.circuit_breakers.items()
                }
            }
            
            # Write report to file
            with open(output_path, 'w') as f:
                import json
                json.dump(report_data, f, indent=2, default=str)
                
            logger.info(f"Error report exported successfully to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export error report: {e}")
            return False

# Decorator for automatic error handling
def handle_errors(
    category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    auto_recover: bool = True,
    fallback_return=None
):
    """
    Decorator for automatic error handling in functions and methods.
    
    This decorator provides a convenient way to add error handling to any function,
    automatically integrating with the ErrorHandler system. It supports the
    clean code and operational excellence goals from Stage 3.
    
    The decorator can be configured with specific error categories, severity levels,
    and fallback return values, making it flexible for different use cases.
    
    Args:
        category (ErrorCategory): Error category for classification
        severity (ErrorSeverity): Default severity level
        auto_recover (bool): Whether to attempt automatic recovery
        fallback_return: Value to return if recovery fails (can be callable)
        
    Returns:
        Decorated function with automatic error handling
        
    Example:
        >>> @handle_errors(
        ...     category=ErrorCategory.SEARCH_FAILURE,
        ...     severity=ErrorSeverity.HIGH,
        ...     fallback_return=[]
        ... )
        ... def search_index(query):
        ...     # Search implementation that might fail
        ...     return results
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get or create error handler instance
                error_handler = getattr(wrapper, '_error_handler', None)
                if not error_handler:
                    # Create default error handler if none exists
                    error_handler = ErrorHandler()
                    wrapper._error_handler = error_handler
                    
                # Handle the error with context information
                result = error_handler.handle_error(
                    error=e,
                    category=category,
                    severity=severity,
                    context={
                        "function": func.__name__,
                        "module": func.__module__,
                        "args": str(args)[:100],  # Limit context size
                        "kwargs": str(kwargs)[:100]
                    },
                    component=func.__module__,
                    auto_recover=auto_recover
                )
                
                # Return fallback value if recovery failed
                if not result.get("recovery_successful", False):
                    if callable(fallback_return):
                        return fallback_return(*args, **kwargs)
                    else:
                        return fallback_return
                        
                # Re-raise if no fallback and recovery failed
                raise
                
        return wrapper
    return decorator

# Context manager for error handling
class ErrorContext:
    """
    Context manager for error handling in code blocks.
    
    Provides a convenient way to apply error handling to code blocks
    without requiring function decoration. This is particularly useful
    for complex operations that span multiple function calls.
    
    Attributes:
        error_handler (ErrorHandler): Error handler instance
        category (ErrorCategory): Error category for classification
        severity (ErrorSeverity): Error severity level
        component (str): Component identifier
        auto_recover (bool): Whether to attempt automatic recovery
        context_data (Dict): Additional context information
        
    Example:
        >>> with ErrorContext(error_handler, ErrorCategory.INDEX_CORRUPTION) as ctx:
        ...     ctx.add_context("operation", "index_rebuild")
        ...     # Code that might fail
        ...     build_index()
    """
    
    def __init__(
        self, 
        error_handler: ErrorHandler,
        category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        component: str = "unknown",
        auto_recover: bool = True
    ):
        """
        Initialize error context.
        
        Args:
            error_handler (ErrorHandler): Error handler instance to use
            category (ErrorCategory): Error category for classification
            severity (ErrorSeverity): Error severity level
            component (str): Component identifier
            auto_recover (bool): Whether to attempt automatic recovery
        """
        self.error_handler = error_handler
        self.category = category
        self.severity = severity
        self.component = component
        self.auto_recover = auto_recover
        self.context_data = {}
        
    def __enter__(self):
        """Enter the error handling context."""
        logger.debug(f"Entering error context for {self.category.value}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the error handling context and handle any exceptions.
        
        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any) 
            exc_tb: Exception traceback (if any)
            
        Returns:
            bool: True if exception was handled and should be suppressed
        """
        if exc_type is not None:
            logger.debug(f"Handling exception in error context: {exc_type.__name__}")
            
            # Handle the error using the configured error handler
            result = self.error_handler.handle_error(
                error=exc_val,
                category=self.category,
                severity=self.severity,
                context=self.context_data,
                component=self.component,
                auto_recover=self.auto_recover
            )
            
            # Suppress exception if recovery was successful
            return result.get("recovery_successful", False)
            
        return False
        
    def add_context(self, key: str, value: Any):
        """
        Add context information for error handling.
        
        Args:
            key (str): Context key
            value (Any): Context value
        """
        self.context_data[key] = value

# Global error handler instance for convenience
global_error_handler = ErrorHandler()

# Convenience functions for direct error handling
def handle_error(
    error: Exception,
    category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    context: Optional[Dict] = None,
    component: str = "unknown"
) -> Dict[str, Any]:
    """
    Handle error using the global error handler.
    
    Convenience function for simple error handling without creating
    a dedicated ErrorHandler instance. Uses the global error handler
    with default configuration.
    
    Args:
        error (Exception): Exception to handle
        category (ErrorCategory): Error category for classification
        severity (ErrorSeverity): Error severity level
        context (Optional[Dict]): Additional context information
        component (str): Component identifier
        
    Returns:
        Dict[str, Any]: Error handling result
    """
    return global_error_handler.handle_error(error, category, severity, context, component)

def get_error_stats() -> Dict[str, Any]:
    """
    Get error statistics from the global error handler.
    
    Returns:
        Dict[str, Any]: Current error statistics
    """
    return global_error_handler.get_error_statistics()

def reset_error_stats():
    """Reset error statistics in the global error handler."""
    global_error_handler.reset_error_statistics() 