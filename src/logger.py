import logging
import os
import time
from datetime import datetime

class Logger:
    """
    Logger class for the secure cloud management system
    
    This class handles logging of system events, security incidents,
    and performance metrics.
    """
    def __init__(self, log_level='INFO', log_dir='logs'):
        """
        Initialize the logger
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory to store log files
        """
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Generate log file name with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'secure_cloud_{timestamp}.log')
        
        # Configure logging
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # Set log level
        level = getattr(logging, log_level.upper())
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            format=log_format,
            datefmt=date_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('secure_cloud')
        self.logger.info(f"Logger initialized. Log file: {log_file}")
        
        # Create separate security log
        security_log_file = os.path.join(log_dir, f'security_{timestamp}.log')
        
        # Configure security logger
        security_handler = logging.FileHandler(security_log_file)
        security_handler.setFormatter(logging.Formatter(log_format))
        
        self.security_logger = logging.getLogger('secure_cloud.security')
        self.security_logger.addHandler(security_handler)
        self.security_logger.setLevel(level)
        
        self.logger.info(f"Security logger initialized. Log file: {security_log_file}")
    
    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message):
        """Log critical message"""
        self.logger.critical(message)
    
    def security(self, message, threat_type=None, severity=None, source=None):
        """
        Log security incident
        
        Args:
            message: Description of the security incident
            threat_type: Type of threat (ddos, cryptojacking, etc.)
            severity: Severity of the threat (0-1)
            source: Source of the threat (VM index, IP, etc.)
        """
        if threat_type and severity and source:
            log_message = f"SECURITY - {threat_type.upper()} - Severity: {severity:.2f} - Source: {source} - {message}"
        else:
            log_message = f"SECURITY - {message}"
        
        self.security_logger.warning(log_message)
        self.logger.warning(log_message)
    
    def performance(self, metrics):
        """
        Log performance metrics
        
        Args:
            metrics: Dictionary of performance metrics
        """
        metrics_str = ' - '.join([f"{k}: {v}" for k, v in metrics.items()])
        self.logger.info(f"PERFORMANCE - {metrics_str}")