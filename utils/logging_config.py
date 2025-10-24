import logging
import os
from datetime import datetime

def setup_logging(log_level=logging.INFO, log_dir="logs"):
    """
    配置统一的日志系统，所有日志都会保存到logs文件夹
    
    Args:
        log_level: 日志级别，默认为INFO
        log_dir: 日志文件夹路径，默认为logs
    """
    # 确保logs文件夹存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建日志文件名（按日期）
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 配置根日志器（只输出到控制台，不写入文件）
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 只添加控制台处理器到根日志器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    root_logger.addHandler(console_handler)
    
    # 为特定模块设置不同的日志文件
    modules_log_config = {
        # 'backend.server.app.task_manager': os.path.join(log_dir, f"task_manager_{today}.log"),
        # 'backend.server.app.router': os.path.join(log_dir, f"router_{today}.log"),
        # 'backend.server.app.accept': os.path.join(log_dir, f"accept_{today}.log"),
        # 'backend.llm': os.path.join(log_dir, f"llm_{today}.log"),
        # 'backend.agents.base': os.path.join(log_dir, f"base_{today}.log"),
        # 'backend.ordagent': os.path.join(log_dir, f"ordagent_{today}.log"),
        # 'backend.celery_tasks': os.path.join(log_dir, f"celery_{today}.log"),
        # '__main__': os.path.join(log_dir, f"app_{today}.log"),  # 主模块日志
    }
    
    # 为每个模块创建专门的日志文件
    for module_name, log_file_path in modules_log_config.items():
        module_logger = logging.getLogger(module_name)
        # 清除现有的处理器
        for handler in module_logger.handlers[:]:
            module_logger.removeHandler(handler)
        
        # 添加文件处理器
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        module_logger.addHandler(file_handler)
        module_logger.setLevel(log_level)
        # 防止日志传播到父日志器
        module_logger.propagate = False
    
    return logging.getLogger(__name__)

def get_logger(name):
    """
    获取配置好的logger实例
    
    Args:
        name: logger名称，通常是__name__
    
    Returns:
        配置好的logger实例
    """
    return logging.getLogger(name)
