"""
日志使用示例

展示如何在项目中使用配置好的日志系统，包括按日期轮转功能。
"""

from app.infra.logging import get_logger

# 获取当前模块的日志记录器
logger = get_logger(__name__)


def example_logging_usage():
    """展示日志使用示例"""
    
    # 不同级别的日志
    logger.debug("这是调试信息，通常只在开发时使用")
    logger.info("这是一般信息，记录程序正常运行状态")
    logger.warning("这是警告信息，表示可能的问题")
    logger.error("这是错误信息，表示程序遇到了错误")
    logger.critical("这是严重错误信息，表示程序无法继续运行")
    
    # 带参数的日志
    user_id = "12345"
    action = "login"
    logger.info(f"用户 {user_id} 执行了 {action} 操作")
    
    # 使用格式化字符串
    logger.info("处理文件: %s, 大小: %d bytes", "example.txt", 1024)
    
    # 异常日志
    try:
        result = 1 / 0
    except ZeroDivisionError as e:
        logger.error("除零错误: %s", str(e), exc_info=True)


def example_daily_rotation():
    """展示按日期轮转的日志功能"""
    from datetime import datetime
    
    # 显示当前日期和日志文件名
    current_date = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"当前日期: {current_date}")
    logger.info(f"今天的日志文件: logs/chocolate_{current_date}.log")
    
    # 模拟不同日期的日志
    logger.info("这是一条测试日志，用于演示按日期轮转功能")
    logger.info("每天午夜会自动创建新的日志文件")


if __name__ == "__main__":
    # 确保日志系统已初始化
    from app.infra.logging import setup_logging
    setup_logging()
    
    # 运行示例
    example_logging_usage()
    example_daily_rotation()
