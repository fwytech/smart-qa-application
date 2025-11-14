import functools
import logging
import time
import traceback
from typing import Callable, Any, Optional
import streamlit as st
from config.settings import Settings

logger = logging.getLogger(__name__)

def error_handler(
    func_name: str = None,
    show_in_ui: bool = True,
    log_level: str = "ERROR",
    return_on_error: Any = None,
    error_message: str = None
):
    """错误处理装饰器
    
    方法用途：为函数提供统一的异常捕获和处理机制，防止程序因异常而崩溃，
    同时提供友好的错误提示和日志记录功能
    
    参数解释：
        func_name (str, 可选): 函数显示名称，用于日志和UI显示，None则使用实际函数名
        show_in_ui (bool, 可选): 是否在Streamlit界面显示错误信息，默认True
        log_level (str, 可选): 日志级别，支持"ERROR", "WARNING", "INFO", "DEBUG"，默认"ERROR"
        return_on_error (Any, 可选): 发生错误时返回的默认值，默认None
        error_message (str, 可选): 自定义错误消息前缀，None则使用默认消息
        
    返回值：
        Callable: 装饰器函数，返回包装后的函数
        
    使用示例：
        # 基本用法 - 捕获异常并返回None
        >>> @error_handler
        >>> def divide(a, b):
        >>>     return a / b
        >>> 
        >>> result = divide(10, 2)   # 正常执行，返回5.0
        >>> result = divide(10, 0)   # 捕获异常，返回None
        
        # 高级用法 - 自定义错误处理
        >>> @error_handler(
        >>>     func_name="安全除法器",
        >>>     return_on_error=-1,
        >>>     error_message="除法计算失败",
        >>>     show_in_ui=True,
        >>>     log_level="WARNING"
        >>> )
        >>> def safe_divide(a, b):
        >>>     return a / b
        >>> 
        >>> result = safe_divide(10, 0)  # 返回-1，记录警告日志，UI显示错误
    """
    def decorator(func: Callable) -> Callable:
        """错误处理装饰器的内部装饰器函数
        
        方法用途：接收被装饰的函数，返回包装后的函数，在包装函数中添加异常捕获和处理逻辑
        
        参数解释：
            func (Callable): 被装饰的原始函数
            
        返回值：
            Callable: 包装后的函数，具有异常处理功能
            
        使用示例：
            >>> decorated_func = decorator(original_func)
            >>> result = decorated_func(*args, **kwargs)  # 安全执行，异常被捕获
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            """错误处理包装函数
            
            方法用途：包装原始函数，在执行时捕获异常并进行处理，
            记录日志，显示错误信息，返回默认值
            
            参数解释：
                *args: 传递给原函数的位置参数
                **kwargs: 传递给原函数的关键字参数
                
            返回值：
                Any: 正常执行时返回原函数结果，异常时返回return_on_error指定的默认值
                
            使用示例：
                >>> result = wrapper(10, 5)        # 正常执行，返回原函数结果
                >>> result = wrapper(10, 0)        # 捕获异常，返回默认值
                >>> result = wrapper("invalid")  # 捕获异常，返回默认值
                
            异常处理：
                捕获所有Exception及其子类，记录日志，显示UI错误（如启用），
                返回默认值，不会重新抛出异常
            """
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 获取函数名称
                actual_func_name = func_name or func.__name__

                # 构建错误信息
                error_msg = error_message or f"函数 '{actual_func_name}' 执行失败"
                full_error_msg = f"{error_msg}: {str(e)}"

                # 记录日志
                log_func = getattr(logger, log_level.lower(), logger.error)
                log_func(full_error_msg)

                # 记录详细错误信息
                logger.debug(f"错误详情:\n{traceback.format_exc()}")

                # 在UI中显示错误（如果使用Streamlit且处于Streamlit环境中）
                if show_in_ui and hasattr(st, 'error'):
                    try:
                        # 检查是否在Streamlit环境中运行
                        if st._is_running_with_streamlit:
                            st.error(f"❌ {full_error_msg}")
                            
                            # 显示详细错误（在开发模式下）
                            if logger.level <= logging.DEBUG:
                                with st.expander("🔍 查看详细错误信息"):
                                    st.code(traceback.format_exc())
                    except (AttributeError, RuntimeError):
                        # 不在Streamlit环境中，忽略UI显示
                        pass

                # 返回错误时的默认值
                return return_on_error

        return wrapper
    return decorator

def log_execution(
    func_name: str = None,
    log_level: str = "INFO",
    log_args: bool = False,
    log_result: bool = False,
    log_time: bool = True
):
    """执行日志装饰器
    
    方法用途：为函数提供详细的执行日志记录，包括开始时间、执行时间、参数和返回值，
    帮助开发者追踪函数执行过程和调试问题
    
    参数解释：
        func_name (str, 可选): 函数名称，用于日志记录，None则使用实际函数名
        log_level (str, 可选): 日志级别，支持"INFO", "DEBUG", "WARNING", "ERROR"，默认"INFO"
        log_args (bool, 可选): 是否记录函数参数，默认False
        log_result (bool, 可选): 是否记录函数返回值，默认False
        log_time (bool, 可选): 是否记录执行时间，默认True
        
    返回值：
        Callable: 装饰器函数，返回包装后的函数
        
    使用示例：
        # 基本用法 - 记录执行时间和状态
        >>> @log_execution
        >>> def calculate_sum(a, b):
        >>>     return a + b
        >>> 
        >>> result = calculate_sum(5, 3)  # 记录：开始执行、执行完成、耗时
        
        # 高级用法 - 记录详细信息
        >>> @log_execution(
        >>>     func_name="数据处理函数",
        >>>     log_args=True,
        >>>     log_result=True,
        >>>     log_level="DEBUG",
        >>>     log_time=True
        >>> )
        >>> def process_data(items):
        >>>     return [item.upper() for item in items]
        >>> 
        >>> # 记录：开始执行、参数、返回值、执行时间
        >>> result = process_data(['hello', 'world'])
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            """执行日志包装函数
            
            方法用途：包装原始函数，在执行前后记录详细的日志信息，
            包括开始执行、参数、执行时间、返回值等
            
            参数解释：
                *args: 传递给原函数的位置参数
                **kwargs: 传递给原函数的关键字参数
                
            返回值：
                Any: 原函数的返回值，保持原函数的行为不变
                
            使用示例：
                >>> result = wrapper(10, 20, key="value")  # 记录详细执行日志并返回结果
                >>> # 日志输出：开始执行函数、参数、执行完成、返回值、耗时
                
            异常处理：
                如果原函数抛出异常，会记录异常信息并重新抛出，
                保持异常传播链不变，便于上层处理
            """
            # 获取函数名称
            actual_func_name = func_name or func.__name__

            # 获取日志函数
            log_func = getattr(logger, log_level.lower(), logger.info)

            try:
                # 记录函数开始执行
                start_time = time.time()
                log_func(f"开始执行函数: {actual_func_name}")

                # 记录参数（如果启用）
                if log_args:
                    args_str = str(args) if args else ""
                    kwargs_str = str(kwargs) if kwargs else ""
                    log_func(f"函数参数 - args: {args_str}, kwargs: {kwargs_str}")

                # 执行函数
                result = func(*args, **kwargs)

                # 记录执行时间（如果启用）
                if log_time:
                    execution_time = time.time() - start_time
                    log_func(f"函数执行完成: {actual_func_name} (耗时: {execution_time:.3f}秒)")
                else:
                    log_func(f"函数执行完成: {actual_func_name}")

                # 记录返回值（如果启用）
                if log_result:
                    result_str = str(result) if result is not None else "None"
                    # 限制结果字符串长度
                    if len(result_str) > 500:
                        result_str = result_str[:500] + "..."
                    log_func(f"函数返回值: {result_str}")

                return result

            except Exception as e:
                # 记录异常信息
                execution_time = time.time() - start_time if log_time else 0
                error_msg = f"函数执行异常: {actual_func_name}"
                if log_time:
                    error_msg += f" (耗时: {execution_time:.3f}秒)"
                error_msg += f" - {str(e)}"

                logger.error(error_msg)
                logger.debug(f"详细错误信息:\n{traceback.format_exc()}")

                # 重新抛出异常，让上层处理
                raise

        return wrapper
    return decorator

def performance_monitor(
    func_name: str = None,
    warning_threshold: float = 1.0,
    error_threshold: float = 5.0
):
    """性能监控装饰器
    
    方法用途：监控函数的执行时间，根据设定的性能阈值记录不同级别的日志，
    帮助开发者及时发现性能瓶颈和慢查询问题
    
    参数解释：
        func_name (str, 可选): 函数名称，用于日志记录，None则使用实际函数名
        warning_threshold (float, 可选): 警告阈值（秒），超过此时间记录警告日志，默认1.0秒
        error_threshold (float, 可选): 错误阈值（秒），超过此时间记录错误日志，默认5.0秒
        
    返回值：
        Callable: 装饰器函数，返回包装后的函数
        
    使用示例：
        # 基本用法 - 使用默认阈值监控
        >>> @performance_monitor
        >>> def slow_function():
        >>>     time.sleep(0.5)
        >>>     return "完成"
        >>> 
        >>> result = slow_function()  # 记录：性能正常 (耗时: 0.500秒)
        
        # 高级用法 - 自定义性能阈值
        >>> @performance_monitor(
        >>>     func_name="数据库查询",
        >>>     warning_threshold=0.1,    # 100ms警告
        >>>     error_threshold=0.5       # 500ms错误
        >>> )
        >>> def query_database(sql):
        >>>     # 模拟数据库查询
        >>>     time.sleep(0.2)
        >>>     return f"查询结果: {sql}"
        >>> 
        >>> result = query_database("SELECT * FROM users")
        >>> # 记录：性能警告 - 函数执行较慢: 数据库查询 (耗时: 0.200秒)
    """
    def decorator(func: Callable) -> Callable:
            """性能监控装饰器的内部装饰器函数
            
            方法用途：接收被装饰的函数，返回包装后的函数，在包装函数中
            添加性能监控和时间记录功能
            
            参数解释：
                func (Callable): 被装饰的原始函数
                
            返回值：
                Callable: 包装后的函数，具有性能监控功能
                
            使用示例：
                >>> decorated_func = decorator(database_query_func)
                >>> result = decorated_func(sql_query)  # 监控执行性能并返回结果
            """
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                """性能监控包装函数
                
                方法用途：包装原始函数，在执行前后记录性能信息，
                根据执行时间与阈值的比较记录不同级别的日志
                
                参数解释：
                    *args: 传递给原函数的位置参数
                    **kwargs: 传递给原函数的关键字参数
                    
                返回值：
                    Any: 原函数的返回值，保持原函数的行为不变
                    
                使用示例：
                    >>> result = wrapper("query", timeout=30)  # 监控执行性能并返回结果
                    >>> # 日志输出：性能正常/警告/告警 - 函数执行完成/较慢/过慢: 函数名 (耗时: x.xxx秒)
                    
                性能分级：
                    - 正常：执行时间 < warning_threshold，记录INFO级别日志
                    - 警告：warning_threshold ≤ 执行时间 < error_threshold，记录WARNING级别日志  
                    - 告警：执行时间 ≥ error_threshold，记录ERROR级别日志
                    
                异常处理：
                    如果原函数抛出异常，会记录异常信息和执行时间，然后重新抛出异常
                """
                actual_func_name = func_name or func.__name__
                start_time = time.time()

                try:
                    # 执行函数
                    result = func(*args, **kwargs)

                    # 计算执行时间
                    execution_time = time.time() - start_time

                    # 根据执行时间记录不同级别的日志
                    if execution_time >= error_threshold:
                        logger.error(f"性能告警 - 函数执行过慢: {actual_func_name} (耗时: {execution_time:.3f}秒)")
                    elif execution_time >= warning_threshold:
                        logger.warning(f"性能警告 - 函数执行较慢: {actual_func_name} (耗时: {execution_time:.3f}秒)")
                    else:
                        logger.info(f"性能正常 - 函数执行完成: {actual_func_name} (耗时: {execution_time:.3f}秒)")

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.error(f"性能监控 - 函数执行异常: {actual_func_name} (耗时: {execution_time:.3f}秒) - {str(e)}")
                    raise

            return wrapper
    return decorator


if __name__ == "__main__":
    """装饰器测试代码
    
    测试内容：
        1. @error_handler - 异常处理装饰器
        2. @log_execution - 执行日志装饰器  
        3. @performance_monitor - 性能监控装饰器
        4. 组合装饰器使用
        5. 装饰器参数化使用
    
    测试输出：
        - 控制台日志输出
        - 异常处理演示
        - 性能监控结果
    """
    
    import time
    # 配置日志处理器以在控制台显示日志
    console_handler = logging.StreamHandler() # 创建一个 控制台日志处理器，将日志信息输出到 终端/控制台
    console_handler.setLevel(logging.DEBUG) # 设置处理器的 日志级别为DEBUG。告诉处理器"所有级别为DEBUG及以上的日志都要处理"，级别顺序 ：DEBUG < INFO < WARNING < ERROR < CRITICAL
    formatter = logging.Formatter('%(levelname)s - %(message)s') # 创建 日志格式器。输出示例 ： DEBUG - 开始执行函数: 数据处理函数
    console_handler.setFormatter(formatter) # 告诉处理器"按照我定义的格式来显示日志"
    logger.addHandler(console_handler) # 让logger知道"我有一个控制台处理器，可以把日志输出到屏幕"
    logger.setLevel(logging.DEBUG) # 设置logger的 全局日志级别为DEBUG
    
    # 测试 @error_handler 装饰器
    print("=== 测试 @error_handler 装饰器 ===")
    
    @error_handler(func_name="除法计算器:divide_numbers", return_on_error=-1)
    def divide_numbers(a, b):
        """除法函数 - 测试异常处理"""
        return a / b
    
    # 正常调用
    result = divide_numbers(10, 2)
    print(f"10 ÷ 2 = {result}")
    
    # 异常调用（会被捕获并记录）
    result = divide_numbers(10, 0)  # 除零异常被捕获
    print(f"10 ÷ 0 = {result}")
    
    print()
    
    # 测试 @log_execution 装饰器
    print("=== 测试 @log_execution 装饰器 ===")
    
    @log_execution(
        func_name="数据处理函数",
        log_args=True,
        log_result=True,
        log_level="DEBUG",
        log_time=True
    )
    def process_data(items):
        """数据处理函数 - 测试执行日志"""
        return [item.upper() for item in items]
    
    result = process_data(['hello', 'world', 'python'])
    print(f"处理结果: {result}")
    
    print()
    
    # 测试 @performance_monitor 装饰器
    print("=== 测试 @performance_monitor 装饰器 ===")
    
    @performance_monitor(
        func_name="慢速函数",
        warning_threshold=0.1,  # 100ms 警告
        error_threshold=0.3     # 300ms 错误
    )
    def slow_function(delay):
        """慢速函数 - 测试性能监控"""
        time.sleep(delay)
        return f"延迟了 {delay} 秒"
    
    # 正常性能
    result = slow_function(0.05)  # 50ms - 正常
    print(f"结果: {result}")
    
    # 警告性能
    result = slow_function(0.15)  # 150ms - 警告
    print(f"结果: {result}")
    
    # 错误性能
    result = slow_function(0.35)  # 350ms - 错误
    print(f"结果: {result}")
    
    print()
    
    # 测试组合装饰器
    print("=== 测试组合装饰器 ===")
    
    @error_handler()
    @log_execution(
        func_name="组合函数",
        log_args=True,
        log_result=True
    )
    @performance_monitor(
        func_name="组合函数",
        warning_threshold=0.1,
        error_threshold=0.5
    )
    def combined_function(x, y):
        """组合装饰器函数 - 同时具有异常处理、日志记录、性能监控"""
        time.sleep(0.05)  # 50ms 延迟
        return x * y + 100
    
    result = combined_function(5, 8)
    print(f"组合函数结果: {result}")
    
    print()
    
    # 测试装饰器不带参数
    print("=== 测试装饰器不带参数 ===")
    
    @error_handler()
    def simple_error_func():
        """简单错误函数"""
        raise ValueError("测试异常")
    
    @log_execution()
    def simple_log_func(name):
        """简单日志函数"""
        return f"Hello, {name}!"
    
    @performance_monitor()
    def simple_perf_func():
        """简单性能函数"""
        time.sleep(0.02)
        return "快速完成"
    
    # 测试简单错误处理
    try:
        simple_error_func()
    except Exception as e:
        print(f"捕获到异常: {e}")
    
    # 测试简单日志
    result = simple_log_func("Python")
    print(f"简单日志结果: {result}")
    
    # 测试简单性能
    result = simple_perf_func()
    print(f"简单性能结果: {result}")
    
    print("\n=== 所有测试完成 ===")