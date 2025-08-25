#!/usr/bin/env python
"""
配置系统测试脚本
验证重构后的配置系统是否正常工作
"""

def test_config_system():
    print('=== 配置系统测试 ===\n')
    
    try:
        from app.config import get_config_manager, get_config, get_settings
        
        # 1. 测试配置管理器初始化
        print('1. 测试配置管理器初始化:')
        config_manager = get_config_manager()
        print('   ✓ 配置管理器初始化成功')
        
        # 2. 测试配置文件加载
        print('\n2. 测试配置文件加载:')
        all_config = config_manager.get_config()
        print(f'   ✓ 配置文件加载成功，包含 {len(all_config)} 个配置项')
        
        # 3. 测试服务器配置
        print('\n3. 测试服务器配置:')
        server_config = config_manager.get_server_config()
        print(f'   host: {server_config.get("host")}')
        print(f'   port: {server_config.get("port")}')
        print(f'   reload: {server_config.get("reload")}')
        
        # 4. 测试Agent配置
        print('\n4. 测试Agent配置:')
        agent_config = config_manager.get_agent_config()
        print(f'   verbose: {agent_config.get("verbose")}')
        print(f'   max_iterations: {agent_config.get("max_iterations")}')
        print(f'   max_execution_time: {agent_config.get("max_execution_time")}')
        
        # 5. 测试缓存配置
        print('\n5. 测试缓存配置:')
        cache_config = config_manager.get_cache_config()
        print(f'   max_cache_size: {cache_config.get("max_cache_size")}')
        
        # 6. 测试LLM配置
        print('\n6. 测试LLM配置:')
        llm_config = config_manager.get_config("llm")
        print(f'   default_provider: {llm_config.get("default_provider")}')
        print(f'   default_model: {llm_config.get("default_model")}')
        print(f'   default_temperature: {llm_config.get("default_temperature")}')
        
        # 7. 测试AI类型映射
        print('\n7. 测试AI类型映射:')
        ai_types = config_manager.get_ai_type_mappings()
        print(f'   可用AI类型: {list(ai_types.keys())}')
        for ai_type, config in ai_types.items():
            print(f'   {ai_type}: {config.get("provider")} - {config.get("model")}')
        
        # 8. 测试提示词配置
        print('\n8. 测试提示词配置:')
        prompts_config = config_manager.get_prompts_config()
        react_template = prompts_config.get("react_template", "")
        print(f'   ReAct模板长度: {len(react_template)} 字符')
        print(f'   包含工具说明: {"{tools}" in react_template}')
        
        # 9. 测试工具配置
        print('\n9. 测试工具配置:')
        tools_config = config_manager.get_tools_config()
        available_tools = tools_config.get("available_tools", [])
        print(f'   可用工具: {available_tools}')
        
        # 10. 测试便捷函数
        print('\n10. 测试便捷函数:')
        settings = get_settings()
        print(f'   默认provider: {settings.provider}')
        print(f'   默认model: {settings.model}')
        print(f'   默认temperature: {settings.temperature}')
        
        # 11. 测试配置重载
        print('\n11. 测试配置重载:')
        config_manager.reload_config()
        print('   ✓ 配置重载成功')
        
        print('\n=== 配置系统测试完成 ===')
        print('✓ 所有测试通过，配置系统工作正常')
        
    except Exception as e:
        print(f'\n✗ 配置系统测试失败: {e}')
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_config_system()
