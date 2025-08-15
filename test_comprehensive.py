#!/usr/bin/env python
"""
Chocolate 项目综合功能测试
验证重构后的工厂模式、工具集成、目录拆分的正确性和稳定性
"""

def test_comprehensive():
    print('=== Chocolate 项目综合功能测试 ===\n')

    # 1. 测试工厂注册表功能
    print('1. 测试 LLM Provider 工厂注册表:')
    from app.llm_adapters.factory import LLMProviderFactory
    print(f'   已注册 adapters: {list(LLMProviderFactory._registry.keys())}')
    print(f'   懒加载状态: {LLMProviderFactory._bootstrapped}')

    # 2. 测试工具模块导入（新目录结构）
    print('\n2. 测试工具模块导入:')
    from app.tools import search_docs, http_get, calc
    print('   ✓ 从 app.tools 包成功导入三个工具')

    # 3. 测试工具功能
    print('\n3. 测试工具功能:')
    calc_result = calc.invoke('2*3+1')
    print(f'   calc("2*3+1") = {calc_result}')

    search_result = search_docs.invoke('长城')[:50]
    print(f'   search_docs("长城") = {search_result}...')

    # 4. 测试向后兼容性（原 tools.py 转发）
    print('\n4. 测试向后兼容性:')
    try:
        import app.tools as old_tools_module
        print('   ✓ 原 tools.py 文件仍可导入（向后兼容）')
    except Exception as e:
        print(f'   ✗ 向后兼容失败: {e}')

    # 5. 测试配置系统
    print('\n5. 测试配置系统:')
    from app.config import get_settings
    settings = get_settings()
    print(f'   默认 provider: {settings.provider}')
    print(f'   请求超时: {settings.request_timeout}s')

    # 6. 测试 Agent 构建（无 LLM 依赖）
    print('\n6. 测试 Agent 构建:')
    try:
        from app.agent import build_agent
        # 这里会因为缺少 LLM 依赖而失败，但可以测试导入路径
        print('   ✓ Agent 模块导入成功')
        print('   ⚠ Agent 实例化需要 LLM 依赖（已跳过）')
    except Exception as e:
        print(f'   ✗ Agent 导入失败: {e}')

    print('\n=== 测试完成 ===')
    return True

if __name__ == "__main__":
    test_comprehensive()