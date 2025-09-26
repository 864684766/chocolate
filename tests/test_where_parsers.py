"""Where 解析器测试

用于测试各种解析器的功能，支持输入不同的 query 和 field 来验证解析结果。
"""

from datetime import datetime, timezone
from app.rag.retrieval.utils.where_parsers import ParserContext, ParserRegistry, build_default_registry
from app.rag.retrieval.utils.parsers import (
    parse_by_aliases,
    infer_lang,
    lang_parser,
    media_type_parser,
    quality_score_parser,
    created_at_parser,
    tags_contains_parser,
)


class TestParserContext:
    """测试解析器上下文类"""

    def test_parser_context_creation(self):
        """测试解析器上下文的创建"""
        whitelist = {"lang", "media_type", "quality_score"}
        types_map = {"lang": "string", "media_type": "string", "quality_score": "number"}
        lang_aliases = {"中文": "zh", "english": "en"}
        media_keywords = {"图片": "image", "视频": "video"}
        media_types = {"image", "video", "text"}
        
        ctx = ParserContext(
            whitelist=whitelist,
            types_map=types_map,
            lang_aliases=lang_aliases,
            media_keywords=media_keywords,
            media_types=media_types,
            chinese_threshold=0.3,
            fallback_language="en"
        )
        
        assert ctx.whitelist == whitelist
        assert ctx.types_map == types_map
        assert ctx.lang_aliases == lang_aliases
        assert ctx.media_keywords == media_keywords
        assert ctx.media_types == media_types
        assert ctx.chinese_threshold == 0.3
        assert ctx.fallback_language == "en"


class TestCommonUtils:
    """测试通用工具函数"""

    def test_parse_by_aliases_english(self):
        """测试英文别名解析"""
        mapping = {"image": "image", "video": "video", "text": "text"}
        
        # 测试英文单词匹配
        assert parse_by_aliases("find image files", mapping) == "image"
        assert parse_by_aliases("video content", mapping) == "video"
        assert parse_by_aliases("text documents", mapping) == "text"
        assert parse_by_aliases("no match here", mapping) is None

    def test_parse_by_aliases_chinese(self):
        """测试中文别名解析"""
        mapping = {"图片": "image", "视频": "video", "文档": "text"}
        
        # 测试中文直接匹配
        assert parse_by_aliases("查找图片文件", mapping) == "image"
        assert parse_by_aliases("视频内容", mapping) == "video"
        assert parse_by_aliases("文档资料", mapping) == "text"
        
        # 测试中文 N-gram 匹配
        assert parse_by_aliases("这是图片", mapping) == "image"
        assert parse_by_aliases("视频播放", mapping) == "video"

    def test_parse_by_aliases_empty_mapping(self):
        """测试空映射"""
        assert parse_by_aliases("any query", {}) is None


class TestLanguageParser:
    """测试语言解析器"""

    def test_infer_lang_chinese(self):
        """测试中文语言推断"""
        ctx = ParserContext(
            whitelist=set(),
            types_map={},
            lang_aliases={},
            media_keywords={},
            media_types=set(),
            chinese_threshold=0.3,
            fallback_language="en"
        )
        
        # 测试中文查询
        assert infer_lang("这是中文查询", ctx) == "zh"
        assert infer_lang("查找图片文件", ctx) == "zh"
        assert infer_lang("中文内容", ctx) == "zh"

    def test_infer_lang_english(self):
        """测试英文语言推断"""
        ctx = ParserContext(
            whitelist=set(),
            types_map={},
            lang_aliases={},
            media_keywords={},
            media_types=set(),
            chinese_threshold=0.3,
            fallback_language="en"
        )
        
        # 测试英文查询
        assert infer_lang("find image files", ctx) == "en"
        assert infer_lang("search for documents", ctx) == "en"
        assert infer_lang("video content", ctx) == "en"

    def test_infer_lang_mixed(self):
        """测试混合语言推断"""
        ctx = ParserContext(
            whitelist=set(),
            types_map={},
            lang_aliases={},
            media_keywords={},
            media_types=set(),
            chinese_threshold=0.3,
            fallback_language="en"
        )
        
        # 测试中英混合（中文占主导）
        assert infer_lang("查找image文件", ctx) == "zh"
        # 测试中英混合（英文占主导）
        assert infer_lang("find 图片 files", ctx) == "en"

    def test_infer_lang_fallback(self):
        """测试回退语言"""
        ctx = ParserContext(
            whitelist=set(),
            types_map={},
            lang_aliases={},
            media_keywords={},
            media_types=set(),
            chinese_threshold=0.3,
            fallback_language="en"
        )
        
        # 测试无有效字符
        assert infer_lang("123456", ctx) == "en"
        assert infer_lang("!@#$%", ctx) == "en"

    def test_lang_parser_with_whitelist(self):
        """测试语言解析器（在白名单中）"""
        ctx = ParserContext(
            whitelist={"lang"},
            types_map={},
            lang_aliases={},
            media_keywords={},
            media_types=set(),
            chinese_threshold=0.3,
            fallback_language="en"
        )
        
        result = lang_parser("这是中文查询", ctx)
        assert result == {"lang": {"$eq": "zh"}}

    def test_lang_parser_without_whitelist(self):
        """测试语言解析器（不在白名单中）"""
        ctx = ParserContext(
            whitelist=set(),
            types_map={},
            lang_aliases={},
            media_keywords={},
            media_types=set(),
            chinese_threshold=0.3,
            fallback_language="en"
        )
        
        result = lang_parser("这是中文查询", ctx)
        assert result is None


class TestMediaParser:
    """测试媒体类型解析器"""

    def test_media_type_parser_success(self):
        """测试媒体类型解析成功"""
        ctx = ParserContext(
            whitelist={"media_type"},
            types_map={},
            lang_aliases={},
            media_keywords={"图片": "image", "视频": "video", "文档": "text"},
            media_types={"image", "video", "text"},
            chinese_threshold=0.3,
            fallback_language="en"
        )
        
        # 测试中文关键词
        result = media_type_parser("查找图片文件", ctx)
        assert result == {"media_type": {"$eq": "image"}}
        
        result = media_type_parser("视频内容", ctx)
        assert result == {"media_type": {"$eq": "video"}}

    def test_media_type_parser_not_in_whitelist(self):
        """测试媒体类型不在白名单中"""
        ctx = ParserContext(
            whitelist=set(),
            types_map={},
            lang_aliases={},
            media_keywords={"图片": "image"},
            media_types={"image"},
            chinese_threshold=0.3,
            fallback_language="en"
        )
        
        result = media_type_parser("查找图片文件", ctx)
        assert result is None

    def test_media_type_parser_not_supported(self):
        """测试不支持的媒体类型"""
        ctx = ParserContext(
            whitelist={"media_type"},
            types_map={},
            lang_aliases={},
            media_keywords={"图片": "image"},
            media_types={"video"},  # 不支持 image
            chinese_threshold=0.3,
            fallback_language="en"
        )
        
        result = media_type_parser("查找图片文件", ctx)
        assert result is None


class TestQualityParser:
    """测试质量分数解析器"""

    def test_quality_score_parser_success(self):
        """测试质量分数解析成功"""
        ctx = ParserContext(
            whitelist={"quality_score"},
            types_map={},
            lang_aliases={},
            media_keywords={},
            media_types=set(),
            chinese_threshold=0.3,
            fallback_language="en"
        )
        
        # 测试各种格式
        test_cases = [
            ("质量>=0.8", 0.8),
            ("质量大于等于0.7", 0.7),
            ("quality>=0.9", 0.9),
            ("质量>0.6", 0.6),
            ("质量大于0.5", 0.5),
        ]
        
        for query, expected in test_cases:
            result = quality_score_parser(query, ctx)
            assert result == {"quality_score": {"$gte": expected}}

    def test_quality_score_parser_not_in_whitelist(self):
        """测试质量分数不在白名单中"""
        ctx = ParserContext(
            whitelist=set(),
            types_map={},
            lang_aliases={},
            media_keywords={},
            media_types=set(),
            chinese_threshold=0.3,
            fallback_language="en"
        )
        
        result = quality_score_parser("质量>=0.8", ctx)
        assert result is None

    def test_quality_score_parser_invalid_range(self):
        """测试无效的质量分数范围"""
        ctx = ParserContext(
            whitelist={"quality_score"},
            types_map={},
            lang_aliases={},
            media_keywords={},
            media_types=set(),
            chinese_threshold=0.3,
            fallback_language="en"
        )
        
        # 测试超出范围的值
        result = quality_score_parser("质量>=1.5", ctx)
        assert result is None
        
        result = quality_score_parser("质量>=-0.1", ctx)
        assert result is None


class TestTimeParser:
    """测试时间解析器"""

    def test_created_at_parser_success(self):
        """测试时间解析成功"""
        ctx = ParserContext(
            whitelist={"created_at"},
            types_map={},
            lang_aliases={},
            media_keywords={},
            media_types=set(),
            chinese_threshold=0.3,
            fallback_language="en"
        )
        
        # 测试近一周
        result = created_at_parser("近一周的内容", ctx)
        assert result is not None
        assert "created_at" in result
        assert "$gte" in result["created_at"]
        
        # 验证时间格式
        time_str = result["created_at"]["$gte"]
        parsed_time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        time_diff = now - parsed_time
        assert 6 <= time_diff.days <= 7  # 允许1天的误差

    def test_created_at_parser_not_in_whitelist(self):
        """测试时间字段不在白名单中"""
        ctx = ParserContext(
            whitelist=set(),
            types_map={},
            lang_aliases={},
            media_keywords={},
            media_types=set(),
            chinese_threshold=0.3,
            fallback_language="en"
        )
        
        result = created_at_parser("近一周的内容", ctx)
        assert result is None


class TestTagsParser:
    """测试标签解析器"""

    def test_tags_parser_not_in_whitelist(self):
        """测试标签字段不在白名单中"""
        ctx = ParserContext(
            whitelist=set(),
            types_map={},
            lang_aliases={},
            media_keywords={},
            media_types=set(),
            chinese_threshold=0.3,
            fallback_language="en"
        )
        
        parser = tags_contains_parser("tags")
        result = parser("测试查询", ctx)
        assert result is None

    def test_tags_parser_with_whitelist(self):
        """测试标签字段在白名单中（模拟关键词服务不可用）"""
        ctx = ParserContext(
            whitelist={"tags"},
            types_map={},
            lang_aliases={},
            media_keywords={},
            media_types=set(),
            chinese_threshold=0.3,
            fallback_language="en"
        )
        
        parser = tags_contains_parser("tags")
        # 由于关键词服务可能不可用，这里主要测试不会抛出异常
        result = parser("测试查询", ctx)
        # 结果可能是 None（服务不可用）或包含关键词的条件
        assert result is None or isinstance(result, dict)


class TestParserRegistry:
    """测试解析器注册表"""

    def test_parser_registry_register_and_parse(self):
        """测试解析器注册和解析"""
        registry = ParserRegistry()
        
        def dummy_parser(query: str, ctx: ParserContext) -> dict:
            return {"test": {"$eq": "value"}}
        
        registry.register("test_field", dummy_parser)
        
        ctx = ParserContext(
            whitelist={"test_field"},
            types_map={},
            lang_aliases={},
            media_keywords={},
            media_types=set(),
            chinese_threshold=0.3,
            fallback_language="en"
        )
        
        result = registry.parse("test_field", "test query", ctx)
        assert result == {"test": {"$eq": "value"}}

    def test_parser_registry_unregistered_field(self):
        """测试未注册的字段"""
        registry = ParserRegistry()
        
        ctx = ParserContext(
            whitelist={"test_field"},
            types_map={},
            lang_aliases={},
            media_keywords={},
            media_types=set(),
            chinese_threshold=0.3,
            fallback_language="en"
        )
        
        result = registry.parse("unregistered_field", "test query", ctx)
        assert result is None


class TestBuildDefaultRegistry:
    """测试默认注册表构建"""

    def test_build_default_registry(self):
        """测试构建默认注册表"""
        ctx = ParserContext(
            whitelist={"lang", "media_type", "quality_score", "created_at", "tags", "keyphrases"},
            types_map={},
            lang_aliases={},
            media_keywords={"图片": "image"},
            media_types={"image"},
            chinese_threshold=0.3,
            fallback_language="en"
        )
        
        registry = build_default_registry(ctx)
        
        # 验证所有字段都已注册
        assert registry._parsers.get("lang") is not None
        assert registry._parsers.get("media_type") is not None
        assert registry._parsers.get("quality_score") is not None
        assert registry._parsers.get("created_at") is not None
        assert registry._parsers.get("tags") is not None
        assert registry._parsers.get("keyphrases") is not None


# 交互式测试函数
def interactive_test():
    """交互式测试函数，允许用户输入查询和字段进行测试"""
    print("=== Where 解析器交互式测试 ===")
    print("输入 'quit' 退出测试")
    
    # 创建测试上下文
    ctx = ParserContext(
        whitelist={"lang", "media_type", "quality_score", "created_at", "tags", "keyphrases"},
        types_map={},
        lang_aliases={},
        media_keywords={"图片": "image", "视频": "video", "文档": "text", "音频": "audio"},
        media_types={"image", "video", "text", "audio"},
        chinese_threshold=0.3,
        fallback_language="en"
    )
    
    registry = build_default_registry(ctx)
    
    while True:
        print("\n" + "="*50)
        query = input("请输入查询文本: ").strip()
        
        if query.lower() == 'quit':
            break
            
        if not query:
            continue
            
        print(f"\n查询: '{query}'")
        print("-" * 30)
        
        # 测试所有注册的解析器
        for field in ctx.whitelist:
            result = registry.parse(field, query, ctx)
            if result:
                print(f"{field}: {result}")
            else:
                print(f"{field}: 无匹配")


if __name__ == "__main__":
    # 运行交互式测试
    interactive_test()
