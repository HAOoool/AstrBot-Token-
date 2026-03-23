# main.py
import asyncio
import json
import os
import shutil
import tiktoken
import yaml
from typing import Dict, List, Any, Union, Optional
from collections import Counter
from datetime import datetime, time, timedelta
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api.provider import ProviderRequest
from astrbot.api import logger

import aiohttp
from aiohttp import web

LOG_PREFIX = "[✨Token统计✨] "

# 会话消息最大保留数量
MAX_SESSION_MESSAGES = 100


@register(
    "astrbot_plugin_token_stats",
    "your_name",
    "自动统计每次 LLM 请求的 token 消耗，支持 /tokenstats 命令查询，按会话区分统计，输入输出均采用 API 返回的真实值",
    "3.9.15",
    "https://github.com/yourname/astrbot_plugin_token_stats"
)
class TokenStatsPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.config = self._load_config()
        self.plugin_rules = self.config.get("plugins", [])
        self.session_stats: Dict[str, Dict[str, int]] = {}
        self.session_messages: Dict[str, List[Dict]] = {}

        self.daily_stats: Dict[str, Dict[str, Dict[str, int]]] = {}
        self.current_date = datetime.now().date()
        self.current_session_today_counter: Dict[str, Dict[str, int]] = {}
        self._daily_lock = asyncio.Lock()
        self._daily_task = None

        self._load_daily_stats()
        self._daily_task = asyncio.create_task(self._daily_reset_loop())

        self.web_app = None
        self.web_runner = None
        self.web_task = asyncio.create_task(self._start_web_server())

    # ---------- 配置加载 ----------
    def _load_config(self) -> dict:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                    if config is None:
                        config = {}
                    if "plugins" not in config or config["plugins"] is None:
                        config["plugins"] = []
                    # 读取统一统计开关（目前保留供未来使用）
                    self.use_unified_stats = config.get("use_unified_stats", False)
                    return config
            except Exception as e:
                logger.error(f"{LOG_PREFIX}加载配置文件失败: {e}")
                return {"plugins": []}
        return {"plugins": []}

    # ---------- 文本提取与 Token 计数 ----------
    def extract_text(self, content: Union[str, List, Dict, None]) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for item in content:
                texts.append(self.extract_text(item))
            return " ".join(texts)
        if isinstance(content, dict):
            if content.get("type") == "text":
                return content.get("text", "")
            return self.extract_text(content.get("text") or content.get("content"))
        return str(content)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.error(f"{LOG_PREFIX}Token计数失败: {e}")
            return 0

    def count_messages_tokens(self, messages: List[Dict]) -> int:
        total = 0
        for msg in messages:
            total += 4
            content_text = self.extract_text(msg.get("content"))
            total += self.count_tokens(content_text)
            if msg.get("name"):
                total += self.count_tokens(msg["name"])
        total += 2
        return total

    # ---------- 消息分类与统计 ----------
    def classify_messages(self, messages: List[Dict]) -> Dict[str, List[Dict]]:
        result = {"persona": [], "context": [], "unclassified": []}
        
        # 初始化插件分类列表
        for rule in self.plugin_rules:
            plugin_name = rule["name"]
            result[f"plugin_{plugin_name}"] = []

        for msg in messages:
            matched = False
            
            # 先尝试匹配插件规则（包括系统消息）
            for rule in self.plugin_rules:
                plugin_name = rule["name"]
                match_criteria = rule.get("match", {})
                if self._matches_rule(msg, match_criteria):
                    result[f"plugin_{plugin_name}"].append(msg)
                    matched = True
                    break
            
            # 如果没有匹配任何规则，按角色分类
            if not matched:
                if msg.get("role") == "system":
                    result["persona"].append(msg)
                else:
                    result["context"].append(msg)
        
        return result

    def _matches_rule(self, msg: Dict, criteria: Dict) -> bool:
        if "role" in criteria and msg.get("role") != criteria["role"]:
            return False
        if "name_contains" in criteria:
            name = msg.get("name") or ""  # 修复：防止 name 为 None
            if criteria["name_contains"] not in name:
                return False
        content_text = self.extract_text(msg.get("content"))
        if "content_startswith" in criteria:
            if not content_text.startswith(criteria["content_startswith"]):
                return False
        if "content_contains" in criteria:
            if criteria["content_contains"] not in content_text:
                return False
        return True

    def calculate_stats(self, classified: Dict[str, List[Dict]]) -> Dict[str, int]:
        stats = {}
        for key, msgs in classified.items():
            if key == "persona":
                stats["persona_tokens"] = self.count_messages_tokens(msgs)
            elif key == "context":
                stats["context_tokens"] = self.count_messages_tokens(msgs)
            elif key.startswith("plugin_"):
                plugin_name = key[7:]
                stats[f"plugin_{plugin_name}_tokens"] = self.count_messages_tokens(msgs)
            else:
                stats["unclassified_tokens"] = stats.get("unclassified_tokens", 0) + self.count_messages_tokens(msgs)
        stats["total_tokens"] = sum(v for k, v in stats.items() if k.endswith("_tokens"))
        return stats

    # ---------- 每日统计持久化 ----------
    def _get_stats_file_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), "daily_stats.json")

    def _load_daily_stats(self):
        path = self._get_stats_file_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.daily_stats = json.load(f)
            except Exception as e:
                logger.error(f"{LOG_PREFIX}加载每日统计数据失败: {e}")
                self.daily_stats = {}

    def _save_daily_stats(self):
        path = self._get_stats_file_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.daily_stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"{LOG_PREFIX}保存每日统计数据失败: {e}")

    async def _add_daily_tokens(self, session_id: str, stats: Dict[str, int]):
        if not stats:
            return
        async with self._daily_lock:
            today = datetime.now().date()
            if today != self.current_date:
                await self._migrate_current_day_to_history()
                self.current_date = today
                self.current_session_today_counter.clear()

            if session_id not in self.current_session_today_counter:
                self.current_session_today_counter[session_id] = {}
            today_counter = self.current_session_today_counter[session_id]

            for key, value in stats.items():
                if key.endswith("_tokens"):
                    today_counter[key] = today_counter.get(key, 0) + value
            if "total_tokens" in stats:
                today_counter["total_tokens"] = today_counter.get("total_tokens", 0) + stats["total_tokens"]

    async def _add_daily_io_tokens(self, session_id: str, input_tokens: int, output_tokens: int):
        if not input_tokens and not output_tokens:
            return
        async with self._daily_lock:
            today = datetime.now().date()
            if today != self.current_date:
                await self._migrate_current_day_to_history()
                self.current_date = today
                self.current_session_today_counter.clear()

            if session_id not in self.current_session_today_counter:
                self.current_session_today_counter[session_id] = {}
            today_counter = self.current_session_today_counter[session_id]
            today_counter["input_tokens"] = today_counter.get("input_tokens", 0) + input_tokens
            today_counter["output_tokens"] = today_counter.get("output_tokens", 0) + output_tokens

    async def _migrate_current_day_to_history(self):
        today_str = self.current_date.isoformat()
        for session_id, counters in self.current_session_today_counter.items():
            if not counters:
                continue
            if session_id not in self.daily_stats:
                self.daily_stats[session_id] = {}
            day_stats = self.daily_stats[session_id]
            if today_str not in day_stats:
                day_stats[today_str] = {}
            for cat, val in counters.items():
                day_stats[today_str][cat] = day_stats[today_str].get(cat, 0) + val
        self._save_daily_stats()
        self.current_session_today_counter.clear()

    async def _daily_reset_loop(self):
        while True:
            try:
                now = datetime.now()
                next_midnight = datetime.combine(now.date() + timedelta(days=1), time(0, 0))
                seconds_to_wait = (next_midnight - now).total_seconds()
                await asyncio.sleep(seconds_to_wait)

                async with self._daily_lock:
                    await self._migrate_current_day_to_history()
                    self.current_date = datetime.now().date()
                    self.current_session_today_counter.clear()
                    logger.info(f"{LOG_PREFIX}每日统计已重置，历史数据已保存")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"{LOG_PREFIX}每日重置循环出错: {e}")
                # 出错后等待一小时再重试
                await asyncio.sleep(3600)

    # ---------- 消息规范化 ----------
    def _normalize_messages(self, messages: List) -> List[Dict]:
        normalized = []
        for msg in messages:
            if isinstance(msg, dict):
                normalized.append(msg)
            elif isinstance(msg, str):
                normalized.append({"role": "user", "content": msg})
            else:
                # 尝试从非字典对象中提取 role 和 content
                try:
                    role = getattr(msg, 'role', None)
                    content = getattr(msg, 'content', None)
                    if role is not None and content is not None:
                        normalized.append({"role": role, "content": content})
                    else:
                        logger.warning(f"{LOG_PREFIX}未知的消息格式: {type(msg)}")
                except Exception:
                    logger.warning(f"{LOG_PREFIX}无法处理的消息对象: {type(msg)}")
        return normalized

    # ---------- 辅助方法：获取完整会话 ID ----------
    def _get_full_session_id(self, event: AstrMessageEvent) -> str:
        if hasattr(event, 'unified_msg_origin') and event.unified_msg_origin:
            sid = event.unified_msg_origin
            logger.debug(f"{LOG_PREFIX}从 unified_msg_origin 获取到会话ID: {sid}")
            return sid
        if hasattr(event, 'get_session_id'):
            sid = event.get_session_id()
            if sid:
                logger.debug(f"{LOG_PREFIX}从 get_session_id 获取到会话ID: {sid}")
                return sid
        if hasattr(event, 'session_id') and event.session_id:
            sid = event.session_id
            logger.debug(f"{LOG_PREFIX}从 session_id 属性获取到会话ID: {sid}")
            return sid
        logger.warning(f"{LOG_PREFIX}无法获取会话ID，使用默认值 unknown")
        return "unknown"

    # ---------- 事件处理 ----------
    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        session_id = self._get_full_session_id(event)

        messages = None
        if hasattr(req, 'contexts') and req.contexts:
            messages = list(req.contexts)
            logger.info(f"{LOG_PREFIX}[{session_id}] 使用 req.contexts 获取到 {len(messages)} 条消息")
        elif hasattr(req, 'conversation') and req.conversation:
            # 修复：添加 None 检查
            if hasattr(req.conversation, 'history') and req.conversation.history is not None:
                raw_messages = list(req.conversation.history)
                messages = self._normalize_messages(raw_messages)
                logger.info(f"{LOG_PREFIX}[{session_id}] 使用 req.conversation.history 获取到 {len(messages)} 条消息")
            else:
                messages = []
                logger.warning(f"{LOG_PREFIX}[{session_id}] req.conversation.history 为空")
        else:
            logger.error(f"{LOG_PREFIX}[{session_id}] 无法从 req 获取消息列表")
            return

        if hasattr(req, 'system_prompt') and req.system_prompt:
            system_msg = {"role": "system", "content": req.system_prompt}
            if not any(msg.get('role') == 'system' for msg in messages):
                messages.insert(0, system_msg)
                logger.info(f"{LOG_PREFIX}[{session_id}] 已添加独立的 system_prompt 到消息列表")

        self.session_messages[session_id] = messages
        # 限制消息数量
        if len(self.session_messages[session_id]) > MAX_SESSION_MESSAGES:
            self.session_messages[session_id] = self.session_messages[session_id][-MAX_SESSION_MESSAGES:]

        logger.info(f"{LOG_PREFIX}[{session_id}] ===== 消息 Token 明细 =====")
        for idx, msg in enumerate(messages):
            content_text = self.extract_text(msg.get("content"))
            tokens = self.count_tokens(content_text) + (4 if msg.get("content") else 0)
            if msg.get("name"):
                tokens += self.count_tokens(msg["name"])
            logger.info(f"{LOG_PREFIX}[{session_id}] 消息[{idx}]")
            logger.info(f"{LOG_PREFIX}[{session_id}]   role (角色): {msg.get('role')}")
            logger.info(f"{LOG_PREFIX}[{session_id}]   name (名称): {msg.get('name')}")
            logger.info(f"{LOG_PREFIX}[{session_id}]   tokens (token数): {tokens}")
            logger.info(f"{LOG_PREFIX}[{session_id}]   content (内容): {content_text}")

        if not messages:
            logger.warning(f"{LOG_PREFIX}[{session_id}] 消息列表为空")
            return

        classified = self.classify_messages(messages)
        stats = self.calculate_stats(classified)
        self.session_stats[session_id] = stats
        await self._add_daily_tokens(session_id, stats)

        logger.info(f"{LOG_PREFIX}[{session_id}] Token统计详情:")
        for key, value in stats.items():
            if value > 0:
                logger.info(f"  {key}: {value}")

    @filter.on_llm_response(priority=-90)
    async def on_llm_response(self, event: AstrMessageEvent, response):
        """捕获 LLM 响应，从 API 获取真实的 prompt_tokens 和 completion_tokens"""
        session_id = self._get_full_session_id(event)

        prompt_tokens = 0
        completion_tokens = 0

        # 1. 尝试从 raw_completion 获取 usage（最接近原始 API 响应）
        if hasattr(response, 'raw_completion') and response.raw_completion:
            raw = response.raw_completion
            if hasattr(raw, 'usage') and raw.usage:
                usage = raw.usage
                if isinstance(usage, dict):
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                else:
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                    completion_tokens = getattr(usage, 'completion_tokens', 0)

        # 2. 如果还没拿到，尝试从 response.usage
        if prompt_tokens == 0 and hasattr(response, 'usage') and response.usage:
            usage = response.usage
            if isinstance(usage, dict):
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
            else:
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)

        # 3. 尝试从 response.completion.usage
        if prompt_tokens == 0 and hasattr(response, 'completion') and response.completion:
            completion_obj = response.completion
            if hasattr(completion_obj, 'usage') and completion_obj.usage:
                usage = completion_obj.usage
                if isinstance(usage, dict):
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                else:
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                    completion_tokens = getattr(usage, 'completion_tokens', 0)

        # 4. 直接从 response 属性（某些包装直接暴露）
        if prompt_tokens == 0:
            for attr in ['prompt_tokens', 'input_tokens']:
                if hasattr(response, attr):
                    prompt_tokens = getattr(response, attr)
                    break
        if completion_tokens == 0:
            for attr in ['completion_tokens', 'output_tokens', 'response_tokens']:
                if hasattr(response, attr):
                    completion_tokens = getattr(response, attr)
                    break

        # 累加到会话统计
        if session_id not in self.session_stats:
            self.session_stats[session_id] = {}
        stats = self.session_stats[session_id]
        stats["input_tokens"] = stats.get("input_tokens", 0) + prompt_tokens
        stats["output_tokens"] = stats.get("output_tokens", 0) + completion_tokens
        await self._add_daily_io_tokens(session_id, prompt_tokens, completion_tokens)
        logger.debug(f"{LOG_PREFIX}[{session_id}] API 返回: 输入 {prompt_tokens}, 输出 {completion_tokens}")

    @filter.after_message_sent(priority=-90)
    async def after_message_sent(self, event: AstrMessageEvent):
        session_id = self._get_full_session_id(event)
        if session_id not in self.session_messages:
            return
        reply_text = None
        if hasattr(event, 'message_result') and event.message_result:
            if isinstance(event.message_result, list):
                for msg in event.message_result:
                    if isinstance(msg, dict) and msg.get('type') == 'text':
                        reply_text = msg.get('data', {}).get('text')
                        if reply_text:
                            break
            elif isinstance(event.message_result, str):
                reply_text = event.message_result
            else:
                reply_text = str(event.message_result)
        if not reply_text and hasattr(event, 'get_result'):
            result = event.get_result()
            if result:
                reply_text = str(result)
        if not reply_text:
            return
        assistant_msg = {"role": "assistant", "content": reply_text}
        if self.session_messages[session_id] and self.session_messages[session_id][-1].get('role') == 'assistant':
            last_content = self.session_messages[session_id][-1].get('content', '')
            if last_content == reply_text:
                return
        self.session_messages[session_id].append(assistant_msg)
        # 限制消息数量
        if len(self.session_messages[session_id]) > MAX_SESSION_MESSAGES:
            self.session_messages[session_id] = self.session_messages[session_id][-MAX_SESSION_MESSAGES:]
        logger.debug(f"{LOG_PREFIX}[{session_id}] 已追加机器人回复到消息记录")

    # ---------- 命令 ----------
    @filter.command("tokenstats")
    async def show_token_stats(self, event: AstrMessageEvent):
        raw_text = event.message_str
        text = raw_text.strip()
        if text.startswith('/'):
            text = text[1:]
        logger.info(f"{LOG_PREFIX}收到命令: {repr(text)}")
        parts = text.split()
        if not parts or parts[0] != 'tokenstats':
            return

        if len(parts) == 1:
            session_id = self._get_full_session_id(event)
            stats = self.session_stats.get(session_id)
            if not stats:
                yield event.plain_result(f"当前会话（{session_id}）暂无统计数据。请先发送一条消息触发 LLM 请求。")
                return
            lines = ["📊 **Token消耗统计**", f"会话: {session_id}", "━━━━━━━━━━━━━━━━━━"]
            if "input_tokens" in stats:
                lines.append(f"📥 输入: {stats['input_tokens']} tokens")
            if "output_tokens" in stats:
                lines.append(f"📤 输出: {stats['output_tokens']} tokens")
            lines.append("━━━━━━━━━━━━━━━━━━")
            if "persona_tokens" in stats:
                lines.append(f"🧠 人格提示词: {stats['persona_tokens']} tokens")
            if "context_tokens" in stats:
                lines.append(f"💬 上下文对话: {stats['context_tokens']} tokens")
            plugin_keys = [k for k in stats.keys() if k.startswith("plugin_") and k.endswith("_tokens")]
            for pk in plugin_keys:
                plugin_name = pk[7:-7]
                lines.append(f"🔌 {plugin_name}: {stats[pk]} tokens")
            if "unclassified_tokens" in stats:
                lines.append(f"❓ 未归类: {stats['unclassified_tokens']} tokens")
            lines.append("━━━━━━━━━━━━━━━━━━")
            # 添加注释
            lines.append("💡 注：输入/输出为 API 返回的真实值（作为计费参考），分类统计为插件计算值（用于分析消耗构成），二者可能不一致。分类统计仅计算输入，因为输出仅包含API返回的消息，无需分类")
            yield event.plain_result("\n".join(lines))
            return

        subcommand = parts[1] if len(parts) > 1 else ''
        if subcommand == 'suggest':
            result = await self._suggest_plugin_rules(event)
            yield event.plain_result(result)
            return
        if subcommand == 'daily':
            if len(parts) == 2:
                result = await self._show_global_daily()
                yield event.plain_result(result)
            else:
                param = ' '.join(parts[2:]).strip('[]')
                result = await self._show_session_daily(param)
                yield event.plain_result(result)
            return
        yield event.plain_result(
            "用法：\n"
            "/tokenstats - 查看当前会话统计\n"
            "/tokenstats suggest - 获取插件规则建议\n"
            "/tokenstats daily - 查看全局每日统计\n"
            "/tokenstats daily <会话ID> - 查看指定会话每日统计"
        )

    # ---------- 辅助方法 ----------
    async def _suggest_plugin_rules(self, event: AstrMessageEvent) -> str:
        session_id = self._get_full_session_id(event)
        messages = self.session_messages.get(session_id)
        if not messages:
            return "当前会话没有消息记录。请先发送一条消息触发 LLM 请求。"
        non_system = [msg for msg in messages if msg.get('role') != 'system']
        if not non_system:
            return "没有非 system 消息，无法分析插件特征。"
        role_counter = Counter(msg.get('role') for msg in non_system)
        name_list = [msg.get('name') for msg in non_system if msg.get('name')]
        name_prefixes = Counter()
        for name in name_list:
            parts = name.split('_')
            if len(parts) > 1:
                name_prefixes[parts[0]] += 1
            else:
                name_prefixes[name] += 1
        content_starts = Counter()
        for msg in non_system:
            content = self.extract_text(msg.get('content')).strip()
            if content:
                start = content[:20]
                content_starts[start] += 1
        lines = [
            "🔍 **插件归属规则建议**",
            "根据当前会话的消息特征，您可以考虑以下规则：",
            "",
            "📌 **消息是谁发的（role）**:"
        ]
        for role, count in role_counter.most_common(3):
            lines.append(f"  - 角色: \"{role}\" (出现 {count} 次)")
        lines.append("")
        if name_prefixes:
            lines.append("📌 **消息的名称（name）里包含的关键词（适用于函数调用）**:")
            for prefix, count in name_prefixes.most_common(5):
                lines.append(f"  - 名称包含: \"{prefix}\" (出现 {count} 次)")
            lines.append("")
        if content_starts:
            lines.append("📌 **消息内容开头**:")
            for start, count in content_starts.most_common(5):
                lines.append(f"  - 内容开头: \"{start}\" (出现 {count} 次)")
            lines.append("")
        lines.append("💡 **怎么用**：")
        lines.append("1. 把上面的条件组合起来，填入 config.yaml 的 plugins 列表中。")
        lines.append("2. 例如：")
        lines.append("   plugins:")
        if role_counter:
            top_role = role_counter.most_common(1)[0][0]
            lines.append(f"     - name: \"你的插件名\"")
            lines.append(f"       match:")
            lines.append(f"         role: \"{top_role}\"")
        if name_prefixes:
            top_prefix = name_prefixes.most_common(1)[0][0]
            lines.append(f"         name_contains: \"{top_prefix}\"")
        elif content_starts:
            top_start = content_starts.most_common(1)[0][0]
            lines.append(f"         content_startswith: \"{top_start}\"")
        lines.append("3. 保存后重启 AstrBot，再发一条消息，然后输入 /tokenstats 查看统计。")
        return "\n".join(lines)

    async def _show_global_daily(self) -> str:
        lines = ["📅 **全局每日 Token 消耗统计**", "━━━━━━━━━━━━━━━━━━"]
        today = datetime.now().date()
        today_str = today.isoformat()
        for i in range(7):
            date = (today - timedelta(days=i)).isoformat()
            total = 0
            for session_stats in self.daily_stats.values():
                if date in session_stats:
                    total += session_stats[date].get("total_tokens", 0)
            if date == today_str:
                async with self._daily_lock:
                    for counters in self.current_session_today_counter.values():
                        total += counters.get("total_tokens", 0)
            lines.append(f"{date}: {total} tokens")
        lines.append("━━━━━━━━━━━━━━━━━━")
        lines.append("💡 使用 `/tokenstats daily <会话ID>` 查看特定会话的详细分类")
        return "\n".join(lines)

    def _get_display_name(self, key: str) -> str:
        if key == "persona_tokens":
            return "人格提示词"
        elif key == "context_tokens":
            return "上下文对话"
        elif key == "unclassified_tokens":
            return "未归类"
        elif key.startswith("plugin_") and key.endswith("_tokens"):
            plugin_name = key[7:-7]
            return f"插件：{plugin_name}"
        elif key == "input_tokens":
            return "输入"
        elif key == "output_tokens":
            return "输出"
        else:
            return key

    async def _show_session_daily(self, session_input: str) -> str:
        matched_ids = []
        # 改进匹配：支持完全相等、结尾匹配、包含匹配
        for sid in self.daily_stats.keys():
            if sid.endswith(f":{session_input}") or sid == session_input or session_input in sid:
                matched_ids.append(sid)
        async with self._daily_lock:
            for sid in self.current_session_today_counter.keys():
                if sid.endswith(f":{session_input}") or sid == session_input or session_input in sid:
                    if sid not in matched_ids:
                        matched_ids.append(sid)

        if not matched_ids:
            sample_sessions = list(self.daily_stats.keys())[:3]
            if not sample_sessions:
                sample_sessions = list(self.current_session_today_counter.keys())[:3]
            sample_str = ', '.join(sample_sessions) if sample_sessions else '无'
            return (f"未找到会话 {session_input} 的统计数据。"
                    f"已有会话示例: {sample_str}")

        if len(matched_ids) > 1:
            results = []
            for sid in matched_ids:
                results.append(await self._show_one_session_daily(sid))
            return "\n\n".join(results)
        else:
            return await self._show_one_session_daily(matched_ids[0])

    async def _show_one_session_daily(self, session_id: str) -> str:
        lines = [f"📅 **会话 {session_id} 每日 Token 消耗**", "━━━━━━━━━━━━━━━━━━"]
        today = datetime.now().date()
        for i in range(7):
            date = (today - timedelta(days=i)).isoformat()
            session_hist = self.daily_stats.get(session_id, {})
            day_data = session_hist.get(date, {})
            total = day_data.get("total_tokens", 0)
            async with self._daily_lock:
                today_counters = self.current_session_today_counter.get(session_id, {})
                if today_counters and date == today.isoformat():
                    total += today_counters.get("total_tokens", 0)
            lines.append(f"{date}: {total} tokens")
            if day_data:
                details = []
                for cat, val in day_data.items():
                    if cat != "total_tokens" and val > 0:
                        display_name = self._get_display_name(cat)
                        details.append(f"{display_name}: {val}")
                if details:
                    lines.append(f"   └─ {' | '.join(details)}")
        lines.append("━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)

    # ---------- 独立 Web 服务 ----------
    async def _start_web_server(self):
        try:
            self.web_app = web.Application()
            self.web_app.router.add_get('/', self._web_index)
            self.web_app.router.add_get('/api/overview', self._api_overview)
            self.web_app.router.add_get('/api/daily', self._api_daily)
            self.web_app.router.add_get('/api/sessions', self._api_sessions)
            self.web_app.router.add_get('/api/session/detail', self._api_session_detail)
            self.web_app.router.add_get('/api/session/messages', self._api_session_messages)
            self.web_app.router.add_get('/api/config/get', self._api_config_get)
            self.web_app.router.add_post('/api/config/save', self._api_config_save)

            self.web_runner = web.AppRunner(self.web_app)
            await self.web_runner.setup()
            port = 8765
            site = web.TCPSite(self.web_runner, 'localhost', port)
            await site.start()
            logger.info(f"{LOG_PREFIX}独立 Web 服务已启动: http://localhost:{port}")
        except Exception as e:
            logger.error(f"{LOG_PREFIX}启动独立 Web 服务失败: {e}")

    async def _web_index(self, request):
        web_dir = os.path.join(os.path.dirname(__file__), "web")
        index_path = os.path.join(web_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                content = f.read()
            return web.Response(text=content, content_type='text/html')
        else:
            return web.Response(text="WebUI 文件未找到", status=404)

    # --- 统计 API ---
    async def _api_overview(self, request):
        total_all = 0
        today_total = 0
        total_input = 0
        total_output = 0
        today_str = datetime.now().date().isoformat()
        for session_stats in self.daily_stats.values():
            for date, day_data in session_stats.items():
                total_all += day_data.get("total_tokens", 0)
                if date == today_str:
                    today_total += day_data.get("total_tokens", 0)
                total_input += day_data.get("input_tokens", 0)
                total_output += day_data.get("output_tokens", 0)
        async with self._daily_lock:
            for counters in self.current_session_today_counter.values():
                today_total += counters.get("total_tokens", 0)
                total_input += counters.get("input_tokens", 0)
                total_output += counters.get("output_tokens", 0)
        return web.json_response({
            "total_all": total_all,
            "today_total": today_total,
            "total_input": total_input,
            "total_output": total_output
        })

    async def _api_daily(self, request):
        days = int(request.query.get("days", 7))
        today = datetime.now().date()
        result = []
        for i in range(days):
            date = (today - timedelta(days=i)).isoformat()
            total = 0
            input_t = 0
            output_t = 0
            for session_stats in self.daily_stats.values():
                if date in session_stats:
                    day_data = session_stats[date]
                    total += day_data.get("total_tokens", 0)
                    input_t += day_data.get("input_tokens", 0)
                    output_t += day_data.get("output_tokens", 0)
            if date == today.isoformat():
                async with self._daily_lock:
                    for counters in self.current_session_today_counter.values():
                        total += counters.get("total_tokens", 0)
                        input_t += counters.get("input_tokens", 0)
                        output_t += counters.get("output_tokens", 0)
            result.append({"date": date, "total": total, "input": input_t, "output": output_t})
        return web.json_response(result)

    def _format_session_display(self, session_id: str) -> str:
        """根据会话ID生成友好的显示名称，支持多种格式"""
        if ':' not in session_id:
            return session_id
        parts = session_id.split(':')
        
        # 尝试解析各种格式
        if len(parts) == 3:
            platform, msg_type, user_id = parts
            if msg_type == 'FriendMessage':
                type_label = '私聊'
            elif msg_type == 'GroupMessage':
                type_label = '群聊'
            elif msg_type == 'private':
                type_label = '私聊'
            elif msg_type == 'group':
                type_label = '群聊'
            else:
                type_label = msg_type
            return f"{platform}({type_label}):{user_id}"
        elif len(parts) == 4:
            platform, robot_id, msg_type, user_id = parts
            if msg_type == 'FriendMessage':
                type_label = '私聊'
            elif msg_type == 'GroupMessage':
                type_label = '群聊'
            elif msg_type == 'private':
                type_label = '私聊'
            elif msg_type == 'group':
                type_label = '群聊'
            else:
                type_label = msg_type
            return f"{platform}({robot_id} {type_label}):{user_id}"
        else:
            return session_id

    async def _api_sessions(self, request):
        sessions = []
        processed_ids = set()
        for session_id, session_hist in self.daily_stats.items():
            processed_ids.add(session_id)
            # 计算该会话的总消耗和插件总消耗（累计所有日期）
            total = 0
            plugin_total = 0
            last_active = None
            for date, day_data in session_hist.items():
                total += day_data.get("total_tokens", 0)
                # 累加插件消耗
                for key, val in day_data.items():
                    if key.startswith("plugin_") and key.endswith("_tokens"):
                        plugin_total += val
                # 记录最新日期
                if last_active is None or date > last_active:
                    last_active = date
            
            # 加上当天未迁移的数据
            async with self._daily_lock:
                today_data = self.current_session_today_counter.get(session_id, {})
                if today_data:
                    total += today_data.get("total_tokens", 0)
                    for key, val in today_data.items():
                        if key.startswith("plugin_") and key.endswith("_tokens"):
                            plugin_total += val
                    # 当天日期可能比 last_active 新
                    today_str = datetime.now().date().isoformat()
                    if last_active is None or today_str > last_active:
                        last_active = today_str
            
            display_name = self._format_session_display(session_id)
            sessions.append({
                "session_id": session_id,
                "display_name": display_name,
                "total": total,
                "plugin_total": plugin_total,
                "last_active": last_active or "无数据"
            })
        
        # 添加仅有当天数据、尚未有历史记录的会话
        async with self._daily_lock:
            for session_id, counters in self.current_session_today_counter.items():
                if session_id in processed_ids:
                    continue
                total = counters.get("total_tokens", 0)
                plugin_total = 0
                for key, val in counters.items():
                    if key.startswith("plugin_") and key.endswith("_tokens"):
                        plugin_total += val
                display_name = self._format_session_display(session_id)
                sessions.append({
                    "session_id": session_id,
                    "display_name": display_name,
                    "total": total,
                    "plugin_total": plugin_total,
                    "last_active": datetime.now().date().isoformat()
                })
        
        sessions.sort(key=lambda x: x["total"], reverse=True)
        return web.json_response(sessions)

    async def _api_session_detail(self, request):
        session_id = request.query.get("session_id", "")
        if not session_id:
            return web.json_response({"error": "缺少 session_id 参数"}, status=400)
        days = int(request.query.get("days", 7))
        today = datetime.now().date()
        result = []
        for i in range(days):
            date = (today - timedelta(days=i)).isoformat()
            day_data = self.daily_stats.get(session_id, {}).get(date, {})
            total = day_data.get("total_tokens", 0)
            if date == today.isoformat():
                async with self._daily_lock:
                    today_data = self.current_session_today_counter.get(session_id, {})
                    if today_data:
                        total += today_data.get("total_tokens", 0)
            details = []
            for cat, val in day_data.items():
                if cat != "total_tokens" and val > 0:
                    display = self._get_display_name(cat)
                    details.append(f"{display}: {val}")
            details_str = " | ".join(details) if details else "无分类数据"
            result.append({
                "date": date,
                "total": total,
                "details": details_str
            })
        return web.json_response(result)

    async def _api_session_messages(self, request):
        session_id = request.query.get("session_id", "")
        logger.debug(f"{LOG_PREFIX} API 请求消息: session_id={session_id}")
        if not session_id:
            logger.warning(f"{LOG_PREFIX} API 请求缺少 session_id")
            return web.json_response({"error": "缺少 session_id 参数"}, status=400)

        try:
            messages = self.session_messages.get(session_id, [])
            for msg in messages:
                if 'content' in msg and not isinstance(msg['content'], str):
                    msg['content'] = str(msg['content'])
            logger.debug(f"{LOG_PREFIX} 返回 {len(messages)} 条消息")
            return web.json_response({"messages": messages})
        except Exception as e:
            logger.error(f"{LOG_PREFIX} 获取消息失败: {e}", exc_info=True)
            return web.json_response({"error": f"服务器内部错误: {str(e)}"}, status=500)

    async def _api_config_get(self, request):
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read()
            return web.json_response({"content": content})
        except Exception as e:
            logger.error(f"{LOG_PREFIX}读取配置文件失败: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _api_config_save(self, request):
        data = await request.json()
        new_content = data.get("content", "")
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        backup_path = config_path + ".bak"
        try:
            yaml.safe_load(new_content)
            if os.path.exists(config_path):
                shutil.copy2(config_path, backup_path)
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            self.config = self._load_config()
            self.plugin_rules = self.config.get("plugins", [])
            
            # 不再清除会话缓存，避免丢失已有统计数据
            # 但需要让用户知道新规则将应用于未来的请求
            logger.info(f"{LOG_PREFIX}配置已更新，新规则将在下次请求时生效。")
            
            return web.json_response({"success": True, "message": "配置已保存，新规则将在下次请求时生效。"})
        except yaml.YAMLError as e:
            logger.error(f"{LOG_PREFIX}保存配置文件失败: YAML 格式错误 - {e}")
            return web.json_response({"error": f"YAML 格式错误: {e}"}, status=400)
        except Exception as e:
            logger.error(f"{LOG_PREFIX}保存配置文件失败: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def terminate(self):
        async with self._daily_lock:
            await self._migrate_current_day_to_history()
        if self._daily_task:
            self._daily_task.cancel()
            try:
                await self._daily_task
            except asyncio.CancelledError:
                pass
        if self.web_runner:
            await self.web_runner.cleanup()
            logger.info(f"{LOG_PREFIX}Web 服务已关闭")
        logger.info(f"{LOG_PREFIX}插件已卸载")