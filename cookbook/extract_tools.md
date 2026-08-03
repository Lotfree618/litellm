# Extract Tools

LiteLLM Proxy 提供统一网页正文提取接口，并保留 Firecrawl `/v2/scrape` 兼容入口。第一版支持 Firecrawl provider；多个同名 deployment 会由 Extract Router 按最少并发和 failure domain 状态选择。

## 配置

```yaml
extract_tools:
  - extract_tool_name: web-extract
    litellm_params:
      extract_provider: firecrawl
      api_key: os.environ/FIRECRAWL_KEY_PRIMARY
      api_base: https://api.firecrawl.dev/v2
      failure_domain: primary
      weight: 1
      max_parallel_requests: 2
      timeout: 50
      num_retries: 0
    extract_tool_info:
      description: Firecrawl-backed web content extraction
```

同一个 `extract_tool_name` 可以配置多个 deployment。建议每个独立 Firecrawl Team 使用一个 `failure_domain`；多个 key 属于同一 Team 时不要把它们当作额外额度或速率限制。

## 接口

统一接口：

```http
POST /v1/extract/web-extract
Authorization: Bearer <virtual-key>
Content-Type: application/json
```

```json
{
  "url": "https://example.com/article",
  "formats": ["markdown"],
  "only_main_content": true,
  "include_tags": [],
  "exclude_tags": [],
  "max_age": 0
}
```

Firecrawl 兼容接口：

```http
POST /firecrawl/v2/scrape
Authorization: Bearer <virtual-key>
Content-Type: application/json
```

兼容入口接受 Firecrawl 的 `onlyMainContent`、`includeTags`、`excludeTags` 和 `maxAge` 字段，并固定调用 `web-extract`。未实现的 `actions`、`location`、`proxy` 等字段会返回 `422`，不会静默忽略。

## 权限

普通 Virtual Key 必须在 object permission 中显式包含 `web-extract`：

```json
{"extract_tools": ["web-extract"]}
```

空列表不授予任何 Extract 工具。Proxy admin 不受该列表限制。统一接口和 Firecrawl 兼容接口使用同一权限检查。

## 路由和故障处理

- `score = inflight / weight`，跳过达到 `max_parallel_requests` 的 deployment。
- `401` 禁用 credential；`402` 标记 failure domain 额度耗尽。
- `429` 按 `Retry-After` 冷却，缺失时使用 45 秒。
- 只对 `401`、`402`、`429` 尝试另一个 deployment，单次请求最多两次尝试。
- `400`、`404`、`422`、`5xx`、网络超时不会切换 credential。
- URL 在调用 provider 前通过 LiteLLM SSRF 校验；禁止凭据 URL、内网/回环/链路本地/组播和云 metadata 地址，并校验重定向目标。
