# Agent RAG MCP

> [!WARNING]
> まだ開発途中のプロジェクトです。

## 概要

Agent RAG MCPサーバーは **AIエージェントのハルシネーション・繰り返しのエラー・ドキュメントとの整合性** を改善することを目的としたツールです。

## プロジェクト構造

```
agent-rag-mcp/
├── src/agent_rag_mcp/     # メインパッケージ
│   ├── __init__.py        # パッケージエントリーポイント
│   ├── config.py          # 設定管理（環境変数の一元化）
│   ├── server.py          # FastMCPサーバー本体
│   ├── gemini_rag.py      # Gemini File Search RAGクライアント
│   └── client.py          # プロキシクライアント
├── docs/                   # ドキュメント
│   ├── AGENT_PROMPT.md    # エージェント用システムプロンプト
│   └── future_jwt_auth.md # JWT認証の将来計画
├── schema/                 # スキーマ定義
│   └── request_schema.*   # リクエストスキーマ
├── .env.example           # 環境変数テンプレート
├── fastmcp.json           # FastMCP設定
├── mcp.json               # MCP接続設定例
└── pyproject.toml         # Python依存関係
```

## 機能

### ドキュメントRAG（実装済み ✅）

* Gemini APIのFile Search Toolを使用したRAGを提供
* サーバー起動時に自動的にドキュメントをインデックス
  * Git リポジトリからクローン（`RAG_REPO_URL`）
  * ローカルディレクトリ（`RAG_LOCAL_DOCS_PATH`）
* **既存ストア検出**: 2回目以降は自動でスキップ（コスト削減）
* `ask_project_document` ツールでプロジェクトについて質問可能

### 認証（実装済み ✅）

* Bearer Token認証（`AUTH_TOKEN`環境変数で有効化）
* 設定しない場合は認証なしで動作（開発モード）

### 動的学習RAG（計画中 🚧）

* エージェントの成功/失敗パターンを学習
* `ask_code_pattern`, `tell_code_pattern` ツール

## 設定

`.env.example` を参照してください。主な設定：

| 環境変数 | 説明 | 必須 |
|---------|------|------|
| `GEMINI_API_KEY` | Gemini APIキー | ✅ |
| `RAG_REPO_URL` | ドキュメントのGitリポジトリURL | △ |
| `RAG_LOCAL_DOCS_PATH` | ローカルドキュメントパス | △ |
| `AUTH_TOKEN` | 認証トークン | - |
| `RAG_FORCE_REINDEX` | 強制再インデックス | - |

## 技術構成

### MCP Server
* **Python** + [uv](https://docs.astral.sh/uv/) + [FastMCP v2](https://github.com/jlowin/fastmcp)
* **Gemini API** - [File Search Tool](https://blog.google/technology/developers/file-search-gemini-api/)

### Deployment
* Docker / Docker Compose