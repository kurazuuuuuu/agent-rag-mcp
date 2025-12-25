# Future: JWT Authentication System / 将来計画: JWT認証システム

> [!NOTE]
> この文書は将来の実装要件を記録したものです。
> 現在はシンプルなトークン認証を使用しています。

## Overview / 概要

Webサイト + FastAPIバックエンドによるJWT認証システムを構築し、
ユーザー登録後にMCPサーバーへの接続トークンを発行する。

## Requirements / 要件

### 1. Web Frontend
- ユーザー登録ページ
- ログインページ
- ダッシュボード（トークン管理）
- トークン生成・失効機能

### 2. FastAPI Backend
- `/auth/register` - ユーザー登録
- `/auth/login` - ログイン、JWT発行
- `/auth/token/mcp` - MCP接続用トークン発行
- `/auth/token/revoke` - トークン失効

### 3. JWT Structure
```json
{
  "sub": "user_id",
  "exp": "expiration_time",
  "scope": ["mcp:read", "mcp:write", "mcp:learn"],
  "team": "team_id"
}
```

### 4. MCP Server Integration
- FastMCP Bearer Auth Provider使用
- JWTの検証（署名、有効期限、スコープ）
- チームごとのデータ分離

### 5. Database
- Users table
- Teams table
- Tokens table (revocation list)

## Architecture / アーキテクチャ

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Web UI    │────▶│  FastAPI    │────▶│  Database   │
│  (React?)   │     │  Backend    │     │ (PostgreSQL)│
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
                           │ JWT
                           ▼
                    ┌─────────────┐
                    │  MCP Server │
                    │ (FastMCP)   │
                    └─────────────┘
```

## Priority / 優先度

- P0: ユーザー登録・ログイン
- P1: MCP接続トークン発行
- P2: チーム機能
- P3: トークンスコープ管理
