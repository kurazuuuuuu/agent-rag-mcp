# Agent RAG MCP
> [!WARNING]
> まだ開発途中のプロジェクトです。

## 概要
Agent RAG MCPサーバーは **AIエージェントのハルシネーション・繰り返しのエラー・ドキュメントとの整合性** を改善することを目的としたツールです。

## 詳細
### > 接続方法
検討中

### > ドキュメントRAG
* Gemini APIのFile Search Toolを使用したRAGを使用します。
* サーバー初回起動時に自動的に設定したリポジトリからRAGを作成してくれます。（設定リポジトリ内の`docs/`を参照します。コンテナに権限がある鍵がないと`Private`リポジトリはクローンできません。）
* エージェントは`ask_project_document`ツールを使用してプロジェクトの気になる部分を質問できます。
  * MCPサーバー内のGeminiが回答を生成しエージェントにレスポンスします。

### > 動的学習RAG
* ローカルLLM(Gemma 3 4b)(埋め込み, 条件分岐) + Gemini APIを使用します。
* ドキュメントを使用するのではなく、エージェントの **うまくいった・失敗した** コードを学習させ、何度も同じ問題を繰り返すことを防ぎます。
* エージェントは`ask_code_pattern`, `tell_code_pattern`の2つのツールを使用します。
  * **`ask_code_pattern`**
    * エージェントは**実装したい内容・機能**をリクエストに含めます。MCPサーバーはそれに応じて良い実装方法をレスポンスします。

  * `tell_code_pattern`
    * 2つのモードでリクエストを送信できます。(`error`, `success`)
    * `error`モード(エラーが発生したのでダメな例を学習させたいとき)
      * エージェントは**実装したい内容・機能・エラー内容・エラーが発生したコード**をリクエストに含めます。MCPサーバーはそれを使用してRAGを生成し、次回以降失敗しないように学習します。
      * もし過去にあった例であればMCPサーバーが良いコードパターンをエージェントにレスポンスします。
    * `success`モード(うまく動いたコードパターンを学習させたいとき)
      * エージェントは**実装したい内容・機能・エラー内容・エラーが発生したコード**をリクエストに含めます。MCPサーバーはそれを使用してRAGを生成し、次回以降失敗しないように学習します。

## 技術構成
### > MCP
* Python
  * [uv](https://docs.astral.sh/uv/)
  * [FastMCP v2](https://github.com/jlowin/fastmcp)
* Google
  * Gemma 3
  * Gemini API
    * [File Search Tool](https://blog.google/technology/developers/file-search-gemini-api/)
* Database
  * [Weaviate](https://github.com/weaviate/weaviate)

### > Deployment, Development
* Docker
  * Docker Compose