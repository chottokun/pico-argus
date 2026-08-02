import argparse
import os
import json
from typing import Optional, List
from pico.memory import MemoryStore

class SQLiteMemoryCLI:
    """MemoryStore を利用する長期記憶想起・書込CLI"""
    def __init__(self, db_path: str = "wiki.db"):
        self.store = MemoryStore(db_path=db_path)

    def search_knowledge_data(self, query: str, limit: int = 3) -> list:
        """日本語想起結果をリストで返却 (printなし)"""
        results = self.store.search(query, limit)
        if not results:
            return []
        
        output = []
        for r in results:
            output.append({
                "filepath": r["filepath"],
                "title": r["title"],
                "tags": r["tags"],
                "content": r["content"],
                "score": r["score"],
                "provenance": r["provenance"],
                "backlinks": r.get("backlinks", []),
                "forward_links": r.get("forward_links", [])
            })
        return output

    def search_knowledge(self, query: str, limit: int = 3):
        """日本語想起 (FTS5 Trigram -> LIKE フォールバック)"""
        output = self.search_knowledge_data(query, limit)
        print(json.dumps({"results": output}, indent=2, ensure_ascii=False))

    def write_knowledge_data(self, filepath: str, title: str, content: str, tags: str = "", aliases: Optional[List[str]] = None) -> dict:
        """新しい記憶を OKF 形式 Markdown に書き込みインデックスを更新して結果を辞書で返却"""
        # OKF Frontmatter を付与したコンテンツの構築
        from datetime import datetime
        import tempfile
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        else:
            dir_name = "."

        lock_file_path = filepath + ".lock"
        okf_content = ""

        # ロックファイルによる排他制御
        with open(lock_file_path, "w", encoding="utf-8") as lock_file:
            try:
                import fcntl
                fcntl.flock(lock_file, fcntl.LOCK_EX)
            except (ImportError, AttributeError):
                pass

            try:
                existing_content = ""
                if os.path.exists(filepath):
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            existing_content = f.read()
                    except Exception:
                        pass

                alias_str = ", ".join(aliases) if aliases else ""

                if existing_content:
                    parts = existing_content.split("---\n")
                    if len(parts) >= 3:
                        frontmatter = parts[1]
                        body = "---\n".join(parts[2:])
                        updated_body = body.strip() + f"\n\n### 🕒 観測記録: {timestamp_str}\n{content}"
                        okf_content = f"---\n{frontmatter}---\n\n{updated_body}"
                    else:
                        okf_content = existing_content.strip() + f"\n\n---\n### 🕒 観測記録: {timestamp_str}\n{content}"
                else:
                    aliases_fm = f"aliases: [{alias_str}]\n" if alias_str else ""
                    okf_content = (
                        f"---\n"
                        f"title: {title}\n"
                        f"tags: {tags}\n"
                        f"{aliases_fm}"
                        f"doc_type: knowledge\n"
                        f"provenance_source: CLI memory write\n"
                        f"provenance_confidence: High\n"
                        f"---\n\n"
                        f"### 🕒 観測記録: {timestamp_str}\n{content}"
                    )

                # 一時ファイルへ書き込み、ディスク完全同期してアトミックに置換
                with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
                    temp_name = tf.name
                    tf.write(okf_content)
                    tf.flush()
                    try:
                        os.fsync(tf.fileno())
                    except OSError:
                        pass

                os.replace(temp_name, filepath)

            finally:
                try:
                    import fcntl
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                except (ImportError, AttributeError):
                    pass

        self.store.add_entry(
            filepath=filepath,
            doc_type="knowledge",
            title=title,
            tags=tags,
            content=okf_content,
            aliases=aliases
        )
        return {"status": "success", "filepath": filepath}

    def write_knowledge(self, filepath: str, title: str, content: str, tags: str = ""):
        """新しい記憶・環境ルールを OKF 形式 Markdown に書き込みインデックスを更新"""
        res = self.write_knowledge_data(filepath, title, content, tags)
        print(json.dumps(res, ensure_ascii=False))

    def close(self):
        self.store.close()

def main():
    parser = argparse.ArgumentParser(description="Memory CLI tool")
    parser.add_argument("--action", choices=["search", "write"], required=True, help="Action to perform")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--limit", type=int, default=3, help="Max results for search")
    parser.add_argument("--file", type=str, help="File path to write")
    parser.add_argument("--title", type=str, help="Title of knowledge to write")
    parser.add_argument("--content", type=str, help="Content of knowledge to write")
    parser.add_argument("--tags", type=str, default="", help="Space separated tags")
    parser.add_argument("--db", type=str, default="wiki.db", help="SQLite DB file path")
    args = parser.parse_args()

    cli = SQLiteMemoryCLI(db_path=args.db)
    try:
        if args.action == "search":
            if args.query is None:
                parser.error("--query is required for search action")
            cli.search_knowledge(args.query, args.limit)
        elif args.action == "write":
            if not all([args.file, args.title, args.content]):
                parser.error("--file, --title, and --content are required for write action")
            cli.write_knowledge(args.file, args.title, args.content, args.tags)
    finally:
        cli.close()

if __name__ == "__main__":
    main()
