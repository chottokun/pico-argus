import sqlite3
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class MemoryStore:
    """SQLite FTS5 Trigram を中核とする、追加VRAM不要の日本語長期記憶想起データベースクラス。"""

    def __init__(self, db_path: str = "wiki.db") -> None:
        self.db_path: str = db_path
        self.conn: sqlite3.Connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        """必要なテーブル定義と日本語全文検索インデックス (FTS5 Trigram) を定義する。"""
        cursor = self.conn.cursor()
        
        # メタデータおよび本文を格納する本体テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS wiki_metadata (
                filepath TEXT PRIMARY KEY,
                doc_type TEXT,
                title TEXT,
                tags TEXT,
                content TEXT,
                last_reviewed TEXT,
                provenance_source TEXT,
                provenance_confidence TEXT
            )
        """)
        
        # CJK（日本語）検索対応のために trigram トークナイザーを使用した FTS5 仮想テーブルを生成
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5(
                filepath,
                doc_type,
                title,
                tags,
                content,
                tokenize='trigram'
            )
        """)
        self.conn.commit()

    def add_entry(
        self, filepath: str, doc_type: str, title: str, tags: str, content: str,
        last_reviewed: str = "2026-07-20", provenance_source: str = "user",
        provenance_confidence: str = "high"
    ) -> None:
        """新しいドキュメント（記憶）を追加、または上書きし、FTS5 インデックスを更新する。"""
        cursor = self.conn.cursor()
        
        # 本体テーブルへの書き込み
        cursor.execute("""
            INSERT OR REPLACE INTO wiki_metadata 
            (filepath, doc_type, title, tags, content, last_reviewed, provenance_source, provenance_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (filepath, doc_type, title, tags, content, last_reviewed, provenance_source, provenance_confidence))
        
        # FTS5仮想テーブルへの書き込み (同期のために一回削除してから挿入)
        cursor.execute("DELETE FROM wiki_fts WHERE filepath = ?", (filepath,))
        cursor.execute("""
            INSERT INTO wiki_fts (filepath, doc_type, title, tags, content)
            VALUES (?, ?, ?, ?, ?)
        """, (filepath, doc_type, title, tags, content))
        
        self.conn.commit()
        logger.info(f"Memory indexed successfully: {filepath} ({title})")


    def search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """FTS5 Trigram と LIKE 句による 2段階フォールバック想起検索を実行する。"""
        # クエリを空白区切りの単語リストにする
        words = [w for w in re.split(r'\s+', query) if w]
        if not words:
            return []

        results: List[Dict[str, Any]] = []
        seen_filepaths = set()

        # --- Phase 1: 3文字以上のキーワードに対する高速 Trigram FTS5 検索 ---
        fts_words = [f'"{w}"' for w in words if len(w) >= 3]
        if fts_words:
            fts_query = " AND ".join(fts_words)
            self.conn.row_factory = sqlite3.Row
            cursor = self.conn.cursor()
            try:
                cursor.execute("""
                    SELECT f.filepath, f.doc_type, f.title, f.tags, f.content, 
                           m.last_reviewed, m.provenance_source, m.provenance_confidence,
                           bm25(wiki_fts) as rank
                    FROM wiki_fts f
                    JOIN wiki_metadata m ON f.filepath = m.filepath
                    WHERE wiki_fts MATCH ?
                    ORDER BY rank ASC
                    LIMIT ?
                """, (fts_query, limit))
                
                for row in cursor.fetchall():
                    filepath = row["filepath"]
                    seen_filepaths.add(filepath)
                    score = float(-row["rank"]) + 10.0
                    results.append(self._row_to_dict(row, score))
            except sqlite3.OperationalError as e:
                logger.warning(f"SQLite FTS5 MATCH operational error: {e}")

        # --- Phase 2: 2文字以下の極小単語、または結果数が不足する場合の LIKE 句による部分一致救済 ---
        if len(results) < limit:
            remaining = limit - len(results)
            self.conn.row_factory = sqlite3.Row
            cursor = self.conn.cursor()
            
            # LIKE句での絞り込みクエリの構築
            like_conditions = []
            like_params = []
            for word in words:
                like_term = f"%{word}%"
                like_conditions.append("(title LIKE ? OR content LIKE ? OR tags LIKE ?)")
                like_params.extend([like_term, like_term, like_term])
            
            where_clause = " AND ".join(like_conditions)
            
            sql = f"""
                SELECT filepath, doc_type, title, tags, content, last_reviewed, provenance_source, provenance_confidence
                FROM wiki_metadata
                WHERE {where_clause}
            """
            
            if seen_filepaths:
                not_in_placeholders = ",".join(["?"] * len(seen_filepaths))
                sql += f" AND filepath NOT IN ({not_in_placeholders})"
                like_params.extend(list(seen_filepaths))
            
            sql += " LIMIT ?"
            like_params.append(remaining)
            
            try:
                cursor.execute(sql, tuple(like_params))
                for row in cursor.fetchall():
                    filepath = row["filepath"]
                    if filepath not in seen_filepaths:
                        seen_filepaths.add(filepath)
                        results.append(self._row_to_dict(row, 1.0))
            except sqlite3.OperationalError as e:
                logger.error(f"LIKE query failed: {e}")

        # 総合スコアの降順で並び替えて制限数に絞り込む
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def close(self) -> None:
        """データベース接続を閉じる。"""
        self.conn.close()
        logger.info("MemoryStore connection closed.")


    def _row_to_dict(self, row: sqlite3.Row, score: float) -> Dict[str, Any]:
        """sqlite3.Row を 標準の辞書オブジェクトに整形する。"""
        return {
            "filepath": row["filepath"],
            "doc_type": row["doc_type"],
            "title": row["title"],
            "tags": row["tags"].split() if row["tags"] else [],
            "content": row["content"],
            "last_reviewed": row["last_reviewed"],
            "provenance": {
                "source": row["provenance_source"],
                "confidence": row["provenance_confidence"]
            },
            "score": score
        }
