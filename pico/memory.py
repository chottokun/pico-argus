import sqlite3
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class MemoryStore:
    """SQLite FTS5 Trigram と 相互リンク(WikiLinks)/バックリンク構造を中核とする、追加VRAM不要の日本語長期記憶データベースクラス。"""

    def __init__(self, db_path: str = "wiki.db") -> None:
        self.db_path: str = db_path
        self.conn: sqlite3.Connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        """必要なテーブル定義、日本語全文検索インデックス (FTS5 Trigram)、および 相互リンク・エイリアステーブルを定義する。"""
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

        # 相互リンク (WikiLinks) リレーションテーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS wiki_links (
                source_filepath TEXT,
                target_title TEXT,
                relation_type TEXT DEFAULT 'wikilink',
                PRIMARY KEY (source_filepath, target_title)
            )
        """)

        # エイリアス（表記ゆれ・名寄せ）テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS wiki_aliases (
                alias_name TEXT PRIMARY KEY,
                target_filepath TEXT
            )
        """)

        self.conn.commit()

    def add_entry(
        self, filepath: str, doc_type: str, title: str, tags: str, content: str,
        last_reviewed: str = "2026-07-20", provenance_source: str = "user",
        provenance_confidence: str = "high", aliases: Optional[List[str]] = None
    ) -> None:
        """新しいドキュメント（記憶）を追加、または上書きし、FTS5 インデックスおよび 相互リンク/エイリアスを更新する。"""
        cursor = self.conn.cursor()
        
        # 1. 本体テーブルへの書き込み
        cursor.execute("""
            INSERT OR REPLACE INTO wiki_metadata 
            (filepath, doc_type, title, tags, content, last_reviewed, provenance_source, provenance_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (filepath, doc_type, title, tags, content, last_reviewed, provenance_source, provenance_confidence))
        
        # 2. FTS5仮想テーブルへの書き込み
        cursor.execute("DELETE FROM wiki_fts WHERE filepath = ?", (filepath,))
        cursor.execute("""
            INSERT INTO wiki_fts (filepath, doc_type, title, tags, content)
            VALUES (?, ?, ?, ?, ?)
        """, (filepath, doc_type, title, tags, content))

        # 3. 相互リンク (WikiLinks: [[...]]) の自動抽出とインデックス更新
        cursor.execute("DELETE FROM wiki_links WHERE source_filepath = ?", (filepath,))
        extracted_targets = set(re.findall(r'\[\[(.*?)\]\]', content))
        for target in extracted_targets:
            target_clean = target.strip()
            if target_clean:
                cursor.execute("""
                    INSERT OR REPLACE INTO wiki_links (source_filepath, target_title, relation_type)
                    VALUES (?, ?, 'wikilink')
                """, (filepath, target_clean))

        # 4. エイリアス（名寄せ）テーブルの更新
        # タイトル自体もエイリアスとして自動登録
        cursor.execute("INSERT OR REPLACE INTO wiki_aliases (alias_name, target_filepath) VALUES (?, ?)", (title, filepath))
        if aliases:
            for alias in aliases:
                alias_clean = alias.strip()
                if alias_clean:
                    cursor.execute("INSERT OR REPLACE INTO wiki_aliases (alias_name, target_filepath) VALUES (?, ?)", (alias_clean, filepath))
        
        self.conn.commit()
        logger.info(f"Memory indexed successfully with links & aliases: {filepath} ({title})")

    def resolve_canonical(self, title_or_alias: str) -> Optional[str]:
        """タイトルまたはエイリアス名から、対応する正規の filepath を返す。"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT target_filepath FROM wiki_aliases WHERE alias_name = ?", (title_or_alias,))
        row = cursor.fetchone()
        if row:
            return row[0]
        
        # 直接 title にヒットするか確認
        cursor.execute("SELECT filepath FROM wiki_metadata WHERE title = ?", (title_or_alias,))
        row = cursor.fetchone()
        return row[0] if row else None

    def get_backlinks(self, title_or_path: str) -> List[Dict[str, Any]]:
        """指定されたタイトルまたはファイルパスを参照している被参照 (バックリンク) 一覧を取得する。"""
        cursor = self.conn.cursor()

        # パスが渡された場合はタイトルを特定
        title = title_or_path
        cursor.execute("SELECT title FROM wiki_metadata WHERE filepath = ?", (title_or_path,))
        row = cursor.fetchone()
        if row:
            title = row[0]

        cursor.execute("""
            SELECT l.source_filepath, m.title, m.tags, m.content
            FROM wiki_links l
            LEFT JOIN wiki_metadata m ON l.source_filepath = m.filepath
            WHERE l.target_title = ? OR l.target_title IN (
                SELECT title FROM wiki_metadata WHERE filepath = ?
            )
        """, (title, title_or_path))

        backlinks = []
        for r in cursor.fetchall():
            backlinks.append({
                "source_filepath": r[0],
                "title": r[1] or "",
                "tags": r[2].split() if r[2] else [],
                "snippet": (r[3] or "")[:150]
            })
        return backlinks

    def get_forward_links(self, filepath: str) -> List[str]:
        """指定ドキュメントから伸びているリンク先 (target_title) リストを返却する。"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT target_title FROM wiki_links WHERE source_filepath = ?", (filepath,))
        return [row[0] for row in cursor.fetchall()]

    def search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """FTS5 Trigram と LIKE 句による 2段階想起検索を実行し、バックリンク/フォワードリンク構造を拡張付与する。"""
        words = [w for w in re.split(r'\s+', query) if w]
        if not words:
            return []

        results: List[Dict[str, Any]] = []
        seen_filepaths = set()

        # Phase 1: Trigram FTS5 検索
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

        # Phase 2: LIKE 句によるフォールバック
        if len(results) < limit:
            remaining = limit - len(results)
            self.conn.row_factory = sqlite3.Row
            cursor = self.conn.cursor()
            
            like_conditions = []
            like_params = []
            for word in words:
                like_term = f"%{word}%"
                like_conditions.append("(title LIKE ? OR content LIKE ? OR tags LIKE ?)")
                like_params.extend([like_term, like_term, like_term])
            
            where_clause = " AND ".join(like_conditions)
            sql = f"SELECT filepath, doc_type, title, tags, content, last_reviewed, provenance_source, provenance_confidence FROM wiki_metadata WHERE {where_clause}"
            
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

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def close(self) -> None:
        """データベース接続を閉じる。"""
        self.conn.close()
        logger.info("MemoryStore connection closed.")

    def _row_to_dict(self, row: sqlite3.Row, score: float) -> Dict[str, Any]:
        """sqlite3.Row を辞書オブジェクトに変換し、バックリンク・フォワードリンク構造を自動付与する。"""
        filepath = row["filepath"]
        title = row["title"]

        backlinks = self.get_backlinks(filepath)
        forward_links = self.get_forward_links(filepath)

        return {
            "filepath": filepath,
            "doc_type": row["doc_type"],
            "title": title,
            "tags": row["tags"].split() if row["tags"] else [],
            "content": row["content"],
            "last_reviewed": row["last_reviewed"],
            "provenance": {
                "source": row["provenance_source"],
                "confidence": row["provenance_confidence"]
            },
            "score": score,
            "backlinks": backlinks,
            "forward_links": forward_links
        }
