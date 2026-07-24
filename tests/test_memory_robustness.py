import pytest
import os
import tempfile
from pico.memory import MemoryStore
from pico.cli.memory import SQLiteMemoryCLI

@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass

def test_fts5_special_characters_handling(temp_db):
    """FTS5 構文エラーを引き起こしやすい特殊文字が含まれるクエリの堅牢性テスト"""
    store = MemoryStore(db_path=temp_db)
    store.add_entry(
        filepath="wiki/special_chars.md",
        doc_type="knowledge",
        title="特殊文字テスト",
        tags="test special",
        content="C++ や C# や 'quotes' \"double\" (parentheses) :colon *star AND OR NOT などの記号テキスト。"
    )

    # 1. ダブルクォート、シングルクォート、コロン、カッコ等の記号を含む検索クエリ
    quirky_queries = [
        'C++',
        'C#',
        '"double quotes"',
        "single'quote",
        '(parentheses)',
        'colon:test',
        '*star*',
        'AND OR NOT',
        '   　  '  # 全角・半角スペースのみ
    ]

    for q in quirky_queries:
        # 例外がスローされずに安全に結果が得られるか検証
        try:
            results = store.search(q)
            assert isinstance(results, list)
        except Exception as e:
            pytest.fail(f"Query '{q}' raised an unexpected exception: {e}")

    store.close()

def test_wikilinks_duplicates_and_empty_edges(temp_db):
    """重複 WikiLink や 空の [[ ]] などの境界値テスト"""
    store = MemoryStore(db_path=temp_db)
    
    # 同一リンクが本文中に複数回登場するドキュメント
    store.add_entry(
        filepath="wiki/duplicate_links.md",
        doc_type="knowledge",
        title="重複リンクテスト",
        tags="test",
        content="[[タマちゃん]] が来た。また [[タマちゃん]] が登場。[[  ]] 空リンクや [[ ]] も含む。"
    )

    forward_links = store.get_forward_links("wiki/duplicate_links.md")
    # 重複なしで 'タマちゃん' が1つだけ登録されていること
    assert len(forward_links) == 1
    assert forward_links[0] == "タマちゃん"

    store.close()

def test_malformed_markdown_resilience(temp_db):
    """Frontmatter が破損した既存 Markdown に対する追記処理の堅牢性テスト"""
    cli = SQLiteMemoryCLI(db_path=temp_db)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        broken_file = os.path.join(temp_dir, "broken.md")
        
        # 1. 破損した Frontmatter (閉じ --- がない)
        with open(broken_file, "w", encoding="utf-8") as f:
            f.write("---\ntitle: Broken File\ntags: bad\nNo closing dashes here...\n")

        # 追記実行
        res = cli.write_knowledge_data(
            filepath=broken_file,
            title="Fix Title",
            content="新たな追記コンテンツ",
            tags="fix"
        )
        assert res["status"] == "success"

        with open(broken_file, "r", encoding="utf-8") as f:
            updated_text = f.read()

        assert "新たな追記コンテンツ" in updated_text
        assert "観測記録" in updated_text

    cli.close()
