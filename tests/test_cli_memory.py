import pytest
import os
import json
import tempfile
from pico.cli.memory import SQLiteMemoryCLI

@pytest.fixture
def temp_db():
    # 一時的なDBファイルパスを生成
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    # テスト後に削除
    if os.path.exists(path):
        os.remove(path)

@pytest.fixture
def temp_md():
    fd, path = tempfile.mkstemp(suffix=".md")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)

def test_write_and_search_trigram(temp_db, temp_md, capsys):
    cli = SQLiteMemoryCLI(db_path=temp_db)
    try:
        # Write
        cli.write_knowledge(
            filepath=temp_md,
            title="猫のタマちゃん",
            content="夕方はタマちゃんに警告音を鳴らさずに話しかける。",
            tags="cat pets"
        )
        
        # Capture print output from write
        captured = capsys.readouterr()
        write_res = json.loads(captured.out)
        assert write_res["status"] == "success"
        assert write_res["filepath"] == temp_md

        # Search 1: 3文字以上の Trigram 検索 ("タマちゃん" -> 5文字)
        cli.search_knowledge("タマちゃん", limit=1)
        captured = capsys.readouterr()
        search_res = json.loads(captured.out)
        
        assert len(search_res["results"]) == 1
        assert search_res["results"][0]["title"] == "猫のタマちゃん"
        assert "警告音" in search_res["results"][0]["content"]

        # Search 2: 2文字以下の LIKE フォールバック検索 ("猫" -> 1文字)
        cli.search_knowledge("猫", limit=1)
        captured = capsys.readouterr()
        search_res_fallback = json.loads(captured.out)
        
        assert len(search_res_fallback["results"]) == 1
        assert search_res_fallback["results"][0]["title"] == "猫のタマちゃん"

    finally:
        cli.close()
