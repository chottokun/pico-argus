import pytest
import os
import tempfile
from pico.memory import MemoryStore

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

def test_wiki_links_extraction_and_backlinks(temp_db):
    store = MemoryStore(db_path=temp_db)

    # 1. ノード A: 飼い猫タマちゃん
    store.add_entry(
        filepath="wiki/known_objects_tama.md",
        doc_type="knowledge",
        title="飼い猫タマちゃん",
        tags="pet cat",
        content="白茶トラの猫。赤色の首輪をしている。"
    )

    # 2. ノード B: 庭 Zone B (ノード A を [[飼い猫タマちゃん]] として参照)
    store.add_entry(
        filepath="wiki/zone_b_garden.md",
        doc_type="knowledge",
        title="庭 (Zone B)",
        tags="zone garden",
        content="夕方に [[飼い猫タマちゃん]] が散歩に訪れるエリア。"
    )

    # 3. バックリンク (被参照) の検証
    # "wiki/known_objects_tama.md" または "飼い猫タマちゃん" を参照しているノードを取得
    backlinks = store.get_backlinks("飼い猫タマちゃん")
    assert len(backlinks) == 1
    assert backlinks[0]["source_filepath"] == "wiki/zone_b_garden.md"

    # 4. フォワードリンク (参照元からのリンク一覧) の検証
    forward_links = store.get_forward_links("wiki/zone_b_garden.md")
    assert "飼い猫タマちゃん" in forward_links

    store.close()

def test_wiki_aliases(temp_db):
    store = MemoryStore(db_path=temp_db)

    # エイリアス付きで登録 ("タマ", "猫のタマ")
    store.add_entry(
        filepath="wiki/known_objects_tama.md",
        doc_type="knowledge",
        title="飼い猫タマちゃん",
        tags="pet cat",
        content="白茶トラの猫。",
        aliases=["タマ", "猫のタマ"]
    )

    # エイリアスから正規パスへの名寄せ検証
    canonical = store.resolve_canonical("猫のタマ")
    assert canonical == "wiki/known_objects_tama.md"

    store.close()

def test_graph_augmented_search(temp_db):
    store = MemoryStore(db_path=temp_db)

    store.add_entry(
        filepath="wiki/known_objects_tama.md",
        doc_type="knowledge",
        title="飼い猫タマちゃん",
        tags="pet cat",
        content="白茶トラの猫。"
    )

    store.add_entry(
        filepath="wiki/zone_b_garden.md",
        doc_type="knowledge",
        title="庭 (Zone B)",
        tags="zone garden",
        content="[[飼い猫タマちゃん]] が散歩する場所。"
    )

    # 想起検索時に backlinks / related_links がレスポンスに含まれるか
    results = store.search("飼い猫タマちゃん")
    assert len(results) >= 1
    target_item = next(r for r in results if r["title"] == "飼い猫タマちゃん")
    assert "backlinks" in target_item
    assert any(b["source_filepath"] == "wiki/zone_b_garden.md" for b in target_item["backlinks"])

    store.close()
