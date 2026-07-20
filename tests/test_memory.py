from pico.memory import MemoryStore

def test_memory_store_lifecycle_and_search() -> None:
    # メモリ内 DB を使用
    store = MemoryStore(db_path=":memory:")
    
    # 記憶の追加
    store.add_entry(
        filepath="docs/tapo_rules.md",
        doc_type="rule",
        title="Tapoカメラの安全規則",
        tags="tapo camera security",
        content="カメラは可動限界を超えて回転させてはならない。モーター過熱や内部断線のリスクがある。"
    )
    store.add_entry(
        filepath="docs/vlm_rules.md",
        doc_type="rule",
        title="VLMとの連携手順",
        tags="vlm ollama gemma",
        content="Ollamaで動作するgemma4:e2bは最も軽量なVLMであり、テスト時に最優先で使用すること。"
    )
    
    # 1. 3文字以上のキーワードによる Trigram 検索
    results_trigram = store.search("カメラ", limit=2)
    assert len(results_trigram) == 1
    assert results_trigram[0]["title"] == "Tapoカメラの安全規則"
    assert "可動限界" in results_trigram[0]["content"]

    # 2. 2文字以下の極小ワードによる LIKE句 フォールバック検索
    # FTS5 Trigram では2文字以下のマッチングが失敗しやすいが、LIKE句により想起できるか検証
    results_like = store.search("VLM", limit=2)
    assert len(results_like) == 1
    assert results_like[0]["title"] == "VLMとの連携手順"
    assert "gemma4" in results_like[0]["content"]
