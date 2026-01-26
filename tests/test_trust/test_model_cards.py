from foodspec.trust.model_cards import ModelCard, write_model_card


def test_model_card_to_dict():
    card = ModelCard(
        name="demo",
        version="0.1",
        overview="Demo model overview.",
        intended_use="test",
    )
    payload = card.to_dict()
    assert payload["name"] == "demo"
    assert payload["overview"] == "Demo model overview."


def test_write_model_card_markdown(tmp_path):
    card = ModelCard(
        name="demo",
        version="0.1",
        overview="Demo model overview.",
        intended_use="test",
    )
    path = write_model_card(tmp_path, card, format="md")
    assert path.exists()
    content = path.read_text()
    assert "Model Card" in content
