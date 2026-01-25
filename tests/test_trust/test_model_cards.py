from foodspec.trust.model_cards import ModelCard


def test_model_card_to_dict():
    card = ModelCard(name="demo", version="0.1", intended_use="test")
    payload = card.to_dict()
    assert payload["name"] == "demo"

