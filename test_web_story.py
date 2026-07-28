import pytest

from dataveritas.agent_tools import generate_web_story


ARTICLE = """
# Inflação desacelera no período analisado

A inflação oficial desacelerou para 4,2% no período mais recente, segundo os dados analisados.
No período anterior, o indicador acumulado estava em 5,1%, uma diferença de 0,9 ponto percentual.
O resultado mostra uma redução no ritmo de aumento dos preços, mas não significa queda generalizada.
Os dados cobrem apenas o intervalo informado no pacote e podem ser revisados pela fonte.

## Como checamos
Os números foram comparados com a tabela original.

## Auditoria factual
Qual é a limitação? O recorte não permite explicar causas.

## Fontes originais
- https://example.gov.br/dados
"""


def test_generate_web_story_is_extractive_and_excludes_audit_sections():
    story = generate_web_story(ARTICLE, max_cards=6)
    assert story["title"] == "Inflação desacelera no período analisado"
    assert 3 <= story["card_count"] <= 6
    assert story["generation"] == "extractive"
    content = " ".join(card["text"] for card in story["cards"])
    assert "4,2%" in content
    assert "Os números foram comparados" not in content
    assert "https://example.gov.br" not in content
    texts = [card["text"] for card in story["cards"] if card["type"] != "closing"]
    assert len(texts) == len(set(texts))


def test_generate_web_story_validates_card_limit():
    with pytest.raises(ValueError, match="entre 3 e 10"):
        generate_web_story(ARTICLE, max_cards=2)
