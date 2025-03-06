FINANCIAL_KEYWORDS = [
    "финанс", "банк", "кредит", "заем", "инвест", "акци", "облигац",
    "закят", "риба", "мурабаха", "мушарака", "иджара", "сукук",
    "такафул", "вакф", "халяль", "харам", "шариат", "ислам"
]

def is_islamic_finance_related(text: str) -> bool:
    text = text.lower()
    return any(keyword in text for keyword in FINANCIAL_KEYWORDS)