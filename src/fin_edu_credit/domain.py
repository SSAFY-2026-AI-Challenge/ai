"""모의 경제의 도메인 어휘.

`transaction.description` 에 들어가는 문자열을 한 곳에 모아 두고,
더미 데이터 생성기와 피처 엔지니어링이 같은 어휘를 공유하도록 한다.
"""

from __future__ import annotations

# --- 계좌 종류 --------------------------------------------------------------
ACCOUNT_CHECKING = "CHECKING"   # 입출금 통장
ACCOUNT_SAVINGS = "SAVINGS"     # 적금 통장
ACCOUNT_LOAN = "LOAN"           # 대출 계좌(잔액이 음수 = 남은 빚)

# --- 수입 -------------------------------------------------------------------
INCOME_SALARY = "주급 급여"
INCOME_BONUS = "칭찬 보너스"
INCOME_MARKET = "알뜰장터 판매"
INCOME_INTEREST = "예금 이자"
INCOME_OPENING = "계좌 개설 축하금"
INCOME_LOAN_DISBURSE = "대출금 입금"
INCOME_SAVINGS_MATURITY = "적금 만기 수령"
INCOME_SAVINGS_WITHDRAW = "적금 중도 인출"

# 계좌 사이를 옮겨 다닌 돈. 벌어들인 수입이 아니므로 소득 집계에서 제외한다.
TRANSFER_IN_DESCRIPTIONS = frozenset(
    {INCOME_LOAN_DISBURSE, INCOME_SAVINGS_MATURITY, INCOME_SAVINGS_WITHDRAW}
)

# --- 지출 -------------------------------------------------------------------
EXPENSE_SNACK = "간식 구매"
EXPENSE_STATIONERY = "학용품 구매"
EXPENSE_BOARDGAME = "보드게임 이용료"
EXPENSE_BOOK = "도서 대여료"
EXPENSE_GACHA = "뽑기 이용료"
EXPENSE_DONATION = "기부"
EXPENSE_SEAT_RENT = "자리 임대료"
EXPENSE_FINE = "벌금 - 학급 규칙 위반"

# --- 의무 지출 --------------------------------------------------------------
EXPENSE_TAX_PREFIX = "세금 납부 - "
EXPENSE_SAVINGS_DEPOSIT = "적금 납입"
EXPENSE_LOAN_REPAY = "대출 상환"

# --- 연체(부실) 이벤트 -------------------------------------------------------
LATE_FEE_LOAN = "대출 상환 연체료"
LATE_FEE_TAX = "세금 체납 가산금"
LATE_EVENT_DESCRIPTIONS = frozenset({LATE_FEE_LOAN, LATE_FEE_TAX})

# 대출 계좌 쪽 기표
LOAN_DISBURSE = "대출 실행"

# 충동적/재량 소비로 분류하는 항목
DISCRETIONARY_DESCRIPTIONS = frozenset(
    {EXPENSE_SNACK, EXPENSE_BOARDGAME, EXPENSE_GACHA}
)

# 재량 지출 후보와 상대 가중치
DISCRETIONARY_MENU: tuple[tuple[str, float, tuple[int, int]], ...] = (
    (EXPENSE_SNACK, 0.32, (20, 90)),
    (EXPENSE_STATIONERY, 0.20, (30, 120)),
    (EXPENSE_BOARDGAME, 0.16, (20, 70)),
    (EXPENSE_BOOK, 0.14, (10, 40)),
    (EXPENSE_GACHA, 0.12, (20, 100)),
    (EXPENSE_DONATION, 0.06, (10, 50)),
)

# --- 학급 직업과 주급 -------------------------------------------------------
JOBS: tuple[tuple[str, int], ...] = (
    ("학급 은행원", 900),
    ("우체부", 620),
    ("도서 관리자", 640),
    ("환경 지킴이", 560),
    ("급식 도우미", 580),
    ("학급 기자", 700),
    ("서기", 720),
    ("시간 관리자", 540),
    ("정리 반장", 600),
    ("알림이", 520),
    ("칠판 관리자", 500),
    ("재활용 반장", 560),
    ("컴퓨터 관리자", 760),
    ("우유 당번", 480),
    ("학급 경찰", 660),
)

# --- 세금 항목 --------------------------------------------------------------
TAX_ITEMS: tuple[tuple[str, float], ...] = (
    ("소득세", 0.10),
    ("건강 보험료", 0.05),
    ("자리 임대료", 0.08),
    ("학급 기금", 0.04),
)


def is_late_event(description: str | None) -> bool:
    return description in LATE_EVENT_DESCRIPTIONS


def is_tax_payment(description: str | None) -> bool:
    return bool(description) and description.startswith(EXPENSE_TAX_PREFIX)
