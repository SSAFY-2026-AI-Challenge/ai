"""학급 모의 경제 더미 데이터 생성기.

fin_edu 스키마의 member / account / transaction / payroll 을 채운다.

설계 요지
---------
학생마다 두 개의 잠재 성향을 부여한다.

* ``theta``   : 성실성. 출석률·숙제완료율·상환 성실도를 함께 끌어올린다.
* ``impulse`` : 충동성. 간식/뽑기 같은 재량 소비를 늘려 잔액을 말린다.

이 성향은 DB 에 저장하지 않는다. 모델은 성향을 볼 수 없고, 성향이 만들어 낸
'행동의 흔적'(거래 내역, 잔액, 출석·숙제 지표)만 보고 연체를 예측해야 한다.

급여는 한 달에 한 번 준다. 한 달을 4주로 두고 아래 리듬을 반복한다.

    1주차  월급 지급, 적금 자동이체, 예금 이자
    2주차  세금 납부일 (소득세 / 건강 보험료 / 자리 임대료 / 학급 기금)
    3주차  대출 상환일
    4주차  의무 지출 없음

재량 소비는 매주 일어난다. 월급을 받자마자 다 써 버리면 2주차 세금과 3주차
상환에서 돈이 모자라고, 그때 연체료가 붙는다. 이 연체료 거래가 곧 신용평가의
정답(bad) 라벨이 된다. '목돈이 들어왔을 때 나눠 쓰는 법'이 이 모의 경제의
핵심 학습 지점이고, 모델이 잡아내야 하는 신호이기도 하다.

한 달을 달력 대신 4주로 고정한 이유가 있다. 달력 월로 급여를 주면 관측
윈도우(8주)에 급여일이 2번 들어갈 때와 3번 들어갈 때가 생겨서, 학생의 행동과
무관하게 소득 피처가 50%씩 출렁인다. 4주 주기로 두면 어떤 8주 구간에도 급여일이
정확히 2번 들어가므로 그런 잡음이 사라진다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from . import domain as D
from .config import SimulationConfig

# --- 급여 주기 ---------------------------------------------------------------
# 납부일을 월 후반에 두는 것이 핵심이다. 월급 바로 다음 주에 걷으면 통장이 아직
# 두둑해서 아무도 밀리지 않고, 그러면 대출을 쓴 학생만 위험해 보이는 모델이 된다.
WEEKS_PER_MONTH = 4               # 한 달 = 4주
PAYDAY_WEEK = 0                   # 월 주기 안에서 월급을 주는 주
TAX_WEEK = 2                      # 세금 납부 주
LOAN_WEEK = 3                     # 대출 상환 주

# --- 시뮬레이션 튜닝 상수 ----------------------------------------------------
OPENING_GIFT_RATIO = 0.25         # 계좌 개설 축하금 = 월급 * 비율
SAVINGS_JOIN_BASE = 0.30          # 적금 가입 확률 기본값
SAVINGS_JOIN_THETA = 0.45         # 성실할수록 적금 가입
LOAN_JOIN_BASE = 0.62             # 대출 이용 확률 기본값
LOAN_JOIN_THETA = -0.22
LOAN_JOIN_IMPULSE = 0.18
# 납부 성실도. 돈이 있어도 깜빡하는 정도이며, 성실한 학생일수록 덜 깜빡한다.
# 이 부분은 잔액만 봐서는 알 수 없고 출석·숙제 같은 생활 습관에만 드러난다.
REPAY_BASE = 0.90                 # 대출 상환일을 챙길 확률 기본값
REPAY_THETA = 0.14
REPAY_IMPULSE = -0.08
TAX_PAY_BASE = 0.93               # 세금 납부일을 챙길 확률 기본값
TAX_PAY_THETA = 0.12
# 달마다 흔들리는 컨디션. 관측되지 않으므로 모델이 맞힐 수 없는 부분이다.
MONTHLY_DILIGENCE_SHOCK = 0.13
LATE_FEE_RATE = 0.10              # 연체료 = 미납액 * 비율
ABSENCE_DEDUCTION = 0.8           # 결석한 주만큼 월급에서 깎이는 비율
TRANSFER_IN_RATE = 0.12           # 학기 중 전학 와서 늦게 계좌를 만드는 학생 비율

# 주간 재량 소비 = 남은 잔액 * 소비성향. 월급은 한 번에 들어오는데 소비는 매주
# 일어나므로, 소비성향이 높으면 세금 납부일 전에 이미 통장이 말라 버린다.
SPEND_RATIO_BASE = 0.20
SPEND_RATIO_IMPULSE = 0.50
SPEND_RATIO_THETA = -0.11
SPEND_RATIO_NOISE = 0.17

# 월급날 과소비. 목돈이 들어온 주에 충동성이 높은 학생은 씀씀이가 커지고,
# 그 결과 다음 주 세금 납부일에 돈이 모자란다.
PAYDAY_SPLURGE = 0.45

# 매주 이 확률로 예상 못 한 지출이 생긴다(준비물, 친구 선물 등).
UNEXPECTED_EXPENSE_PROB = 0.12

_SURNAMES = (
    "김", "이", "박", "최", "정", "강", "조", "윤", "장", "임",
    "한", "오", "서", "신", "권", "황", "안", "송", "류", "전",
)
_GIVEN_NAMES = (
    "서준", "하윤", "도윤", "지우", "시우", "서연", "예준", "하은", "주원", "지민",
    "지호", "수아", "준우", "다은", "건우", "채원", "우진", "지아", "선우", "유진",
    "민준", "소율", "은우", "예린", "태윤", "가온", "이준", "나윤", "현우", "서아",
)


@dataclass
class SimulatedEconomy:
    """DB 적재 전의 시뮬레이션 결과."""

    members: pd.DataFrame       # login_id 기준
    accounts: pd.DataFrame      # account_id, login_id, account_type, created_at, balance
    transactions: pd.DataFrame  # account_id, amount, description, occured_at, login_id
    payroll_keys: pd.DataFrame  # login_id, account_id, occured_at (급여 거래 식별용)

    def summary(self) -> dict[str, float | int]:
        txn = self.transactions
        late = txn[txn["description"].isin(D.LATE_EVENT_DESCRIPTIONS)]
        students = self.members[self.members["role"] == "STUDENT"]
        return {
            "members": len(self.members),
            "students": len(students),
            "accounts": len(self.accounts),
            "transactions": len(txn),
            "late_events": len(late),
            "students_with_late_event": late["login_id"].nunique(),
            "period_start": str(txn["occured_at"].min()),
            "period_end": str(txn["occured_at"].max()),
        }


def _mondays(start: date, end: date) -> list[date]:
    cur = start
    out: list[date] = []
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=7)
    return out


def _round_to(value: float, unit: int = 10) -> int:
    return int(max(unit, round(value / unit) * unit))


def _ceil_to(value: float, unit: int = 10) -> int:
    return int(math.ceil(value / unit) * unit)


def _make_name(rng: np.random.Generator) -> str:
    return rng.choice(_SURNAMES) + rng.choice(_GIVEN_NAMES)


def _assign_taxes(rng: np.random.Generator, salary: int) -> dict[str, object]:
    n_tax = int(rng.integers(2, 5))
    chosen = rng.choice(len(D.TAX_ITEMS), size=n_tax, replace=False)
    fields: dict[str, object] = {}
    for slot in range(1, 6):
        fields[f"tax{slot}_name"] = None
        fields[f"tax{slot}_amount"] = None
    for slot, idx in enumerate(sorted(chosen), start=1):
        name, rate = D.TAX_ITEMS[idx]
        fields[f"tax{slot}_name"] = name
        fields[f"tax{slot}_amount"] = _round_to(salary * rate, 10)
    return fields


def _member_taxes(member: dict) -> list[tuple[str, int]]:
    out = []
    for slot in range(1, 6):
        name = member.get(f"tax{slot}_name")
        amount = member.get(f"tax{slot}_amount")
        if name and amount:
            out.append((name, int(amount)))
    return out


def _simulate_student_cashflow(
    rng: np.random.Generator,
    member: dict,
    weeks: list[date],
) -> tuple[list[dict], list[dict], list[dict]]:
    """한 학생의 계좌/거래를 생성한다.

    반환: (accounts, transactions, payroll_keys)
    """
    login_id = member["login_id"]
    theta = member["_theta"]
    impulse = member["_impulse"]
    salary = int(member["salary"])
    taxes = _member_taxes(member)
    tax_total = sum(amount for _, amount in taxes)

    # 일부 학생은 학기 중에 전학 온다. 계좌 개설 시점이 달라진다.
    join_week = 0 if rng.random() > TRANSFER_IN_RATE else int(rng.integers(1, 13))
    checking_id = f"CHK-{login_id}"
    open_at = datetime.combine(
        weeks[join_week] - timedelta(days=3), datetime.min.time()
    ).replace(hour=9)
    accounts = [
        {
            "account_id": checking_id,
            "login_id": login_id,
            "account_type": D.ACCOUNT_CHECKING,
            "created_at": open_at,
        }
    ]
    txns: list[dict] = []
    payroll_keys: list[dict] = []

    def add(account_id: str, amount: int, description: str, when: datetime) -> None:
        txns.append(
            {
                "account_id": account_id,
                "login_id": login_id,
                "amount": int(amount),
                "description": description,
                "occured_at": when,
            }
        )

    balance = _round_to(salary * OPENING_GIFT_RATIO, 10)
    add(checking_id, balance, D.INCOME_OPENING, open_at)

    def cover_shortfall(needed: int, when: datetime) -> bool:
        """입출금 잔액이 모자라면 적금을 깨서 메운다.

        적금이 '위기 때 나를 지켜 주는 돈'이 되도록 하는 장치다.
        이 규칙이 없으면 적금은 유동성만 줄이는 나쁜 신호가 되어 버린다.
        """
        nonlocal balance, savings_balance
        if balance >= needed:
            return True
        if not savings_opened or savings_balance <= 0:
            return False
        short = needed - balance
        draw = min(savings_balance, _ceil_to(short, 10))
        if draw <= 0:
            return False
        add(savings_id, -draw, D.INCOME_SAVINGS_WITHDRAW, when)
        add(checking_id, draw, D.INCOME_SAVINGS_WITHDRAW, when)
        savings_balance -= draw
        balance += draw
        return balance >= needed

    # 적금 가입 여부
    has_savings = rng.random() < min(0.95, SAVINGS_JOIN_BASE + SAVINGS_JOIN_THETA * theta)
    savings_id = f"SAV-{login_id}"
    savings_balance = 0
    savings_open_week = join_week + int(rng.integers(0, 4)) if has_savings else -1
    savings_rate = float(rng.uniform(0.10, 0.30))
    savings_opened = False

    # 대출 이용 여부. 대출 실행과 상환은 모두 '상환 주'에 일어난다.
    loan_prob = LOAN_JOIN_BASE + LOAN_JOIN_THETA * theta + LOAN_JOIN_IMPULSE * impulse
    has_loan = rng.random() < float(np.clip(loan_prob, 0.05, 0.95))
    loan_id = f"LON-{login_id}"
    loan_day_weeks = [
        i for i in range(join_week + 1, len(weeks) - WEEKS_PER_MONTH)
        if i % WEEKS_PER_MONTH == LOAN_WEEK
    ]
    loan_start_week = int(rng.choice(loan_day_weeks)) if has_loan and loan_day_weeks else -1
    loan_outstanding = 0
    installment = 0
    loan_opened = False

    repay_will = float(
        np.clip(REPAY_BASE + REPAY_THETA * theta + REPAY_IMPULSE * impulse, 0.55, 0.99)
    )
    tax_will = float(np.clip(TAX_PAY_BASE + TAX_PAY_THETA * theta, 0.6, 0.995))

    attendance = float(member["cs_attendance"])
    # 그 달의 사정. 거래 내역 어디에도 남지 않는 요인이라 모델은 알 수 없다.
    # 현실의 신용평가가 완벽할 수 없는 이유를 데이터에 그대로 넣어 둔 것이다.
    month_shock = 0.0

    for w, monday in enumerate(weeks):
        if w < join_week:
            continue
        day = datetime.combine(monday, datetime.min.time())
        phase = w % WEEKS_PER_MONTH

        # 1) 월급날 -- 급여, 적금 자동이체, 예금 이자 ---------------------------
        if phase == PAYDAY_WEEK:
            month_shock = float(rng.normal(0, MONTHLY_DILIGENCE_SHOCK))
            # 결석한 주만큼 급여에서 깎인다. 출석이 곧 소득으로 이어지는 장치다.
            absent_weeks = sum(
                1 for _ in range(WEEKS_PER_MONTH) if rng.random() > attendance
            )
            deduction = absent_weeks / WEEKS_PER_MONTH * ABSENCE_DEDUCTION
            pay = _round_to(salary * (1 - deduction) * (1 + rng.normal(0, 0.02)), 10)
            if absent_weeks < WEEKS_PER_MONTH:
                add(checking_id, pay, D.INCOME_SALARY, day.replace(hour=9))
                payroll_keys.append(
                    {"login_id": login_id, "account_id": checking_id, "occured_at": day.replace(hour=9)}
                )
                balance += pay

            if rng.random() < 0.25 + 0.35 * theta:
                bonus = _round_to(salary * rng.uniform(0.03, 0.12), 10)
                add(checking_id, bonus, D.INCOME_BONUS, day.replace(hour=13))
                balance += bonus

            if has_savings and w >= savings_open_week:
                if not savings_opened:
                    accounts.append(
                        {
                            "account_id": savings_id,
                            "login_id": login_id,
                            "account_type": D.ACCOUNT_SAVINGS,
                            "created_at": day.replace(hour=9, minute=30),
                        }
                    )
                    savings_opened = True
                deposit = _round_to(salary * savings_rate, 10)
                keeps_plan = rng.random() < 0.55 + 0.40 * theta
                # 이번 달 세금과 대출 상환분은 남겨 두고 남는 돈만 적금에 넣는다.
                reserve = tax_total + (installment if loan_outstanding > 0 else 0)
                if keeps_plan and balance - deposit >= reserve:
                    when = day.replace(hour=9, minute=40)
                    add(checking_id, -deposit, D.EXPENSE_SAVINGS_DEPOSIT, when)
                    add(savings_id, deposit, D.EXPENSE_SAVINGS_DEPOSIT, when)
                    balance -= deposit
                    savings_balance += deposit
                if savings_balance > 0:
                    interest = _round_to(savings_balance * 0.01, 10)
                    add(savings_id, interest, D.INCOME_INTEREST, day.replace(hour=16))
                    savings_balance += interest

        # 2) 세금 납부일 -------------------------------------------------------
        # 돈이 있어도 납부일을 깜빡할 수 있다. 이 '예측 불가능한 부분'이 있어야
        # 모델이 유동성만 보고 정답을 맞히는 시시한 문제가 되지 않는다.
        if phase == TAX_WEEK:
            if rng.random() >= tax_will + month_shock:
                when = day.replace(hour=9, minute=20)
                fee = _round_to(tax_total * LATE_FEE_RATE, 10)
                add(checking_id, -fee, D.LATE_FEE_TAX, when)
                balance -= fee
            else:
                for t_idx, (tax_name, tax_amount) in enumerate(taxes):
                    when = day.replace(hour=9, minute=20 + t_idx)
                    if cover_shortfall(tax_amount, when):
                        add(checking_id, -tax_amount, f"{D.EXPENSE_TAX_PREFIX}{tax_name}", when)
                        balance -= tax_amount
                    else:
                        fee = _round_to(tax_amount * LATE_FEE_RATE, 10)
                        add(checking_id, -fee, D.LATE_FEE_TAX, when)
                        balance -= fee

        # 3) 대출 상환일 -------------------------------------------------------
        if phase == LOAN_WEEK:
            if has_loan and w == loan_start_week:
                principal = _round_to(salary * rng.uniform(0.8, 2.2), 50)
                term_months = int(rng.integers(2, 7))
                installment = _round_to(principal * 1.05 / term_months, 10)
                loan_outstanding = principal
                accounts.append(
                    {
                        "account_id": loan_id,
                        "login_id": login_id,
                        "account_type": D.ACCOUNT_LOAN,
                        "created_at": day.replace(hour=10),
                    }
                )
                loan_opened = True
                add(loan_id, -principal, D.LOAN_DISBURSE, day.replace(hour=10))
                add(checking_id, principal, D.INCOME_LOAN_DISBURSE, day.replace(hour=10, minute=1))
                balance += principal
            elif loan_opened and loan_outstanding > 0:
                due = min(installment, loan_outstanding)
                when = day.replace(hour=10, minute=30)
                willing = rng.random() < repay_will + month_shock
                if willing and cover_shortfall(due, when):
                    add(checking_id, -due, D.EXPENSE_LOAN_REPAY, when)
                    add(loan_id, due, D.EXPENSE_LOAN_REPAY, when)
                    balance -= due
                    loan_outstanding -= due
                else:
                    fee = _round_to(due * LATE_FEE_RATE, 10)
                    add(checking_id, -fee, D.LATE_FEE_LOAN, when)
                    balance -= fee

        # 4) 알뜰장터 판매 -----------------------------------------------------
        if rng.random() < 0.08:
            sale = _round_to(salary * rng.uniform(0.03, 0.10), 10)
            add(checking_id, sale, D.INCOME_MARKET, day.replace(hour=15) + timedelta(days=3))
            balance += sale

        # 5) 예상 못 한 지출 ----------------------------------------------------
        # 달마다 통장 사정이 달라지게 만드는 장치. 이게 없으면 씀씀이만으로 연체가
        # 결정돼 '한 번 연체한 학생은 반드시 또 연체한다'는 뻔한 데이터가 된다.
        if rng.random() < UNEXPECTED_EXPENSE_PROB:
            shock = _round_to(salary * rng.uniform(0.05, 0.22), 10)
            shock = min(shock, max(10, balance))
            add(checking_id, -shock, D.EXPENSE_UNEXPECTED, day.replace(hour=12) + timedelta(days=2))
            balance -= shock

        # 6) 학급 규칙 위반 벌금 ------------------------------------------------
        if rng.random() < 0.05 + 0.22 * (1 - theta):
            fine = _round_to(salary * rng.uniform(0.012, 0.04), 10)
            fine = min(fine, max(10, balance))
            add(checking_id, -fine, D.EXPENSE_FINE, day.replace(hour=14) + timedelta(days=1))
            balance -= fine

        # 7) 재량 소비 -- 월급은 한 번에 들어오지만 소비는 매주 일어난다 ---------
        spend_ratio = float(
            np.clip(
                SPEND_RATIO_BASE
                + SPEND_RATIO_IMPULSE * impulse
                + SPEND_RATIO_THETA * theta
                + rng.normal(0, SPEND_RATIO_NOISE),
                0.03,
                0.92,
            )
        )
        if phase == PAYDAY_WEEK:
            spend_ratio = min(0.95, spend_ratio * (1 + PAYDAY_SPLURGE * impulse))
        budget = max(0, balance) * spend_ratio
        n_spend = int(rng.poisson(2.0 + 3.2 * impulse)) + 1
        # 충동성이 높은 학생은 간식·뽑기·보드게임 쪽으로 소비가 쏠린다.
        weights = np.array(
            [
                w_ * (1 + 2.2 * impulse if desc in D.DISCRETIONARY_DESCRIPTIONS else 1 - 0.5 * impulse)
                for desc, w_, _ in D.DISCRETIONARY_MENU
            ],
            dtype=float,
        )
        weights = weights / weights.sum()
        shares = rng.dirichlet(np.ones(n_spend) * 1.6)
        for share in shares:
            if balance <= 10:
                break
            pick = int(rng.choice(len(D.DISCRETIONARY_MENU), p=weights))
            desc, _, max_share = D.DISCRETIONARY_MENU[pick]
            amount = _round_to(budget * share, 10)
            cap = max(10, min(balance, int(salary * max_share)))
            amount = int(np.clip(amount, 10, cap))
            when = day.replace(hour=int(rng.integers(9, 17)), minute=int(rng.integers(0, 59)))
            when += timedelta(days=int(rng.integers(0, 5)))
            add(checking_id, -amount, desc, when)
            balance -= amount

        # 8) 적금 만기 ---------------------------------------------------------
        if savings_opened and savings_balance > 0 and w == savings_open_week + 24:
            when = day.replace(hour=11)
            add(savings_id, -savings_balance, D.INCOME_SAVINGS_MATURITY, when)
            add(checking_id, savings_balance, D.INCOME_SAVINGS_MATURITY, when)
            balance += savings_balance
            savings_balance = 0

    return accounts, txns, payroll_keys


def _simulate_teacher(rng: np.random.Generator, member: dict, weeks: list[date]) -> tuple[list[dict], list[dict]]:
    login_id = member["login_id"]
    account_id = f"CHK-{login_id}"
    open_at = datetime.combine(weeks[0] - timedelta(days=3), datetime.min.time()).replace(hour=9)
    accounts = [
        {
            "account_id": account_id,
            "login_id": login_id,
            "account_type": D.ACCOUNT_CHECKING,
            "created_at": open_at,
        }
    ]
    txns = []
    for w, monday in enumerate(weeks):
        if w % WEEKS_PER_MONTH != PAYDAY_WEEK:
            continue
        day = datetime.combine(monday, datetime.min.time()).replace(hour=8, minute=30)
        txns.append(
            {
                "account_id": account_id,
                "login_id": login_id,
                "amount": 20000,
                "description": "학급 운영비 지원",
                "occured_at": day,
            }
        )
        spend = _round_to(rng.uniform(15000, 21000), 10)
        txns.append(
            {
                "account_id": account_id,
                "login_id": login_id,
                "amount": -spend,
                "description": "학급 물품 구매",
                "occured_at": day + timedelta(days=2),
            }
        )
    return accounts, txns


def simulate(cfg: SimulationConfig | None = None) -> SimulatedEconomy:
    cfg = cfg or SimulationConfig()
    rng = np.random.default_rng(cfg.random_seed)
    weeks = _mondays(cfg.period_start, cfg.period_end)

    members: list[dict] = []
    accounts: list[dict] = []
    txns: list[dict] = []
    payroll_keys: list[dict] = []

    for grade in cfg.grades:
        for cls_no in range(1, cfg.classes_per_grade + 1):
            class_name = f"{grade}-{cls_no}"

            teacher = {
                "login_id": f"t{grade}{cls_no}",
                "pw": "teacher!23",
                "name": f"{rng.choice(_SURNAMES)}선생님",
                "role": "TEACHER",
                "avatar_url": f"/avatars/t{int(rng.integers(1, 9))}.png",
                "status": "ACTIVE",
                "class": class_name,
                "job": "담임 교사",
                "salary": "0",
                "cs_attendance": 1.0,
                "cs_overdue": 0,
                "cs_homework": 1.0,
                "_theta": 1.0,
                "_impulse": 0.0,
            }
            for slot in range(1, 6):
                teacher[f"tax{slot}_name"] = None
                teacher[f"tax{slot}_amount"] = None
            members.append(teacher)
            t_acc, t_txn = _simulate_teacher(rng, teacher, weeks)
            accounts.extend(t_acc)
            txns.extend(t_txn)

            for idx in range(1, cfg.students_per_class + 1):
                theta = float(np.clip(rng.beta(2.4, 1.7), 0.02, 0.99))
                # 충동성은 성실성과 약한 음의 상관을 갖는다.
                impulse = float(
                    np.clip(rng.beta(2.0, 2.6) * 0.75 + (1 - theta) * 0.35, 0.0, 1.0)
                )
                job_name, base_salary = D.JOBS[int(rng.integers(0, len(D.JOBS)))]
                salary = _round_to(base_salary * rng.uniform(0.9, 1.15), 10)

                member = {
                    "login_id": f"s{grade}{cls_no}{idx:02d}",
                    "pw": "fin!edu24",
                    "name": _make_name(rng),
                    "role": "STUDENT",
                    "avatar_url": f"/avatars/{int(rng.integers(1, 21)):02d}.png",
                    "status": "ACTIVE",
                    "class": class_name,
                    "job": job_name,
                    "salary": str(salary),
                    "cs_attendance": round(
                        float(np.clip(0.80 + 0.19 * theta + rng.normal(0, 0.035), 0.55, 1.0)), 4
                    ),
                    "cs_overdue": int(np.clip(rng.poisson(4.2 * (1 - theta) + 0.2), 0, 15)),
                    "cs_homework": round(
                        float(np.clip(0.66 + 0.33 * theta + rng.normal(0, 0.06), 0.3, 1.0)), 4
                    ),
                    "_theta": theta,
                    "_impulse": impulse,
                }
                member.update(_assign_taxes(rng, salary))
                members.append(member)

                s_acc, s_txn, s_pay = _simulate_student_cashflow(rng, member, weeks)
                accounts.extend(s_acc)
                txns.extend(s_txn)
                payroll_keys.extend(s_pay)

    members_df = pd.DataFrame(members)
    accounts_df = pd.DataFrame(accounts)
    txns_df = pd.DataFrame(txns).sort_values("occured_at").reset_index(drop=True)
    payroll_df = pd.DataFrame(payroll_keys)

    balances = txns_df.groupby("account_id")["amount"].sum()
    accounts_df["balance"] = accounts_df["account_id"].map(balances).fillna(0).astype(int)

    return SimulatedEconomy(
        members=members_df,
        accounts=accounts_df,
        transactions=txns_df,
        payroll_keys=payroll_df,
    )


# --- DB 적재 ----------------------------------------------------------------

_MEMBER_COLUMNS = (
    "login_id", "pw", "name", "role", "avatar_url", "status", "class", "job", "salary",
    "tax1_name", "tax1_amount", "tax2_name", "tax2_amount", "tax3_name", "tax3_amount",
    "tax4_name", "tax4_amount", "tax5_name", "tax5_amount",
    "cs_attendance", "cs_overdue", "cs_homework",
)


def truncate_all(conn) -> None:
    """기존 데이터를 모두 비운다(FK 순서 주의)."""
    from .db import execute

    execute(conn, "SET FOREIGN_KEY_CHECKS = 0")
    for table in ("credit_score", "payroll", "ai_recommendation", "ai_report",
                  "transaction", "account", "member"):
        try:
            execute(conn, f"TRUNCATE TABLE `{table}`")
        except Exception:  # noqa: BLE001 - 아직 없는 테이블은 무시
            pass
    execute(conn, "SET FOREIGN_KEY_CHECKS = 1")
    conn.commit()


def load_into_db(conn, economy: SimulatedEconomy, *, reset: bool = True) -> dict[str, int]:
    """시뮬레이션 결과를 fin_edu 스키마에 적재한다."""
    from .db import executemany, fetch_df

    if reset:
        truncate_all(conn)

    placeholders = ", ".join(["%s"] * len(_MEMBER_COLUMNS))
    columns = ", ".join(f"`{c}`" for c in _MEMBER_COLUMNS)
    member_rows = [
        tuple(None if pd.isna(row[c]) else row[c] for c in _MEMBER_COLUMNS)
        for _, row in economy.members.iterrows()
    ]
    n_members = executemany(
        conn, f"INSERT INTO `member` ({columns}) VALUES ({placeholders})", member_rows
    )
    conn.commit()

    id_map = fetch_df(conn, "SELECT id, login_id FROM member")
    login_to_id = dict(zip(id_map["login_id"], id_map["id"]))

    account_rows = [
        (
            row["account_id"],
            int(login_to_id[row["login_id"]]),
            row["account_type"],
            int(row["balance"]),
            row["created_at"].to_pydatetime() if hasattr(row["created_at"], "to_pydatetime") else row["created_at"],
        )
        for _, row in economy.accounts.iterrows()
    ]
    n_accounts = executemany(
        conn,
        "INSERT INTO `account` (`id`, `member_id`, `account_type`, `balance`, `created_at`) "
        "VALUES (%s, %s, %s, %s, %s)",
        account_rows,
    )
    conn.commit()

    txn_rows = (
        (
            row.account_id,
            int(row.amount),
            row.description,
            row.occured_at.to_pydatetime() if hasattr(row.occured_at, "to_pydatetime") else row.occured_at,
        )
        for row in economy.transactions.itertuples(index=False)
    )
    n_txns = executemany(
        conn,
        "INSERT INTO `transaction` (`account_id`, `amount`, `description`, `occured_at`) "
        "VALUES (%s, %s, %s, %s)",
        txn_rows,
    )
    conn.commit()

    # 급여 거래를 payroll 에 연결
    salary_txns = fetch_df(
        conn,
        "SELECT t.id AS transaction_id, a.member_id "
        "FROM `transaction` t JOIN `account` a ON a.id = t.account_id "
        "WHERE t.description = %s",
        (D.INCOME_SALARY,),
    )
    payroll_rows = [
        (int(r.member_id), int(r.transaction_id)) for r in salary_txns.itertuples(index=False)
    ]
    n_payroll = executemany(
        conn,
        "INSERT INTO `payroll` (`member_id`, `transaction_id`) VALUES (%s, %s)",
        payroll_rows,
    )
    conn.commit()

    return {
        "member": n_members,
        "account": n_accounts,
        "transaction": n_txns,
        "payroll": n_payroll,
    }


def bad_rate_preview(economy: SimulatedEconomy, obs_weeks: int = 8, perf_weeks: int = 4) -> float:
    """성과 윈도우 기준 부실률을 빠르게 확인한다(튜닝용)."""
    txn = economy.transactions
    students = set(economy.members.loc[economy.members["role"] == "STUDENT", "login_id"])
    start = txn["occured_at"].min().normalize() + pd.Timedelta(weeks=obs_weeks)
    end = txn["occured_at"].max().normalize() - pd.Timedelta(weeks=perf_weeks)
    dates = pd.date_range(start, end, freq="14D")
    rates = []
    late = txn[txn["description"].isin(D.LATE_EVENT_DESCRIPTIONS)]
    for d in dates:
        window = late[(late["occured_at"] > d) & (late["occured_at"] <= d + pd.Timedelta(weeks=perf_weeks))]
        bad = window["login_id"].nunique()
        rates.append(bad / max(1, len(students)))
    return float(np.mean(rates)) if rates else math.nan
