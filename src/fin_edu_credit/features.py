"""신용평가 피처 엔지니어링.

입력 테이블
-----------
* ``member``      : cs_attendance(출석률), cs_homework(숙제완료율), cs_overdue(도서 연체) 등 비금융 정보
* ``account``     : 계좌 종류와 잔액(balance)
* ``transaction`` : 계좌별 입출금 원장

관측 윈도우(기준일 직전 N주)의 행동으로 피처를 만들고,
성과 윈도우(기준일 직후 M주)에 연체가 발생했는지를 정답으로 삼는다.
기준일 이후 정보는 절대 피처에 섞이지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from . import domain as D
from .config import ModelConfig


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    label: str          # 한국어 설명 (사유코드/리포트에 사용)
    higher_is_better: bool
    advice: str         # 점수를 깎는 요인일 때 학생에게 줄 조언


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    # --- 비금융(생활 습관) ---
    FeatureSpec("cs_attendance", "출석률", True, "지각·결석을 줄이면 신용이 올라가요."),
    FeatureSpec("cs_homework", "숙제 완료율", True, "밀린 숙제를 끝내면 신용이 올라가요."),
    FeatureSpec("cs_overdue", "도서 연체 횟수", False, "빌린 책을 기한 안에 반납해 보세요."),
    # --- 상환/연체 이력 ---
    FeatureSpec("late_events_8w", "최근 연체 횟수", False, "약속한 납부일을 지키는 것이 가장 중요해요."),
    FeatureSpec("late_fee_amount_8w", "최근 연체료 합계", False, "연체료로 새는 돈부터 막아 보세요."),
    FeatureSpec("days_since_last_late", "마지막 연체 이후 경과일", True, "연체 없이 지낸 기간이 길수록 좋아요."),
    FeatureSpec("repay_ontime_ratio", "대출 상환 이행률", True, "대출 상환일을 달력에 표시해 두세요."),
    FeatureSpec("tax_paid_ratio", "세금 납부 이행률", True, "급여를 받으면 세금부터 먼저 내 보세요."),
    FeatureSpec("fine_count_8w", "학급 규칙 위반 벌금 횟수", False, "학급 규칙을 지키면 벌금이 줄어요."),
    # --- 소득/현금흐름 ---
    FeatureSpec("income_8w", "최근 8주 수입", True, "직업 활동으로 수입을 늘려 보세요."),
    FeatureSpec("expense_8w", "최근 8주 지출", False, "지출을 기록하고 새는 돈을 찾아보세요."),
    FeatureSpec("net_cashflow_8w", "최근 8주 순현금흐름", True, "쓰는 돈보다 버는 돈이 많아야 해요."),
    FeatureSpec("consumption_income_ratio", "수입 대비 소비 비율", False, "수입의 80% 안에서 쓰는 습관을 들여 보세요."),
    FeatureSpec("income_stability", "급여 수령 규칙성", True, "맡은 직업 활동을 꾸준히 해 보세요."),
    # --- 저축/자산 ---
    FeatureSpec("savings_rate", "저축률", True, "수입의 일부를 먼저 저축해 보세요."),
    FeatureSpec("savings_deposit_8w", "최근 8주 적금 납입액", True, "적금을 조금씩이라도 이어 가 보세요."),
    FeatureSpec("balance_savings", "적금 잔액", True, "적금을 깨지 않고 모아 보세요."),
    FeatureSpec("balance_checking", "입출금 잔액", True, "비상금을 통장에 남겨 두세요."),
    FeatureSpec("balance_total", "총 자산", True, "자산을 꾸준히 늘려 보세요."),
    FeatureSpec("has_savings", "적금 가입 여부", True, "적금 통장을 만들어 보세요."),
    # --- 부채/유동성 ---
    FeatureSpec("loan_outstanding", "남은 대출 잔액", False, "빚을 조금씩 줄여 나가 보세요."),
    FeatureSpec("debt_to_income", "월 소득 대비 부채", False, "소득에 비해 빚이 많아요. 새 대출은 미뤄 보세요."),
    FeatureSpec("has_loan", "대출 이용 여부", False, "꼭 필요한 대출인지 다시 생각해 보세요."),
    FeatureSpec("liquidity_weeks", "지출 대비 여유 자금(주)", True, "1~2주치 생활비는 남겨 두세요."),
    FeatureSpec("min_balance_8w", "최근 8주 최저 잔액", True, "잔액이 0에 가까워지지 않게 관리해 보세요."),
    FeatureSpec("low_balance_ratio", "잔액 부족 상태 비율", False, "잔액이 자주 바닥나요. 지출 속도를 늦춰 보세요."),
    FeatureSpec("balance_volatility", "잔액 변동성", False, "쓰는 돈의 기복을 줄이면 관리가 쉬워져요."),
    # --- 소비 습관 ---
    FeatureSpec("discretionary_ratio", "충동 소비 비중", False, "간식·뽑기 지출을 절반으로 줄여 보세요."),
    FeatureSpec("account_age_days", "계좌 개설 경과일", True, "통장을 오래 쓸수록 신용 기록이 두터워져요."),
)

FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)
FEATURE_LABELS: dict[str, str] = {s.name: s.label for s in FEATURE_SPECS}
FEATURE_BY_NAME: dict[str, FeatureSpec] = {s.name: s for s in FEATURE_SPECS}

NO_LATE_SENTINEL = 999.0  # 관측 기간 내 연체가 한 번도 없을 때의 경과일


@dataclass
class SourceData:
    """DB 에서 읽어 전처리를 마친 원천 데이터."""

    members: pd.DataFrame
    accounts: pd.DataFrame
    transactions: pd.DataFrame

    @property
    def student_ids(self) -> pd.Index:
        return pd.Index(self.members.loc[self.members["role"] == "STUDENT", "id"], name="member_id")

    @property
    def max_date(self) -> pd.Timestamp:
        return self.transactions["occured_at"].max().normalize()

    @property
    def min_date(self) -> pd.Timestamp:
        return self.transactions["occured_at"].min().normalize()


MEMBER_SQL = """
SELECT id, login_id, name, role, `class`, job, salary,
       cs_attendance, cs_overdue, cs_homework
FROM member
"""

ACCOUNT_SQL = "SELECT id AS account_id, member_id, account_type, balance, created_at FROM account"

TRANSACTION_SQL = """
SELECT t.id, t.account_id, a.member_id, a.account_type,
       t.amount, t.description, t.occured_at
FROM `transaction` t
JOIN account a ON a.id = t.account_id
"""


def load_source_frames(conn) -> SourceData:
    """DB 에서 member/account/transaction 을 읽고 파생 플래그를 붙인다."""
    from .db import fetch_df

    members = fetch_df(conn, MEMBER_SQL)
    accounts = fetch_df(conn, ACCOUNT_SQL)
    transactions = fetch_df(conn, TRANSACTION_SQL)
    return prepare_source(members, accounts, transactions)


def prepare_source(
    members: pd.DataFrame, accounts: pd.DataFrame, transactions: pd.DataFrame
) -> SourceData:
    members = members.copy()
    members["salary"] = pd.to_numeric(members["salary"], errors="coerce").fillna(0).astype(int)
    for col in ("cs_attendance", "cs_homework"):
        members[col] = pd.to_numeric(members[col], errors="coerce").fillna(1.0).astype(float)
    members["cs_overdue"] = pd.to_numeric(members["cs_overdue"], errors="coerce").fillna(0).astype(int)

    accounts = accounts.copy()
    accounts["created_at"] = pd.to_datetime(accounts["created_at"])
    accounts["balance"] = pd.to_numeric(accounts["balance"]).astype(int)

    tx = transactions.copy()
    tx["occured_at"] = pd.to_datetime(tx["occured_at"])
    tx["amount"] = pd.to_numeric(tx["amount"]).astype(int)
    tx = tx.sort_values(["account_id", "occured_at", "id"]).reset_index(drop=True)

    desc = tx["description"].fillna("")
    is_checking = tx["account_type"] == D.ACCOUNT_CHECKING
    tx["is_checking"] = is_checking
    tx["is_late"] = desc.isin(D.LATE_EVENT_DESCRIPTIONS) & is_checking
    tx["is_loan_late"] = (desc == D.LATE_FEE_LOAN) & is_checking
    tx["is_tax_late"] = (desc == D.LATE_FEE_TAX) & is_checking
    tx["is_tax_paid"] = desc.str.startswith(D.EXPENSE_TAX_PREFIX) & is_checking
    tx["is_salary"] = (desc == D.INCOME_SALARY) & is_checking
    tx["is_savings_deposit"] = (desc == D.EXPENSE_SAVINGS_DEPOSIT) & is_checking
    tx["is_loan_repay"] = (desc == D.EXPENSE_LOAN_REPAY) & is_checking
    tx["is_fine"] = (desc == D.EXPENSE_FINE) & is_checking
    tx["is_discretionary"] = desc.isin(D.DISCRETIONARY_DESCRIPTIONS) & is_checking

    not_income = desc.isin(D.TRANSFER_IN_DESCRIPTIONS)
    tx["income_amount"] = np.where(is_checking & (tx["amount"] > 0) & ~not_income, tx["amount"], 0)
    tx["expense_amount"] = np.where(is_checking & (tx["amount"] < 0), -tx["amount"], 0)
    tx["late_fee_amount"] = np.where(tx["is_late"], -tx["amount"], 0)
    tx["discretionary_amount"] = np.where(tx["is_discretionary"], -tx["amount"], 0)
    tx["savings_deposit_amount"] = np.where(tx["is_savings_deposit"], -tx["amount"], 0)
    tx["loan_repay_amount"] = np.where(tx["is_loan_repay"], -tx["amount"], 0)

    # 계좌별 시점 잔액(거래 직후 잔액). 잔액 = 거래액의 누적합.
    tx["run_balance"] = tx.groupby("account_id")["amount"].cumsum()

    return SourceData(members=members, accounts=accounts, transactions=tx)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, fill: float = 0.0) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return (numerator / denom).fillna(fill)


def build_features(
    src: SourceData,
    as_of: pd.Timestamp,
    observation_weeks: int = 8,
    member_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """기준일(as_of) 시점의 학생별 피처 행렬을 만든다."""
    as_of = pd.Timestamp(as_of).normalize() + pd.Timedelta(hours=23, minutes=59)
    win_start = as_of - pd.Timedelta(weeks=observation_weeks)

    members = src.members[src.members["role"] == "STUDENT"].set_index("id")
    if member_ids is not None:
        members = members.loc[members.index.intersection(pd.Index(member_ids))]
    idx = members.index.rename("member_id")

    tx = src.transactions
    past = tx[tx["occured_at"] <= as_of]
    obs = past[past["occured_at"] > win_start]
    obs_chk = obs[obs["is_checking"]]

    def agg(frame: pd.DataFrame, column: str, how: str = "sum") -> pd.Series:
        if frame.empty:
            return pd.Series(0.0, index=idx)
        series = frame.groupby("member_id")[column].agg(how)
        return series.reindex(idx).fillna(0.0)

    weeks = float(observation_weeks)
    f = pd.DataFrame(index=idx)

    # --- 비금융 지표 ---
    f["cs_attendance"] = members["cs_attendance"].to_numpy()
    f["cs_homework"] = members["cs_homework"].to_numpy()
    f["cs_overdue"] = members["cs_overdue"].to_numpy().astype(float)

    # --- 현금흐름 ---
    f["income_8w"] = agg(obs_chk, "income_amount")
    f["expense_8w"] = agg(obs_chk, "expense_amount")
    f["net_cashflow_8w"] = f["income_8w"] - f["expense_8w"]
    # 저축과 대출 상환은 '쓴 돈'이 아니라 '옮긴 돈'이므로 소비에서 뺀다.
    consumption = (
        f["expense_8w"] - agg(obs_chk, "savings_deposit_amount") - agg(obs_chk, "loan_repay_amount")
    ).clip(lower=0)
    f["consumption_income_ratio"] = _safe_ratio(consumption, f["income_8w"], fill=1.5).clip(0, 3)
    # 분모를 전체 지출로 잡으면 대출 상환액이 많은 학생일수록 비중이 낮게 나와
    # '충동 소비가 적은 학생'처럼 보인다. 순수 소비만 분모로 쓴다.
    f["discretionary_ratio"] = _safe_ratio(
        agg(obs_chk, "discretionary_amount"), consumption
    ).clip(0, 1)
    f["income_stability"] = (agg(obs_chk, "is_salary") / weeks).clip(0, 1.5)

    # --- 연체/상환 이력 ---
    f["late_events_8w"] = agg(obs_chk, "is_late")
    f["late_fee_amount_8w"] = agg(obs_chk, "late_fee_amount")
    f["fine_count_8w"] = agg(obs_chk, "is_fine")

    last_late = obs_chk[obs_chk["is_late"]].groupby("member_id")["occured_at"].max().reindex(idx)
    f["days_since_last_late"] = (
        (as_of - last_late).dt.total_seconds().div(86400).fillna(NO_LATE_SENTINEL).clip(0, NO_LATE_SENTINEL)
    )

    repay_cnt = agg(obs_chk, "is_loan_repay")
    loan_late_cnt = agg(obs_chk, "is_loan_late")
    f["repay_ontime_ratio"] = np.where(
        (repay_cnt + loan_late_cnt) > 0, repay_cnt / (repay_cnt + loan_late_cnt).replace(0, np.nan), 1.0
    )
    tax_cnt = agg(obs_chk, "is_tax_paid")
    tax_late_cnt = agg(obs_chk, "is_tax_late")
    f["tax_paid_ratio"] = np.where(
        (tax_cnt + tax_late_cnt) > 0, tax_cnt / (tax_cnt + tax_late_cnt).replace(0, np.nan), 1.0
    )

    # --- 저축/자산/부채 ---
    f["savings_deposit_8w"] = agg(obs_chk, "savings_deposit_amount")
    f["savings_rate"] = _safe_ratio(f["savings_deposit_8w"], f["income_8w"]).clip(0, 1)

    bal = (
        past.groupby(["member_id", "account_type"])["amount"].sum().unstack(fill_value=0)
        if not past.empty
        else pd.DataFrame()
    )
    for acc_type in (D.ACCOUNT_CHECKING, D.ACCOUNT_SAVINGS, D.ACCOUNT_LOAN):
        if acc_type not in bal.columns:
            bal[acc_type] = 0
    bal = bal.reindex(idx).fillna(0)

    f["balance_checking"] = bal[D.ACCOUNT_CHECKING].astype(float)
    f["balance_savings"] = bal[D.ACCOUNT_SAVINGS].astype(float)
    f["balance_total"] = f["balance_checking"] + f["balance_savings"]
    f["loan_outstanding"] = (-bal[D.ACCOUNT_LOAN]).clip(lower=0).astype(float)

    opened = src.accounts[src.accounts["created_at"] <= as_of]
    has_type = (
        opened.groupby(["member_id", "account_type"]).size().unstack(fill_value=0)
        if not opened.empty
        else pd.DataFrame()
    )
    for acc_type in (D.ACCOUNT_SAVINGS, D.ACCOUNT_LOAN):
        if acc_type not in has_type.columns:
            has_type[acc_type] = 0
    has_type = has_type.reindex(idx).fillna(0)
    f["has_savings"] = (has_type[D.ACCOUNT_SAVINGS] > 0).astype(float)
    f["has_loan"] = (f["loan_outstanding"] > 0).astype(float)

    monthly_income = (f["income_8w"] / weeks * 4).clip(lower=1)
    f["debt_to_income"] = (f["loan_outstanding"] / monthly_income).clip(0, 20)

    weekly_expense = (f["expense_8w"] / weeks).clip(lower=1)
    f["liquidity_weeks"] = (f["balance_checking"] / weekly_expense).clip(-5, 20)

    # --- 잔액 안정성 ---
    if obs_chk.empty:
        f["min_balance_8w"] = 0.0
        f["low_balance_ratio"] = 0.0
        f["balance_volatility"] = 0.0
    else:
        grouped = obs_chk.groupby("member_id")["run_balance"]
        f["min_balance_8w"] = grouped.min().reindex(idx).fillna(0.0)
        mean_bal = grouped.mean().reindex(idx).fillna(0.0)
        std_bal = grouped.std().reindex(idx).fillna(0.0)
        f["balance_volatility"] = _safe_ratio(std_bal, mean_bal.abs()).clip(0, 5)

        # '잔액 부족' 기준: 주급 1주치도 남아 있지 않은 상태
        threshold = members["salary"].reindex(idx).astype(float)
        low = obs_chk.assign(
            below=obs_chk["run_balance"].to_numpy()
            < obs_chk["member_id"].map(threshold).fillna(0).to_numpy()
        )
        f["low_balance_ratio"] = (
            low.groupby("member_id")["below"].mean().reindex(idx).fillna(0.0)
        )

    first_open = src.accounts.groupby("member_id")["created_at"].min().reindex(idx)
    f["account_age_days"] = (
        (as_of - first_open).dt.total_seconds().div(86400).fillna(0).clip(lower=0)
    )

    f = f[list(FEATURE_NAMES)].astype(float)
    f.insert(0, "as_of", as_of.normalize())
    return f


def build_labels(
    src: SourceData,
    as_of: pd.Timestamp,
    performance_weeks: int = 4,
    member_ids: Iterable[int] | None = None,
) -> pd.Series:
    """성과 윈도우 안에 연체(대출 상환 연체료 / 세금 체납 가산금)가 있으면 1."""
    as_of = pd.Timestamp(as_of).normalize() + pd.Timedelta(hours=23, minutes=59)
    end = as_of + pd.Timedelta(weeks=performance_weeks)

    idx = pd.Index(member_ids, name="member_id") if member_ids is not None else src.student_ids
    tx = src.transactions
    window = tx[(tx["occured_at"] > as_of) & (tx["occured_at"] <= end) & tx["is_late"]]
    bad = window.groupby("member_id").size().reindex(idx).fillna(0)
    return (bad > 0).astype(int).rename("y")


def snapshot_dates(src: SourceData, cfg: ModelConfig) -> list[pd.Timestamp]:
    """학습용 패널 데이터의 기준일 목록."""
    start = src.min_date + pd.Timedelta(weeks=cfg.observation_weeks)
    end = src.max_date - pd.Timedelta(weeks=cfg.performance_weeks)
    if end < start:
        return []
    return list(pd.date_range(start, end, freq=f"{cfg.snapshot_stride_days}D"))


def build_panel(src: SourceData, cfg: ModelConfig | None = None) -> pd.DataFrame:
    """여러 기준일에 대해 (피처, 정답) 쌍을 쌓아 학습용 패널 데이터를 만든다.

    같은 학생이 여러 기준일에 등장하므로, 학습/검증 분할은 반드시
    member_id 단위(GroupSplit)로 해야 누수가 없다.
    """
    cfg = cfg or ModelConfig()
    frames = []
    for as_of in snapshot_dates(src, cfg):
        feats = build_features(src, as_of, cfg.observation_weeks)
        # 관측 윈도우에 급여가 한 번도 없으면 아직 경제 활동 전으로 보고 제외한다.
        feats = feats[feats["income_stability"] > 0]
        labels = build_labels(src, as_of, cfg.performance_weeks, member_ids=feats.index)
        merged = feats.join(labels)
        frames.append(merged.reset_index())
    if not frames:
        return pd.DataFrame(columns=["member_id", "as_of", *FEATURE_NAMES, "y"])
    return pd.concat(frames, ignore_index=True)
