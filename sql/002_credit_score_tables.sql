-- ---------------------------------------------------------------------------
-- 002_credit_score_tables.sql
--
-- 신용평가 결과를 적재하기 위한 테이블.
-- 기존 서비스 테이블(member/account/transaction/payroll)은 건드리지 않고,
-- 평가 산출물만 별도 테이블에 쌓는다.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS `credit_model_run` (
  `id`             int NOT NULL AUTO_INCREMENT,
  `model_version`  varchar(64)  NOT NULL COMMENT '모델 버전 (예: lr-20260825-1200)',
  `algorithm`      varchar(45)  NOT NULL COMMENT '채택 알고리즘',
  `trained_at`     datetime     NOT NULL,
  `n_train`        int          NOT NULL COMMENT '학습 표본 수',
  `n_test`         int          NOT NULL COMMENT '검증 표본 수',
  `bad_rate`       decimal(6,5) NOT NULL COMMENT '전체 연체(부실) 발생률',
  `auc`            decimal(6,5) NOT NULL COMMENT 'ROC AUC (holdout)',
  `ks`             decimal(6,5) NOT NULL COMMENT 'KS 통계량 (holdout)',
  `gini`           decimal(6,5) NOT NULL,
  `brier`          decimal(7,6) NOT NULL COMMENT '확률 보정 품질',
  `metrics_json`   json         DEFAULT NULL COMMENT '상세 지표/피처 중요도',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_credit_model_run_version` (`model_version`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
  COMMENT='신용평가 모델 학습 이력';

CREATE TABLE IF NOT EXISTS `credit_score` (
  `id`             int NOT NULL AUTO_INCREMENT,
  `member_id`      int          NOT NULL,
  `evaluated_at`   date         NOT NULL COMMENT '평가 기준일',
  `model_version`  varchar(64)  NOT NULL,
  `score`          int          NOT NULL COMMENT '모델 신용점수 (350~1000)',
  `grade`          tinyint      NOT NULL COMMENT '신용등급 1(최우수)~10(위험)',
  `grade_label`    varchar(20)  NOT NULL COMMENT '등급 이름',
  `pd`             decimal(7,6) NOT NULL COMMENT '향후 4주 연체 예측 확률',
  `rule_score`     int          NOT NULL COMMENT '교실 설명용 규칙기반 점수',
  `rule_grade`     tinyint      NOT NULL,
  `reason_json`    json         DEFAULT NULL COMMENT '점수 상승/하락 사유코드',
  `feature_json`   json         DEFAULT NULL COMMENT '평가에 사용된 입력 피처 스냅샷',
  `created_at`     datetime     NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_credit_score_member_date_model` (`member_id`, `evaluated_at`, `model_version`),
  KEY `idx_credit_score_member` (`member_id`),
  KEY `idx_credit_score_evaluated_at` (`evaluated_at`),
  CONSTRAINT `fk_credit_score_member` FOREIGN KEY (`member_id`) REFERENCES `member` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
  COMMENT='학생별 신용평가 결과 스냅샷';
