-- Classroom-level economy snapshot. There is no class table in fin_edu;
-- class_id is stored as int (member.class remains the varchar label like '3-1').
-- Money totals follow account.balance / transaction.amount (int).
-- Averages keep two decimal places; rates can be signed (week-over-week).
CREATE TABLE IF NOT EXISTS `economic_metric` (
  `id` int NOT NULL AUTO_INCREMENT,
  `class_id` int NOT NULL COMMENT '학급 식별자',
  `total_money` int NOT NULL COMMENT '학급 총화폐량',
  `average_asset` decimal(12,2) NOT NULL COMMENT '1인당 평균 자산',
  `weekly_transaction_volume` int NOT NULL COMMENT '주간 거래량',
  `average_consumption` decimal(12,2) NOT NULL COMMENT '1인당 평균 소비',
  `consumption_change_rate` decimal(8,4) NOT NULL COMMENT '소비 변화율',
  `transaction_change_rate` decimal(8,4) NOT NULL COMMENT '거래 변화율',
  `loan_change_rate` decimal(8,4) NOT NULL COMMENT '대출 변화율',
  `saving_rate` decimal(8,4) NOT NULL COMMENT '저축률',
  `wealth_gap` decimal(8,4) NOT NULL COMMENT '빈부격차',
  `measured_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '측정 시각',
  PRIMARY KEY (`id`),
  KEY `idx_economic_metric_class_measured` (`class_id`, `measured_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
COMMENT='학급 경제 지표 스냅샷';
