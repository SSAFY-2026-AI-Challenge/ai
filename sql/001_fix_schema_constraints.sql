-- ---------------------------------------------------------------------------
-- 001_fix_schema_constraints.sql
--
-- fin_edu 스키마에는 복사-붙여넣기로 잘못 들어간 UNIQUE 제약이 남아 있다.
--
--   1) transaction.`member_id_UNIQUE` UNIQUE (account_id)
--      -> 한 계좌에 거래를 단 1건만 넣을 수 있어 거래 원장이 성립하지 않는다.
--   2) payroll.`member_id_UNIQUE` UNIQUE (member_id)
--      -> 한 학생에게 급여를 평생 1회만 지급할 수 있다.
--
-- 신용평가는 "계좌별 거래 흐름"을 입력으로 쓰기 때문에 1)은 반드시 제거해야 한다.
--
-- 주의: 두 UNIQUE 인덱스는 각각 외래키(fk_transaction_account, payroll_member_id_fk)를
-- 떠받치고 있다. 그래서 대체 인덱스를 먼저 만든 뒤에 UNIQUE 를 제거한다.
-- 이미 정리된 환경에서 다시 실행해도 안전하다(존재 여부를 확인한 뒤 실행).
-- ---------------------------------------------------------------------------

-- 1) 대체 인덱스 먼저 생성 ---------------------------------------------------
SET @sql := (
    SELECT IF(
        EXISTS (
            SELECT 1 FROM information_schema.statistics
            WHERE table_schema = DATABASE()
              AND table_name = 'transaction'
              AND index_name = 'idx_transaction_account_id'
        ),
        'SELECT "transaction.idx_transaction_account_id 이미 존재" AS msg',
        'ALTER TABLE `transaction` ADD INDEX `idx_transaction_account_id` (`account_id`)'
    )
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @sql := (
    SELECT IF(
        EXISTS (
            SELECT 1 FROM information_schema.statistics
            WHERE table_schema = DATABASE()
              AND table_name = 'transaction'
              AND index_name = 'idx_transaction_occured_at'
        ),
        'SELECT "transaction.idx_transaction_occured_at 이미 존재" AS msg',
        'ALTER TABLE `transaction` ADD INDEX `idx_transaction_occured_at` (`occured_at`)'
    )
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @sql := (
    SELECT IF(
        EXISTS (
            SELECT 1 FROM information_schema.statistics
            WHERE table_schema = DATABASE()
              AND table_name = 'payroll'
              AND index_name = 'idx_payroll_member_id'
        ),
        'SELECT "payroll.idx_payroll_member_id 이미 존재" AS msg',
        'ALTER TABLE `payroll` ADD INDEX `idx_payroll_member_id` (`member_id`)'
    )
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 2) 잘못된 UNIQUE 제거 ------------------------------------------------------
SET @sql := (
    SELECT IF(
        EXISTS (
            SELECT 1 FROM information_schema.statistics
            WHERE table_schema = DATABASE()
              AND table_name = 'transaction'
              AND index_name = 'member_id_UNIQUE'
        ),
        'ALTER TABLE `transaction` DROP INDEX `member_id_UNIQUE`',
        'SELECT "transaction.member_id_UNIQUE 없음 - 건너뜀" AS msg'
    )
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @sql := (
    SELECT IF(
        EXISTS (
            SELECT 1 FROM information_schema.statistics
            WHERE table_schema = DATABASE()
              AND table_name = 'payroll'
              AND index_name = 'member_id_UNIQUE'
        ),
        'ALTER TABLE `payroll` DROP INDEX `member_id_UNIQUE`',
        'SELECT "payroll.member_id_UNIQUE 없음 - 건너뜀" AS msg'
    )
);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
