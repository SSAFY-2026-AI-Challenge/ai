-- Move ai_recommendation.content / expected_effect onto ai_report, then empty ai_recommendation.
-- Multiple recommendations for one report are joined with newlines, ordered by recommendation id.

ALTER TABLE ai_report
  ADD COLUMN content TEXT NULL AFTER summary,
  ADD COLUMN expected_effect TEXT NULL AFTER content;

SET SESSION group_concat_max_len = 65535;

UPDATE ai_report AS report
JOIN (
  SELECT
    report_id,
    GROUP_CONCAT(content ORDER BY id SEPARATOR '\n') AS content,
    GROUP_CONCAT(expected_effect ORDER BY id SEPARATOR '\n') AS expected_effect
  FROM ai_recommendation
  GROUP BY report_id
) AS rec ON rec.report_id = report.id
SET
  report.content = rec.content,
  report.expected_effect = rec.expected_effect;

TRUNCATE TABLE ai_recommendation;
