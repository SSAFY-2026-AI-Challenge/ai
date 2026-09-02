-- contributions is a derived explain payload (score_lift/direction).
-- It is still computed in memory and returned by the scoring API,
-- but is not stored on ai_report. features + summary/content/expected_effect
-- are enough to persist a report snapshot.
ALTER TABLE ai_report DROP COLUMN contributions;
