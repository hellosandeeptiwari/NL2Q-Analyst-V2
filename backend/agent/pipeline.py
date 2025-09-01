from backend.nl2sql.generator import generate_sql
from backend.db.engine import get_adapter
from backend.exports.csv_export import to_csv
from backend.exports.plotly_helper import auto_plot_spec
from backend.audit.audit_log import log_audit

class NLQueryNode:
    def __init__(self, job_id, nl, schema_snapshot, guardrail_cfg):
        self.job_id = job_id
        self.nl = nl
        self.schema_snapshot = schema_snapshot
        self.guardrail_cfg = guardrail_cfg

    def run(self, dry_run=False):
        generated = generate_sql(self.nl, self.schema_snapshot, self.guardrail_cfg)
        adapter = get_adapter()
        result = adapter.run(generated.sql, dry_run=dry_run)
        csv_path = to_csv(self.job_id, result.rows)
        log_audit(self.nl, generated.sql, result.execution_time, len(result.rows), result.error)
        import pandas as pd
        df = pd.DataFrame(result.rows)
        plotly_spec = auto_plot_spec(df)
        return {
            "sql": generated.sql,
            "rows": result.rows,
            "csv": csv_path,
            "plotly_spec": plotly_spec,
            "error": result.error
        }
