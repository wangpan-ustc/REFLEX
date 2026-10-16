# Security

Supply API keys through process environment variables or the CCAA scripts'
hidden terminal prompt. Keep credentials and generated run artifacts out of
Git.

REFLEX executes generated Python. Run experiments in an isolated environment
with only the files and permissions they need. The strict CCAA runner checks
the candidate AST and limits imports, but Python-level checks are not a full
operating-system sandbox.

Review generated datasets and run logs before publishing them.

Report vulnerabilities privately to the repository maintainer with a minimal
reproducer. Exclude live credentials from reports.
