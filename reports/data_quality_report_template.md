# Data Quality Report

- Total records: {{total}}
{{#each columns}}
- {{this.name}}: {{this.missing}} missing ({{this.pct}}%)
{{/each}}
