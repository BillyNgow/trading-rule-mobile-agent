# Trading Rule Hosted Mobile Agent

Built for iPhone 11.

Outputs:
- `output/index.html` = mobile card dashboard for GitHub Pages
- `output/daily_rule_report.html` = desktop table view
- `output/daily_rule_report.csv` = CSV backup

GitHub repository secrets needed:
- `ALPHA_VANTAGE_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `PUBLIC_REPORT_URL`

Recommended repo setting:
- Public repository
- Settings > Pages > Source: GitHub Actions

Schedule:
- 6:30 AM Malaysia time, Monday to Friday
- Cron: `30 22 * * 1-5`

This is not a buy/sell signal system. It only ranks setup quality.
