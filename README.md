# ScraperBot

Instagram lead-generation tooling used to discover, qualify, and track potential influencer/partner accounts for outreach. This repo contains two related but independently runnable bots.

## Projects

### [`ScrapeInstagramBot/ScrapeInstagramBot-main`](ScrapeInstagramBot/ScrapeInstagramBot-main)
Discovers new Instagram accounts by crawling the following-lists of seed accounts, qualifies them against configurable criteria (followers, location, profession keywords, optional AI verification), and appends qualified leads to a Google Sheet.

### [`SupplierScraping`](SupplierScraping)
Same core scraping/qualification engine as above, plus a CSV import workflow (`importMembersFromCsv.ts`) for cross-referencing scraped leads against a known member list.

Both bots share the same architecture:
- **Auth**: Playwright-driven Instagram login with cookie persistence (`src/auth`)
- **Scraping**: profile/follower extraction via Instagram's internal API with DOM fallback (`src/core`)
- **Qualification**: keyword/context-based profession and location detection, with optional OpenAI verification
- **Output**: qualified leads appended to Google Sheets; session summaries posted to Discord
- **Config-driven**: per-campaign settings live under `src/config/<campaign-name>/` (follower limits, target locations, professions, etc.)

## Getting started

Each bot has its own setup instructions, environment variables, and usage guide — see:
- [`ScrapeInstagramBot/ScrapeInstagramBot-main/README.md`](ScrapeInstagramBot/ScrapeInstagramBot-main/README.md)
- [`SupplierScraping/README.md`](SupplierScraping/README.md)

## Working with this code

- Never commit `.env`, `service_account.json`, `cookies.json`, or any generated CSV/log output — these contain real credentials or scraped personal data. See each bot's `.gitignore` entries and `.env.example` for what's expected locally.
- Use the `google_service_account_example.json` in each bot's folder as a template for your own Google service account credentials.
- Respect Instagram's terms of service and the built-in rate limiting / off-peak-hours logic — don't bypass it (`FORCE_RUN`) without understanding why it's there.
