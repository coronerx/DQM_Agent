# Discord review service

This service turns each course submission into a Discord review card. An
authorized reviewer can:

- approve immediately;
- approve with written feedback; or
- request revision with required feedback.

Only an approved stage unlocks the next stage in the course page.

## Configuration

1. Create a Discord application and bot, then install the bot in the server.
2. Give the bot permission to view the target channel and send messages.
3. Copy `.env.example` to `.env` and fill in:
   - `DISCORD_APPLICATION_ID`
   - `DISCORD_PUBLIC_KEY`
   - `DISCORD_CHANNEL_ID`
   - `DISCORD_REVIEWER_USER_IDS` (comma-separated Discord user IDs)
   - `DISCORD_BOT_TOKEN`
   - `LEARNER_API_KEY` (a long random value shared only with the course page)
   - `ALLOWED_ORIGINS` (the exact origin hosting the course page)
4. Install dependencies and start the service:

   ```bash
   python3 -m pip install -r review_server/requirements.txt
   python3 -m review_server.app
   ```

5. Deploy the service behind HTTPS. In the Discord Developer Portal, set the
   **Interactions Endpoint URL** to:

   ```text
   https://YOUR-SERVICE.example/discord/interactions
   ```

   This is not an OAuth redirect URI.

6. In `demo/daily_work_log.html`, configure the public service URL, learner
   name/ID, and `LEARNER_API_KEY`.

For a container deployment, build from the repository root:

```bash
docker build -f review_server/Dockerfile -t collaboration-review .
```

Mount persistent storage and set `DATABASE_PATH=/data/reviews.db`; otherwise
review state may disappear when the container restarts.

## Security boundaries

- Never commit `.env`, the bot token, or the learner API key.
- `DISCORD_PUBLIC_KEY`, application ID, channel ID, and Discord user IDs are
  identifiers, not bot credentials.
- Discord interaction requests are accepted only after Ed25519 signature
  verification.
- Review decisions are accepted only from IDs in
  `DISCORD_REVIEWER_USER_IDS`.
- The learner API accepts only the configured `X-Learner-Key`.
- Configure `ALLOWED_ORIGINS` narrowly; do not use `*` in production.
