# Daily Financial Systems News Digest (Ollama + GitHub Actions)

This project reads eight LiveMint RSS feeds, deduplicates new articles, asks a local Ollama model to assess relevance, and emails a concise HTML digest.

## Relevance policy

The model is asked to look for substantive connections to:

1. **Financial system design** — financial architecture, regulation, market/payment infrastructure, digital public infrastructure, KYC/identity, consent-based data sharing, interoperability, resilience and consumer protection.
2. **Financial inclusion** — access, affordability and suitable use of financial services by underserved households, rural communities, women, informal workers and small businesses.
3. **Micro-enterprises** — micro/very small firms, own-account workers and small merchants; working capital, credit, payments, cash flow, insurance, formalisation, productivity and livelihoods.
4. **Banking and monetary policy** — RBI/central-bank policy, rates, liquidity, monetary transmission, inflation when policy-relevant, banking/NBFC regulation, deposits, credit allocation, asset quality and financial stability.

Stories are not included just because they mention business, money, banks or technology. Routine share-price, earnings, IPO and market stories are excluded unless they have a concrete systemic, inclusion, micro-enterprise or policy implication. Serious opinion analysis can qualify and is labelled as an opinion/analysis where the RSS section identifies it.

The default inclusion threshold is **6/10**. Change `MIN_RELEVANCE_SCORE` in the workflow to `5` for a broader digest or `7` for a tighter one.

## RSS feeds

The eight user-provided LiveMint feeds are listed in `feeds.json`. Edit that file to add or remove sources.

## Setup (GitHub Free)

1. To keep the cloud compute free, create a **public** GitHub repository and upload the contents of this folder, preserving `.github/workflows/daily-news-digest.yml`. The source code and configuration will be public, but GitHub Secrets and the emailed digest are not published by this workflow. Do not add private data to the repository. If you prefer a private repository, see the billing note below.
2. In the repository, open **Settings → Secrets and variables → Actions → New repository secret** and add:
   - `EMAIL_FROM`: Gmail address used to send the message.
   - `EMAIL_TO`: destination email address (can be the same address).
   - `SMTP_USERNAME`: Gmail address used to authenticate (usually same as `EMAIL_FROM`).
   - `SMTP_PASSWORD`: a Google App Password, not your normal Gmail password.
3. For the simplest Gmail SMTP setup, enable 2-Step Verification and create a Google App Password. Store it only as `SMTP_PASSWORD`; do not put it in a file or commit it. Google labels App Passwords a less-secure sign-in method and recommends Sign in with Google when available; use a dedicated sending account if you choose this SMTP method.
4. In **Settings → Actions → General**, ensure the workflow has permission to read and write repository contents (the workflow also requests `contents: write`).
5. Open **Actions → Daily Financial Systems News Digest → Run workflow** to test it manually.
6. The workflow is scheduled for **8:00 AM Asia/Kolkata** every day. Change `cron: "0 8 * * *"` and keep `timezone: "Asia/Kolkata"` to choose another time.

GitHub Actions supports timezone-aware scheduled workflows. The workflow uses the current major releases of the official checkout and Python setup actions. Scheduled runs execute from the repository's default branch. GitHub may delay scheduled starts during periods of high load.

## Free-tier notes

- GitHub's standard hosted runners are free for public repositories. A public `ubuntu-latest` runner currently has 4 CPU cores and 16 GB RAM, which is a better fit for a 3B CPU model. The repository source code is public, so keep personal information and all credentials out of tracked files.
- Private repositories use the account's included GitHub Actions compute allowance. On GitHub Free, the current included compute amount is $12/month; standard Linux runner usage is deducted from it. The private-repository `ubuntu-latest` runner currently has 2 CPU cores and 8 GB RAM. Check the account's billing page before relying on a private repository for daily use.
- Public repositories' scheduled workflows may be disabled after 60 days with no repository activity. Check the Actions tab if a schedule stops.
- The standard Ubuntu hosted runner is CPU-only for this workflow. `qwen2.5:3b` is used as a practical starting point for nuanced filtering. Model download and CPU inference time depend on feed volume and runner performance.
- The workflow downloads the model on each fresh runner. No GPU or persistent Ollama server is assumed.
- The source code and public RSS feed list are visible in a public repository; keep email addresses, app passwords, and other personal data only in GitHub Secrets. The digest itself is emailed and is not committed to the repository.
- The `state/seen.json` file is committed after a successful run so articles are not repeatedly processed. If the email step fails, the run fails and the processed state is not committed, allowing retry on a later run.
- By default, no email is sent on a day with no matching stories. Set `SEND_EMPTY_DIGEST` to `"true"` in the workflow if you prefer a daily no-news note.

## Local test (optional)

Install Ollama and Python dependencies, start Ollama, pull the model, then run:

```bash
pip install -r requirements.txt
ollama pull qwen2.5:3b
ollama serve
```

In another terminal, set the email environment variables and run:

```bash
python daily_digest.py
```

## Important behavior

- The script considers articles published within the last 48 hours that have not been processed before. This overlap helps catch delayed RSS updates.
- It caps processing at 80 candidates per run. Increase `MAX_ARTICLES` if needed, keeping runtime limits in mind.
- RSS descriptions are used for AI assessment; the script does not scrape full article pages or bypass paywalls. Summaries can therefore be limited when a feed provides only a short excerpt.
- AI relevance scores and summaries can be imperfect. Each selected story links to the original article so you can verify the source.
- Feed failures are logged and shown in the email when a digest is sent. The workflow fails only if all feeds fail or a critical processing/email step fails.
