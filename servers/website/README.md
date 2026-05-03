# Stego Challenge Website

TODO(hadriano) please review this jekyll website.

Jekyll site for the competition. Designed to be deployed on GitHub Pages.

## Configuration

Before deploying, edit `_config.yml` and set the API URL:

```yaml
api_url: "http://12.34.56.78:8000"
```

This value is referenced as `{{ site.api_url }}` throughout the site templates (the Submit page, Python examples, etc). Replace it with the public IP/domain of your running verification API.

## Adding challenge programs

Drop files in `_challenge_programs/`. Each file is a Markdown file with YAML front matter:

```yaml
---
title: "Program N — Short Description"
bitstring_prefix: "010110..."
code: |
  import foo
  ...
---
```

The index page renders all programs from this collection automatically.

## Local preview

```bash
cd servers/website
bundle install
bundle exec jekyll serve
```

Then open `http://localhost:4000`.

## Deploy to GitHub Pages

1. **Deploy the API first** (see `servers/api/README.md`) and note the public IP.

2. **Create a new public repo** for the competition site (e.g. `stego-challenge`). This keeps the cipher and internal code private in the main repo.

3. **Copy the website files** into the new repo:

   ```bash
   cp -r servers/website/* /path/to/stego-challenge/
   ```

4. **Set the API URL** in `_config.yml`:

   ```yaml
   api_url: "http://<YOUR_API_PUBLIC_IP>:8000"
   ```

5. **Push and enable Pages:**

   ```bash
   cd /path/to/stego-challenge
   git init && git add -A && git commit -m "initial site"
   git remote add origin git@github.com:<your-org>/stego-challenge.git
   git push -u origin main
   ```

   Then in the repo's GitHub Settings > Pages, set source to **Deploy from a branch** > `main` / `/ (root)`.

6. The site will be live at `https://<your-org>.github.io/stego-challenge/` within a minute or two. If using a custom domain, set `url` and `baseurl` in `_config.yml` accordingly.

## What NOT to put in the public repo

- The `ciphers/` directory
- The `decoder.py` file
- The `servers/api/` directory
- Anything from the main research repo that reveals the encoding scheme
