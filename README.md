# DeepLearning.Earth

Technical blog on oriented object detection for Earth Observation and the [oriented-det](https://github.com/DL4EO/oriented-det) package: [https://deeplearning.earth](https://deeplearning.earth)

- **Home (`/`)** — recent posts (blog index)
- **About (`/about/`)** — author bio, photo, company, and social links

Built with [Hugo](https://gohugo.io/) and the [hugo-coder](https://github.com/luizdepra/hugo-coder) theme (git submodule at `themes/hugo-coder`).

## Local development

```bash
git submodule update --init --recursive
hugo server -D
```

Open [http://localhost:1313/](http://localhost:1313/).

Requires Hugo **Extended** (for SCSS). The Netlify config pins `HUGO_VERSION` in `netlify.toml`.

## Deploy (Netlify)

Settings are defined in `netlify.toml`:

- **Build command:** `hugo --minify`
- **Publish directory:** `public`
- **Environment:** `HUGO_VERSION`, `GIT_SUBMODULE_STRATEGY=recursive`

After pushing to `main`, Netlify installs Hugo, initializes the theme submodule, and builds. If the Netlify UI still overrides build settings, either clear those overrides or align them with `netlify.toml`.

## Content

Posts live under `content/posts/`. The about page is `content/about.md` (layout: `layouts/_default/about.html`). Static assets (images, favicon) are under `static/`.

## Layout overrides

- `layouts/index.html` — blog home (post list) instead of the theme’s avatar landing page
- `layouts/_default/about.html` — about page with avatar and social links
- `layouts/partials/header.html` — fixes theme header build issue: upstream `hugo-coder` uses `.Site` inside `{{ with .Site }}`, which breaks the build. Consider fixing the same file in [jeffaudi/hugo-coder](https://github.com/jeffaudi/hugo-coder) and removing this override once the fork is updated.
