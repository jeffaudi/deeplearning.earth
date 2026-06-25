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

**Taxonomies:** `tags`, `categories`, and `series` (see `config.toml`). The oriented-det series hub is at `/series/oriented-det/` (`content/series/oriented-det/_index.md`). Legacy YOLO/Kaggle posts use `categories: ["archive"]`.

**URLs:** Blog home is `/`. `/posts/` redirects to `/` on Netlify (individual posts stay at `/posts/<slug>/`).

## Layout overrides

- `layouts/index.html` — blog home (post list) instead of the theme’s avatar landing page
- `layouts/posts/li.html` — post title plus `description` on list pages
- `layouts/_default/about.html` — about page with avatar and social links
- `layouts/partials/header.html` — menu `target`/`rel` support and theme header build fix
- `layouts/partials/list.html` — taxonomy/series list pages use the same post list markup
- `layouts/posts/list.html` — `/posts/` section list (redirects to `/` on Netlify)
- `assets/css/custom.css` — single-column blog layout (post list, about page, post meta)
- `assets/css/medium-zoom.css` — overlay styles for click-to-zoom images ([medium-zoom](https://github.com/francoischalifour/medium-zoom))
- `assets/js/medium-zoom.min.js` + `assets/js/image-zoom.js` — click-to-zoom on post images
- `layouts/_default/_markup/render-image.html` — markdown images get class `post-image-zoom` (strips `#layoutTextWidth` from URLs)
