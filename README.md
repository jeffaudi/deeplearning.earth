# DeepLearning.Earth

Blog about Deep Learning for Earth Observation: [https://deeplearning.earth](https://deeplearning.earth)

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

Posts live under `content/posts/`. Static assets (images, favicon) are under `static/`.

## Theme override

`layouts/partials/header.html` overrides the theme header: the upstream merge in `hugo-coder` uses `.Site` inside `{{ with .Site }}`, which breaks the build. Consider fixing the same file in [jeffaudi/hugo-coder](https://github.com/jeffaudi/hugo-coder) and removing this override once the fork is updated.
