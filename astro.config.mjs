import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://iondy.com',
  output: 'static',
  trailingSlash: 'always',
  devToolbar: { enabled: false },
  markdown: {
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark' },
      defaultColor: false,
    },
  },
});
