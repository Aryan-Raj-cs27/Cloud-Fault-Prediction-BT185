/** @type {import('tailwindcss').Config} */
const token = (name) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: ['class', '[data-theme="dark"]'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"IBM Plex Sans"', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      colors: {
        bg: token('bg'),
        surface: token('surface'),
        sunken: token('sunken'),
        line: token('line'),
        ink: token('ink'),
        ink2: token('ink2'),
        ink3: token('ink3'),
        accent: token('accent'),
        ok: token('ok'),
        warn: token('warn'),
        crit: token('crit'),
      },
      fontSize: { '2xs': ['11px', '16px'] },
      borderRadius: { DEFAULT: '6px' },
    },
  },
  plugins: [],
};
