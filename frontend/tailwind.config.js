/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: { sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'] },
      colors: {
        ink: '#101828',
        midnight: '#0b1020',
        panel: '#11182b',
        muted: '#8691a9',
        accent: '#7c6cff',
        mint: '#43d9ad',
      },
      boxShadow: {
        glow: '0 0 28px rgba(124, 108, 255, 0.18)',
        card: '0 12px 40px rgba(12, 19, 40, 0.18)',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out both',
        'pulse-soft': 'pulseSoft 2s ease-in-out infinite',
        float: 'float 5s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: { from: { opacity: '0', transform: 'translateY(8px)' }, to: { opacity: '1', transform: 'translateY(0)' } },
        pulseSoft: { '0%, 100%': { opacity: '1' }, '50%': { opacity: '.55' } },
        float: { '0%, 100%': { transform: 'translateY(0)' }, '50%': { transform: 'translateY(-5px)' } },
      },
    },
  },
  plugins: [],
};
