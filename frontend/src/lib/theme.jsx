import { createContext, useContext, useEffect, useState } from 'react';

// Chart colours as hex (SVG attributes cannot read CSS variables). Categorical slots come from
// a CVD-validated palette and are assigned to entities in a FIXED order.
const PALETTES = {
  light: {
    series: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'],
    seq: ['#eef3fa', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95'],
    accent: '#2a78d6', muted: '#c6ccd4', ink3: '#767d87', crit: '#c43434', grid: '#dee2e7', surface: '#ffffff',
  },
  dark: {
    series: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
    seq: ['#1d222a', '#1b2d47', '#184f95', '#256abf', '#3987e5', '#6da7ec', '#9ec5f4'],
    accent: '#3987e5', muted: '#3b424c', ink3: '#787f89', crit: '#e66767', grid: '#2c3138', surface: '#16191d',
  },
};

// Model families -> fixed categorical slot (colour follows the entity, never its rank).
export const FAMILY_SLOT = { trivial: 6, 'classical ML': 1, 'deep learning': 2, unsupervised: 3, fusion: 0 };

const ThemeContext = createContext({ mode: 'light', palette: PALETTES.light, setPreference: () => {} });

function systemMode() {
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function readPreference() {
  try { return localStorage.getItem('fa-theme') || 'system'; } catch { return 'system'; }
}

export function ThemeProvider({ children }) {
  const [preference, setPreferenceState] = useState(readPreference);
  const [system, setSystem] = useState(systemMode);
  useEffect(() => {
    const mq = window.matchMedia?.('(prefers-color-scheme: dark)');
    const onChange = () => setSystem(systemMode());
    mq?.addEventListener?.('change', onChange);
    return () => mq?.removeEventListener?.('change', onChange);
  }, []);
  const mode = preference === 'system' ? system : preference;
  useEffect(() => {
    const root = document.documentElement;
    if (preference === 'system') root.removeAttribute('data-theme');
    else root.setAttribute('data-theme', preference);
  }, [preference]);
  const setPreference = (value) => {
    setPreferenceState(value);
    try { localStorage.setItem('fa-theme', value); } catch { /* storage unavailable */ }
  };
  return <ThemeContext.Provider value={{ mode, preference, palette: PALETTES[mode], setPreference }}>{children}</ThemeContext.Provider>;
}

export const useTheme = () => useContext(ThemeContext);
