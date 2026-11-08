import { Moon, Sun } from 'lucide-react';
import { useTheme } from '../lib/theme.jsx';

/** One button: switches between light and dark (starts from the OS setting). */
export default function ThemeToggle() {
  const { mode, setPreference } = useTheme();
  const dark = mode === 'dark';
  return <button type="button" onClick={() => setPreference(dark ? 'light' : 'dark')}
    aria-label={dark ? 'Switch to light theme' : 'Switch to dark theme'} title={dark ? 'Light theme' : 'Dark theme'}
    className="relative grid h-8 w-8 place-items-center overflow-hidden rounded border border-line bg-surface text-ink2 transition-colors hover:bg-sunken hover:text-ink">
    <Sun size={15} className={`absolute transition-all duration-300 ${dark ? 'rotate-90 scale-50 opacity-0' : 'rotate-0 scale-100 opacity-100'}`} />
    <Moon size={15} className={`absolute transition-all duration-300 ${dark ? 'rotate-0 scale-100 opacity-100' : '-rotate-90 scale-50 opacity-0'}`} />
  </button>;
}
