import { useEffect, useState } from 'react';
import { Activity, BarChart3, BookOpen, LayoutGrid, LogOut, Menu, Network, PanelLeftClose, PanelLeftOpen, X } from 'lucide-react';
import ErrorBoundary from './components/ErrorBoundary.jsx';
import ThemeToggle from './components/ThemeToggle.jsx';
import { useMonitor } from './hooks/useMonitor.js';
import { AuthProvider, useAuth } from './lib/auth.jsx';
import { ThemeProvider } from './lib/theme.jsx';
import Overview from './pages/Overview.jsx';
import LiveInference from './pages/LiveInference.jsx';
import Login from './pages/Login.jsx';
import Models from './pages/Models.jsx';
import Scheduling from './pages/Scheduling.jsx';
import Method from './pages/Method.jsx';

const ROUTES = [
  { id: 'overview', label: 'Overview', Icon: LayoutGrid, Page: Overview },
  { id: 'inference', label: 'Live inference', Icon: Activity, Page: LiveInference },
  { id: 'models', label: 'Models', Icon: BarChart3, Page: Models },
  { id: 'scheduling', label: 'FARS scheduling', Icon: Network, Page: Scheduling },
  { id: 'method', label: 'Data & method', Icon: BookOpen, Page: Method },
];

const readRoute = () => ROUTES.find((r) => `#/${r.id}` === window.location.hash)?.id ?? 'overview';

function useRoute() {
  const [route, setRoute] = useState(readRoute);
  useEffect(() => {
    const onHash = () => { setRoute(readRoute()); window.scrollTo({ top: 0 }); };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  return [route, (id) => { window.location.hash = `/${id}`; }];
}

function usePersistentFlag(key, initial) {
  const [value, setValue] = useState(() => { try { return localStorage.getItem(key) === '1' ? true : localStorage.getItem(key) === '0' ? false : initial; } catch { return initial; } });
  const set = (v) => { setValue(v); try { localStorage.setItem(key, v ? '1' : '0'); } catch { /* storage unavailable */ } };
  return [value, set];
}

function Brand({ collapsed }) {
  return <div className="flex h-12 items-center gap-2 overflow-hidden px-[18px]">
    <svg width="20" height="20" viewBox="0 0 16 16" className="shrink-0" aria-hidden><rect width="16" height="16" rx="3" fill="rgb(var(--accent))" /><path d="M3 10h2l2-5 2 7 2-4h2" stroke="white" strokeWidth="1.5" fill="none" /></svg>
    <div className={`whitespace-nowrap leading-tight transition-opacity duration-200 ${collapsed ? 'opacity-0' : 'opacity-100'}`}><div className="text-[13px] font-semibold">FaultAware</div><div className="text-2xs text-ink3">Console · BT-185</div></div>
  </div>;
}

function NavItems({ route, collapsed, onNavigate }) {
  return <nav className="flex flex-col gap-0.5 p-2" aria-label="Main">
    {ROUTES.map(({ id, label, Icon }) => {
      const active = route === id;
      return <a key={id} href={`#/${id}`} onClick={onNavigate} aria-current={active ? 'page' : undefined} aria-label={collapsed ? label : undefined}
        className={`group relative flex h-9 items-center gap-3 rounded px-[11px] text-[13px] transition-colors duration-150 ${active ? 'bg-sunken font-medium text-ink' : 'text-ink2 hover:bg-sunken hover:text-ink'}`}>
        <span className={`absolute left-0 top-1.5 bottom-1.5 w-[2px] rounded-full bg-accent transition-opacity duration-200 ${active ? 'opacity-100' : 'opacity-0'}`} />
        <Icon size={16} className={`shrink-0 transition-colors ${active ? 'text-accent' : 'text-ink3 group-hover:text-ink2'}`} />
        <span className={`whitespace-nowrap transition-opacity duration-200 ${collapsed ? 'pointer-events-none opacity-0' : 'opacity-100'}`}>{label}</span>
        {collapsed && <span role="tooltip" className="pointer-events-none absolute left-full z-50 ml-2 whitespace-nowrap rounded border border-line bg-surface px-2 py-1 text-xs text-ink opacity-0 shadow-md transition-opacity duration-150 group-hover:opacity-100 group-focus-visible:opacity-100">{label}</span>}
      </a>;
    })}
  </nav>;
}

function UserMenu({ user, onSignOut }) {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!open) return undefined;
    const close = (e) => { if (!e.target.closest?.('[data-user-menu]')) setOpen(false); };
    const esc = (e) => e.key === 'Escape' && setOpen(false);
    document.addEventListener('mousedown', close); document.addEventListener('keydown', esc);
    return () => { document.removeEventListener('mousedown', close); document.removeEventListener('keydown', esc); };
  }, [open]);
  return <div className="relative" data-user-menu>
    <button onClick={() => setOpen(!open)} aria-haspopup="menu" aria-expanded={open}
      className="flex h-8 items-center gap-2 rounded border border-line bg-surface pl-1 pr-2.5 text-xs transition-colors hover:bg-sunken">
      <span className="grid h-6 w-6 place-items-center rounded-[4px] bg-accent/15 text-2xs font-semibold uppercase text-accent">{user.slice(0, 2)}</span>
      <span className="hidden sm:inline">{user}</span>
    </button>
    <div role="menu" className={`absolute right-0 mt-1 w-48 origin-top-right rounded border border-line bg-surface py-1 shadow-lg transition-all duration-150 ${open ? 'scale-100 opacity-100' : 'pointer-events-none scale-95 opacity-0'}`}>
      <div className="border-b border-line px-3 py-2 text-2xs text-ink3">Signed in as <span className="font-medium text-ink">{user}</span></div>
      <button role="menuitem" onClick={onSignOut} className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs text-ink2 hover:bg-sunken hover:text-ink"><LogOut size={14} /> Sign out</button>
    </div>
  </div>;
}

function Console() {
  const { user, signOut } = useAuth();
  const monitor = useMonitor();
  const [route, go] = useRoute();
  const [collapsed, setCollapsed] = usePersistentFlag('fa-nav-collapsed', false);
  const [drawer, setDrawer] = useState(false);
  const current = ROUTES.find((r) => r.id === route);
  const offline = monitor.connection === 'offline';

  return <div className="min-h-screen bg-bg text-ink">
    <aside className={`fixed inset-y-0 left-0 z-30 hidden flex-col border-r border-line bg-surface transition-[width] duration-200 ease-out lg:flex ${collapsed ? 'w-[60px]' : 'w-56'}`}>
      <Brand collapsed={collapsed} />
      <NavItems route={route} collapsed={collapsed} />
      <div className="mt-auto border-t border-line p-2">
        <button onClick={() => setCollapsed(!collapsed)} aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'} title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className="flex h-9 w-full items-center gap-3 rounded px-[11px] text-xs text-ink3 transition-colors hover:bg-sunken hover:text-ink">
          {collapsed ? <PanelLeftOpen size={16} className="shrink-0" /> : <PanelLeftClose size={16} className="shrink-0" />}
          <span className={`whitespace-nowrap transition-opacity duration-200 ${collapsed ? 'opacity-0' : 'opacity-100'}`}>Collapse</span>
        </button>
      </div>
    </aside>

    <div className={`fixed inset-0 z-40 lg:hidden ${drawer ? '' : 'pointer-events-none'}`} aria-hidden={!drawer}>
      <button tabIndex={drawer ? 0 : -1} aria-label="Close menu" onClick={() => setDrawer(false)} className={`absolute inset-0 bg-black/40 transition-opacity duration-200 ${drawer ? 'opacity-100' : 'opacity-0'}`} />
      <aside className={`relative flex h-full w-64 flex-col border-r border-line bg-surface transition-transform duration-200 ease-out ${drawer ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="flex items-center justify-between"><Brand collapsed={false} /><button aria-label="Close menu" onClick={() => setDrawer(false)} className="mr-2 p-2 text-ink3 hover:text-ink"><X size={16} /></button></div>
        <NavItems route={route} collapsed={false} onNavigate={() => setDrawer(false)} />
      </aside>
    </div>

    <div className={`transition-[padding] duration-200 ease-out ${collapsed ? 'lg:pl-[60px]' : 'lg:pl-56'}`}>
      <header className="sticky top-0 z-20 flex h-12 items-center gap-3 border-b border-line bg-surface/90 px-4 backdrop-blur">
        <button aria-label="Open menu" onClick={() => setDrawer(true)} className="-ml-1 p-1 text-ink2 lg:hidden"><Menu size={18} /></button>
        <div className="min-w-0 flex-1 truncate text-xs text-ink3"><span className="hidden sm:inline">hdfs-v1<span className="mx-1.5">/</span>test-period replay<span className="mx-1.5">/</span></span><span className="font-medium text-ink">{current.label}</span></div>
        <span className="hidden items-center gap-1.5 rounded border border-line px-2 py-1 text-2xs text-ink2 sm:inline-flex" title="API status">
          <span className={`h-1.5 w-1.5 rounded-full ${{ online: 'bg-ok', degraded: 'bg-warn', offline: 'bg-crit' }[monitor.connection] ?? 'bg-ink3'}`} />
          {{ online: 'API online', degraded: 'Models pending', offline: 'API offline' }[monitor.connection] ?? 'Connecting'}
        </span>
        <ThemeToggle />
        <UserMenu user={user} onSignOut={signOut} />
      </header>
      <main className="mx-auto max-w-[1400px] px-4 py-5 sm:px-6">
        {monitor.health?.status === 'degraded' && <div className="fa-enter mb-4 rounded border border-warn/30 bg-warn/10 px-3 py-2 text-xs text-ink">{monitor.health.detail}</div>}
        {offline && <div className="fa-enter mb-4 rounded border border-crit/30 bg-crit/10 px-3 py-2 text-xs text-ink">Cannot reach the API. Start it with <span className="font-mono">make api</span> (or <span className="font-mono">.\make.ps1 api</span>).</div>}
        <div key={route} className="fa-page"><current.Page monitor={monitor} go={go} /></div>
      </main>
    </div>
  </div>;
}

function Gate() {
  const { status } = useAuth();
  if (status === 'checking') return <div className="grid min-h-screen place-items-center bg-bg"><span className="h-5 w-5 animate-spin rounded-full border-2 border-line border-t-accent" aria-label="Loading" /></div>;
  return status === 'authed' ? <Console /> : <Login />;
}

export default function App() {
  return <ErrorBoundary><ThemeProvider><AuthProvider><Gate /></AuthProvider></ThemeProvider></ErrorBoundary>;
}
