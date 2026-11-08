import { useMemo, useState } from 'react';
import { Eye, EyeOff, Lock, LogIn, ShieldCheck } from 'lucide-react';
import { useAuth } from '../lib/auth.jsx';
import ThemeToggle from '../components/ThemeToggle.jsx';

// Deterministic node grid: a few nodes "degrading" (risk rising) and one down, echoing FARS.
function ClusterArt() {
  const cells = useMemo(() => Array.from({ length: 96 }, (_, i) => {
    const risk = [17, 18, 41, 66].includes(i) ? 3 : [16, 40, 42, 65, 67].includes(i) ? 2 : (i * 37) % 11 === 0 ? 1 : 0;
    return { i, risk, down: i === 53 };
  }), []);
  const fill = ['rgb(var(--ink) / 0.07)', 'rgb(var(--accent) / 0.25)', 'rgb(var(--accent) / 0.55)', 'rgb(var(--accent))'];
  return <svg viewBox="0 0 240 160" className="w-full max-w-sm" aria-hidden>
    {cells.map(({ i, risk, down }) => <rect key={i} x={(i % 12) * 20 + 2} y={Math.floor(i / 12) * 20 + 2} width="16" height="16" rx="3"
      fill={down ? 'rgb(var(--crit))' : fill[risk]} className={risk === 3 ? 'fa-pulse' : undefined} style={{ animationDelay: `${(i % 5) * 0.3}s` }} />)}
  </svg>;
}

export default function Login() {
  const { signIn, expired } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [show, setShow] = useState(false);
  const [caps, setCaps] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(expired ? 'Your session ended. Please sign in again.' : '');

  const submit = async (event) => {
    event.preventDefault();
    if (!username || !password) { setError('Enter your username and password.'); return; }
    setBusy(true); setError('');
    try { await signIn(username.trim(), password); }
    catch (e) { setError(e.message || 'Sign-in failed.'); setPassword(''); }
    finally { setBusy(false); }
  };
  const capsCheck = (e) => setCaps(Boolean(e.getModifierState?.('CapsLock')));

  return <div className="grid min-h-screen bg-bg text-ink lg:grid-cols-[1.1fr_1fr]">
    <aside className="relative hidden flex-col justify-between border-r border-line bg-surface p-10 lg:flex">
      <div className="flex items-center gap-2">
        <svg width="22" height="22" viewBox="0 0 16 16" aria-hidden><rect width="16" height="16" rx="3" fill="rgb(var(--accent))" /><path d="M3 10h2l2-5 2 7 2-4h2" stroke="white" strokeWidth="1.5" fill="none" /></svg>
        <span className="text-sm font-semibold">FaultAware</span><span className="text-xs text-ink3">Console</span>
      </div>
      <div className="fa-enter max-w-md space-y-6">
        <ClusterArt />
        <div>
          <h2 className="text-xl font-semibold leading-snug">Predict faults before they happen, then schedule around them.</h2>
          <p className="mt-2 text-sm leading-relaxed text-ink2">Machine-learning and deep-learning risk scores from HDFS block logs drive FARS, a fault-aware scheduler that migrates and checkpoints work away from nodes that are about to fail.</p>
        </div>
      </div>
      <p className="text-2xs text-ink3">BT-185 capstone · IILM University · Guide: Ms. Niharika Chaudhary</p>
    </aside>

    <main className="relative flex items-center justify-center p-6">
      <div className="absolute right-4 top-4"><ThemeToggle /></div>
      <form onSubmit={submit} noValidate className="fa-enter w-full max-w-sm space-y-5" aria-describedby={error ? 'login-error' : undefined}>
        <div>
          <div className="mb-4 inline-grid h-9 w-9 place-items-center rounded border border-line bg-surface text-accent"><Lock size={16} /></div>
          <h1 className="text-lg font-semibold">Sign in</h1>
          <p className="mt-1 text-xs text-ink3">Access is restricted to the BT-185 project team.</p>
        </div>

        {error && <div id="login-error" role="alert" className="fa-shake rounded border border-crit/30 bg-crit/10 px-3 py-2 text-xs text-ink">{error}</div>}

        <label className="block">
          <span className="text-xs font-medium text-ink2">Username</span>
          <input autoFocus autoComplete="username" spellCheck={false} maxLength={64} value={username} onChange={(e) => setUsername(e.target.value)}
            className="mt-1 w-full rounded border border-line bg-surface px-3 py-2 text-sm outline-none transition-colors focus:border-accent" />
        </label>

        <label className="block">
          <span className="text-xs font-medium text-ink2">Password</span>
          <div className="relative mt-1">
            <input type={show ? 'text' : 'password'} autoComplete="current-password" maxLength={256} value={password}
              onChange={(e) => setPassword(e.target.value)} onKeyUp={capsCheck} onKeyDown={capsCheck}
              className="w-full rounded border border-line bg-surface py-2 pl-3 pr-10 text-sm outline-none transition-colors focus:border-accent" />
            <button type="button" onClick={() => setShow(!show)} aria-label={show ? 'Hide password' : 'Show password'}
              className="absolute inset-y-0 right-0 grid w-9 place-items-center text-ink3 hover:text-ink">{show ? <EyeOff size={15} /> : <Eye size={15} />}</button>
          </div>
          {caps && <span className="mt-1 block text-2xs text-warn">Caps Lock is on.</span>}
        </label>

        <button type="submit" disabled={busy}
          className="flex w-full items-center justify-center gap-2 rounded border border-accent bg-accent px-3 py-2 text-sm font-medium text-white transition-[background-color,transform] hover:bg-accent/90 active:scale-[.99] disabled:cursor-wait disabled:opacity-70">
          {busy ? <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" aria-hidden /> : <LogIn size={15} />}
          {busy ? 'Signing in…' : 'Sign in'}
        </button>

        <div className="flex items-start gap-2 border-t border-line pt-4 text-2xs leading-relaxed text-ink3">
          <ShieldCheck size={13} className="mt-px shrink-0 text-ok" />
          <span>There is no self sign-up. Accounts are created by the administrator. Passwords are stored only as scrypt hashes; sessions use HttpOnly, SameSite=Strict cookies, and repeated failures lock sign-in for 15 minutes.</span>
        </div>
      </form>
    </main>
  </div>;
}
