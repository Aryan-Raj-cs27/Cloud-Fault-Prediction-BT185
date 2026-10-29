import { Component } from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';

export default class ErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { error: null }; }
  static getDerivedStateFromError(error) { return { error }; }
  render() {
    if (this.state.error) return <main className="grid min-h-screen place-items-center bg-[#0b1020] p-6 text-slate-100"><div className="max-w-md rounded-2xl border border-white/10 bg-[#11182a] p-8 text-center"><span className="mx-auto grid h-12 w-12 place-items-center rounded-xl bg-rose-400/10 text-rose-300"><AlertTriangle /></span><h1 className="mt-4 text-lg font-semibold">Something went wrong</h1><p className="mt-2 text-sm text-slate-400">The dashboard hit an unexpected error. Reload the page to try again.</p><button onClick={() => window.location.reload()} className="mt-5 inline-flex items-center gap-2 rounded-lg bg-violet-500 px-4 py-2 text-sm font-semibold"><RotateCcw size={14} /> Reload dashboard</button></div></main>;
    return this.props.children;
  }
}
