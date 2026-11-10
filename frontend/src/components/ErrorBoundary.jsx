import { Component } from 'react';

export default class ErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { error: null }; }

  static getDerivedStateFromError(error) { return { error }; }

  render() {
    if (!this.state.error) return this.props.children;
    return <main className="grid min-h-screen place-items-center bg-bg p-6 text-ink">
      <div className="max-w-md rounded border border-line bg-surface p-6">
        <h1 className="text-sm font-semibold">The console hit an unexpected error</h1>
        <p className="mt-2 font-mono text-2xs text-ink3">{String(this.state.error?.message ?? this.state.error)}</p>
        <button onClick={() => window.location.reload()} className="mt-4 rounded border border-line px-3 py-1.5 text-xs hover:bg-sunken">Reload</button>
      </div>
    </main>;
  }
}
