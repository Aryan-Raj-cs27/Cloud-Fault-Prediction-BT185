import { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { fetchMe, login as apiLogin, logout as apiLogout } from './api.js';

const AuthContext = createContext(null);

/** Session state backed by the server's HttpOnly cookie (the token never touches JS). */
export function AuthProvider({ children }) {
  const [state, setState] = useState({ status: 'checking', user: null });

  useEffect(() => {
    let alive = true;
    fetchMe().then(({ data }) => alive && setState({ status: 'authed', user: data.username }))
      .catch(() => alive && setState({ status: 'anon', user: null }));
    const onUnauthorized = () => setState({ status: 'anon', user: null, expired: true });
    window.addEventListener('fa:unauthorized', onUnauthorized);
    return () => { alive = false; window.removeEventListener('fa:unauthorized', onUnauthorized); };
  }, []);

  const signIn = useCallback(async (username, password) => {
    const { data } = await apiLogin(username, password);
    setState({ status: 'authed', user: data.username });
  }, []);

  const signOut = useCallback(async () => {
    try { await apiLogout(); } finally { setState({ status: 'anon', user: null }); }
  }, []);

  return <AuthContext.Provider value={{ ...state, signIn, signOut }}>{children}</AuthContext.Provider>;
}

export const useAuth = () => useContext(AuthContext);
