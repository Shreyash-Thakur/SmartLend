import { useEffect, useState } from 'react';
import {
  signInWithPopup,
  signOut,
  onAuthStateChanged,
} from 'firebase/auth';
import { auth, firebaseAuthEnabled, firebaseConfigDiagnostics, provider } from '../lib/firebase';
import { useAuthStore } from '../store/authStore';

// Guest identity: a stable, browser-local user for deployments without
// Firebase (e.g. served from a raw IP, where Firebase Auth cannot authorize
// the domain). The uid persists in localStorage so "my applications" keeps
// pointing at the same applicant across visits.
const GUEST_USER_KEY = 'smartlend.guestUser';

interface StoredGuestUser {
  uid: string;
  email: string;
  displayName: string;
  photoURL: string | null;
}

const loadGuestUser = (): StoredGuestUser | null => {
  try {
    const raw = localStorage.getItem(GUEST_USER_KEY);
    return raw ? (JSON.parse(raw) as StoredGuestUser) : null;
  } catch {
    return null;
  }
};

export const useAuth = () => {
  const [loading, setLoading] = useState(true);
  const { setUser, user, setRole, role, logout } = useAuthStore();

  useEffect(() => {
    if (!firebaseAuthEnabled || !auth) {
      const guest = loadGuestUser();
      if (guest) {
        setUser(guest);
      }
      setLoading(false);
      return;
    }

    const unsubscribe = onAuthStateChanged(auth, (firebaseUser) => {
      if (firebaseUser) {
        setUser({
          uid: firebaseUser.uid,
          email: firebaseUser.email || '',
          displayName: firebaseUser.displayName || 'User',
          photoURL: firebaseUser.photoURL,
        });
      } else {
        setUser(null);
      }
      setLoading(false);
    });

    return () => unsubscribe();
  }, [setUser]);

  const loginWithGoogle = async () => {
    if (!firebaseAuthEnabled || !auth || !provider) {
      setLoading(false);
      const missingKeys = firebaseConfigDiagnostics.missingConfigKeys.join(', ');
      throw new Error(
        `Google sign-in is unavailable. Missing Firebase keys: ${missingKeys}. `
        + 'Set VITE_FIREBASE_* values in frontend/.env.',
      );
    }

    try {
      setLoading(true);
      const result = await signInWithPopup(auth, provider);
      setUser({
        uid: result.user.uid,
        email: result.user.email || '',
        displayName: result.user.displayName || 'User',
        photoURL: result.user.photoURL,
      });
      return result.user;
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const loginAsGuest = () => {
    const existing = loadGuestUser();
    const guest: StoredGuestUser = existing ?? {
      uid: `guest-${Math.random().toString(36).slice(2, 14)}`,
      email: 'guest@smartlend.local',
      displayName: 'Guest User',
      photoURL: null,
    };
    try {
      localStorage.setItem(GUEST_USER_KEY, JSON.stringify(guest));
    } catch {
      // storage unavailable (private mode): session-only guest still works
    }
    setUser(guest);
    setLoading(false);
    return guest;
  };

  const logout_user = async () => {
    try {
      localStorage.removeItem(GUEST_USER_KEY);
    } catch {
      // ignore
    }
    if (!firebaseAuthEnabled || !auth) {
      logout();
      return;
    }

    try {
      await signOut(auth);
      logout();
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  return {
    user,
    role,
    loading,
    isAuthenticated: !!user,
    loginWithGoogle,
    loginAsGuest,
    logout: logout_user,
    setRole,
  };
};
