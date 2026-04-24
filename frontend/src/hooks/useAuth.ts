import { useEffect, useState } from 'react';
import {
  signInWithPopup,
  signOut,
  onAuthStateChanged,
} from 'firebase/auth';
import { auth, firebaseAuthEnabled, firebaseConfigDiagnostics, provider } from '../lib/firebase';
import { useAuthStore } from '../store/authStore';

export const useAuth = () => {
  const [loading, setLoading] = useState(true);
  const { setUser, user, setRole, role, logout } = useAuthStore();

  useEffect(() => {
    if (!firebaseAuthEnabled || !auth) {
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

  const logout_user = async () => {
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
    logout: logout_user,
    setRole,
  };
};
