import { useEffect, useState } from 'react';
import {
  signInWithPopup,
  signOut,
  User as FirebaseUser,
  onAuthStateChanged,
} from 'firebase/auth';
import { auth, provider } from '../lib/firebase';
import { useAuthStore } from '../store/authStore';

export const useAuth = () => {
  const [loading, setLoading] = useState(true);
  const { setUser, user, setRole, role, logout } = useAuthStore();

  useEffect(() => {
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
